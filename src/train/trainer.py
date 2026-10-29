"""Training and evaluation utilities for the TurboAE model."""

import os
from typing import Literal, cast

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import wandb
from src.inference import inference_loop
from src.train.utils import resolve_device

TrainingMode = Literal["encoder", "decoder"]


class Trainer:
    """Train, validate, and run inference for a TurboAE model."""

    def __init__(
        self,
        config: wandb.sdk.wandb_config.Config,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_encoder: Optimizer,
        optimizer_decoder: Optimizer,
    ) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", config.workstation_GPU_index))
        self.global_rank = int(os.environ.get("RANK", config.workstation_GPU_index))
        self.device = resolve_device(self.local_rank)
        self.config = config
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder

        # Wrap the model with DDP when needed. An inference-only run produces no
        # gradients to synchronize, so it uses the unwrapped model and only
        # communicates model state and evaluation statistics during inference.
        if self.config.use_ddp and not getattr(self.config, "is_inference", False):
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )
            else:
                self.model = DistributedDataParallel(
                    self.model,
                    find_unused_parameters=True,
                )

    @staticmethod
    def _debug_assert_finite(name: str, tensor: torch.Tensor, mode: str) -> None:
        """Raise an informative error when a tensor contains NaN or infinity."""
        # Return immediately in the normal case. Computing detailed statistics can
        # be expensive and is only useful when a non-finite value has been detected.
        if torch.isfinite(tensor).all().item():
            return

        # Diagnostics must not become part of the autograd graph, so inspect a
        # detached view while leaving the original tensor and its gradients intact.
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = finite_mask.sum().item()
        total_count = detached.numel()
        nonfinite_count = total_count - finite_count

        # Min, max, and mean are undefined when every value is NaN or infinity.
        # In that case, report only the tensor shape and finite-value counts.
        if finite_count == 0:
            statistics = (
                f"shape={tuple(detached.shape)}, "
                f"finite=0/{total_count}, nonfinite={nonfinite_count}"
            )
        else:
            # Exclude NaN and infinite entries so they do not contaminate the
            # summary statistics of the remaining valid values.
            finite_values = detached[finite_mask]
            statistics = (
                f"shape={tuple(detached.shape)}, "
                f"finite={finite_count}/{total_count}, "
                f"nonfinite={nonfinite_count}, "
                f"min={finite_values.min().item():.6g}, "
                f"max={finite_values.max().item():.6g}, "
                f"mean={finite_values.mean().item():.6g}"
            )

        raise FloatingPointError(f"{name} became non-finite during {mode}: {statistics}")

    def _prepare_bit_predictions_for_loss(
        self, predictions: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """
        Validate and clamp predicted bit probabilities before loss evaluation.

        BCE-based losses evaluate logarithms of the predicted probabilities, so
        their inputs are kept away from exactly zero and one. Other losses only
        require predictions to remain in the valid probability range ``[0, 1]``.

        Args:
            predictions: Tensor containing predicted probabilities for each bit.
            mode: Current training or evaluation mode, used in error diagnostics.

        Returns:
            The predictions clamped to the range required by the configured loss.

        Raises:
            FloatingPointError: If ``predictions`` contains NaN or infinity.

        """
        # Fail early with diagnostic information; clamping cannot repair NaN or
        # infinite values and would otherwise hide where numerical instability began.
        self._debug_assert_finite("bit predictions before loss", predictions, mode)

        # Loss names containing "bce" expect probabilities. Keep them inside an
        # epsilon margin to avoid evaluating log(0) in binary cross-entropy.
        loss_function = str(getattr(self.config, "loss_function", "")).lower()
        if "bce" in loss_function:
            eps = float(getattr(self.config, "bce_epsilon", 1e-7))

            # Constrain a user-provided epsilon to a valid interval. The upper
            # limit must stay below 0.5 so the lower bound remains below the upper.
            eps = max(0.0, min(eps, 0.5 - 1e-12))
            return torch.clamp(predictions, eps, 1.0 - eps)

        # Non-BCE losses do not need an epsilon margin, but predictions are still
        # probabilities and therefore must remain within their mathematical range.
        return torch.clamp(predictions, 0.0, 1.0)

    def _clip_and_check_gradients(self, mode: str, *optimizers: Optimizer) -> bool:
        """Validate and clip gradients before the supplied optimizers step.

        ``True`` tells the caller that it is safe to call ``optimizer.step()``.
        ``False`` means a non-finite gradient was found and all participating
        optimizers were cleared, so the current update must be skipped. When
        ``skip_nonfinite_gradients`` is disabled, the original error is raised
        instead of returning ``False``.

        Gradient checking and clipping are both disabled when
        ``config.grad_clip_norm`` is unset or non-positive.

        Args:
            mode: Training mode used to identify errors in diagnostics.
            *optimizers: Optimizers participating in the current update. Passing
                multiple optimizers keeps joint updates all-or-nothing.

        Returns:
            Whether the caller may safely apply the optimizer step.

        """
        # A missing or non-positive threshold explicitly disables this safeguard.
        max_norm = getattr(self.config, "grad_clip_norm", None)
        if max_norm is None or max_norm <= 0:
            return True

        # Optimizer groups can include frozen or currently unused parameters.
        # Only tensors populated by backward() need validation and clipping.
        parameters = [
            param
            for optimizer in optimizers
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not parameters:
            return True

        # Inspect each gradient separately first so any failure identifies the
        # offending parameter instead of reporting only a non-finite total norm.
        for param_index, param in enumerate(parameters):
            gradient = param.grad
            if gradient is None:
                continue
            try:
                self._debug_assert_finite(f"gradient for parameter {param_index}", gradient, mode)
            except FloatingPointError as error:
                # A non-finite gradient can either skip this optimizer step or
                # stop training, according to the configured recovery policy.
                if getattr(self.config, "skip_nonfinite_gradients", True):
                    print(f"Skipping {mode} optimizer step because {error}")
                    # Clear every optimizer involved in a joint update; retaining
                    # any gradients could leak this invalid batch into the next one.
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    return False
                raise

        # clip_grad_norm_ treats all collected gradients as one vector. It scales
        # them in place only when their combined norm exceeds max_norm and returns
        # the norm measured before clipping.
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=float(max_norm), error_if_nonfinite=True
            )
        except RuntimeError as error:
            if getattr(self.config, "skip_nonfinite_gradients", True):
                print(
                    f"Skipping {mode} optimizer step because gradient norm "
                    f"became non-finite: {error}"
                )
                for optimizer in optimizers:
                    optimizer.zero_grad()
                return False
            raise

        # Keep this final assertion as a defensive check on the value returned by
        # PyTorch, even though error_if_nonfinite normally catches this condition.
        self._debug_assert_finite("gradient norm", total_norm, mode)
        return True

    def _generate_data_epoch(self) -> DataLoader[torch.Tensor]:
        train_data = torch.randint(
            low=0,
            high=2,
            size=(self.config.num_samples, self.config.k, 1),
            dtype=torch.float32,
        ).to(self.device)
        # Tensor implements the Dataset protocol, despite the narrower torch stub.
        train_dataset = cast("Dataset[torch.Tensor]", train_data)
        if self.config.use_ddp:
            return DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.config.seed),
                sampler=DistributedSampler(train_dataset),
            )
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator().manual_seed(self.config.seed),
            shuffle=True,
        )

    def _run_batch(self, X: torch.Tensor, mode: TrainingMode) -> float:
        """Update the encoder or decoder through the AWGN model."""
        if mode == "encoder":
            optimizer = self.optimizer_encoder
            snr_low = self.config.train_enc_snr_low
            snr_high = self.config.train_enc_snr_high
        elif mode == "decoder":
            optimizer = self.optimizer_decoder
            snr_low = self.config.train_dec_snr_low
            snr_high = self.config.train_dec_snr_high
        else:
            raise ValueError(f"AWGN training supports encoder and decoder modes, got {mode!r}.")

        self.model.zero_grad()
        predictions = self.model(X, SNR_db_low=snr_low, SNR_db_high=snr_high)
        predictions = self._prepare_bit_predictions_for_loss(predictions, mode)
        loss = self.loss_fn(predictions, X)
        self._debug_assert_finite("loss", loss, mode)
        loss.backward()
        if self._clip_and_check_gradients(mode, optimizer):
            optimizer.step()
        return loss.item()

    def _run_epoch(self, mode: TrainingMode) -> float:
        # generate data for each epoch
        X_train = self._generate_data_epoch()
        # set model to training mode
        self.model.train()
        # ! update the learning rate
        for param_group in self.optimizer_encoder.param_groups:
            param_group["lr"] = self.config.learning_rate
        for param_group in self.optimizer_decoder.param_groups:
            param_group["lr"] = self.config.learning_rate

        train_loss_batches = []
        # iterate over the data
        for X in X_train:
            train_loss_batches.append(self._run_batch(X.to(self.device), mode))
        return sum(train_loss_batches) / len(train_loss_batches)

    def _save_snapshot(self, epoch: int) -> None:
        # Saving model
        print(f"Saving model [epoch {epoch}]")
        if self.config.use_ddp:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": cast("torch.nn.Module", self.model.module).state_dict(),
                },
                f"./trained_models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                },
                f"./trained_models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )

    def val_TurboAE_AWGN(self) -> tuple[float, float, float]:
        """Validate AWGN bit predictions at the encoder's lower SNR."""
        self.model.eval()
        loss_total = 0.0
        bit_errors = 0
        block_errors = 0
        num_samples = 0
        num_bits = 0
        with torch.no_grad():
            for X in self._generate_data_epoch():
                X = X.to(self.device)
                predictions = self.model(
                    X,
                    SNR_db_low=self.config.train_enc_snr_low,
                    SNR_db_high=self.config.train_enc_snr_low,
                )
                predictions = self._prepare_bit_predictions_for_loss(predictions, "validation")
                loss = self.loss_fn(predictions, X)
                self._debug_assert_finite("loss", loss, "validation")
                loss_total += loss.item() * X.shape[0]
                errors = torch.round(predictions) != X
                bit_errors += int(errors.sum().item())
                block_errors += int(errors.reshape(X.shape[0], -1).any(dim=1).sum().item())
                num_samples += X.shape[0]
                num_bits += X.numel()
        return loss_total / num_samples, bit_errors / num_bits, block_errors / num_samples

    def train_TurboAE_AWGN(self) -> None:
        """Alternate encoder/decoder training and adapt batch size and learning rate."""
        wandb.watch(self.model, self.loss_fn, log="all", log_freq=5000)
        wandb.config.update({"trainable_parameters": count_parameters(self.model)}, allow_val_change=True)
        print(f"Number of total parameters:     {total_parameters(self.model)}")
        print(f"Number of trainable parameters: {count_parameters(self.model)}")
        print("Starting TurboAE training over AWGN")
        stuck_counter = 0
        lowest_cost = float("inf")
        max_batch_size = 4000
        min_learning_rate = 1e-6

        for epoch in range(self.config.start_epoch, self.config.epochs + 1):
            train_losses = []
            for _ in range(self.config.num_train_enc):
                train_losses.append(self._run_epoch(mode="encoder"))
            for _ in range(self.config.num_train_dec):
                train_losses.append(self._run_epoch(mode="decoder"))
            train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            val_loss, val_ber, val_bler = self.val_TurboAE_AWGN()
            if self.config.use_ddp:
                # Every rank must make the same scheduling decision so future
                # epochs retain matching batch counts and optimizer settings.
                validation_metrics = torch.tensor([val_loss, val_ber, val_bler], dtype=torch.float64, device=self.device)
                torch.distributed.all_reduce(validation_metrics, op=torch.distributed.ReduceOp.SUM)
                validation_metrics /= torch.distributed.get_world_size()
                val_loss, val_ber, val_bler = validation_metrics.tolist()

            if val_loss > lowest_cost:
                stuck_counter += 1
            else:
                stuck_counter = 0
                lowest_cost = val_loss

            if stuck_counter > self.config.max_stuck_counter:
                if self.config.batch_size < max_batch_size:
                    new_batch_size = min(self.config.batch_size * 2, max_batch_size)
                    # Keep the training workload proportional to the batch
                    # increase, including a partial increase at the cap.
                    new_num_samples = self.config.num_samples * new_batch_size // self.config.batch_size
                    print(f"Loss stuck for {stuck_counter} epochs: increase batch size to {new_batch_size}.")
                    wandb.config.update(
                        {"batch_size": new_batch_size, "num_samples": new_num_samples},
                        allow_val_change=True,
                    )
                    stuck_counter = 0
                elif self.config.learning_rate > min_learning_rate:
                    new_learning_rate = max(self.config.learning_rate / 10, min_learning_rate)
                    print(f"Loss stuck for {stuck_counter} epochs: reduce learning rate to {new_learning_rate:.6e}.")
                    wandb.config.update({"learning_rate": new_learning_rate}, allow_val_change=True)
                    for optimizer in (self.optimizer_encoder, self.optimizer_decoder):
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = new_learning_rate
                    stuck_counter = 0

            print(
                f"Epoch: {epoch:>2}/{self.config.epochs:<2} | "
                f"Train loss: {train_loss:>10.6f} | Val. loss: {val_loss:>10.6f} | "
                f"Val. BER ({self.config.train_enc_snr_low} dB): {val_ber:>10.6f} | "
                f"Val. BLER ({self.config.train_enc_snr_low} dB): {val_bler:>10.6f} | "
                f"Batch size: {self.config.batch_size:>4} | "
                f"Learning rate: {self.config.learning_rate:>10.6e} | "
                f"Stuck: {stuck_counter:>3}"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_BER": val_ber,
                    "val_BLER": val_bler,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "stuck_counter": stuck_counter,
                },
                step=epoch,
            )
            if (not self.config.use_ddp or self.global_rank == 0) and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

    def inference(self, test_dataloader: DataLoader[object]) -> None:
        """Run inference over a test data loader."""
        inference_loop(
            config=self.config,
            model=self.model,
            test_dataloader=test_dataloader,
            device=self.device,
        )


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of total parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters())
