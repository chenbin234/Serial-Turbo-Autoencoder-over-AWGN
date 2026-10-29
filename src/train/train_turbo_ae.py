"""Train and evaluate the serial Turbo AutoEncoder over the AWGN channel."""

# import libraries
import argparse
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch
from torch import nn, optim
from torch.distributed import barrier, broadcast_object_list, destroy_process_group, get_rank
from torch.utils.data import DataLoader

import wandb
from src.features import InfobitDataset
from src.inference.inference_turboae import make_inference_dataloader
from src.models import Turbo_AutoEncoder, generate_permutation_array
from src.train.trainer import Trainer
from src.train.utils import (
    broadcast_seed,
    build_loss_fn,
    ddp_setup,
    get_git_metadata,
    resolve_device,
    setup_seed,
)


@dataclass
class TurboAEConfig:
    """TurboAE hyperparameters with their default values."""

    run_name: str | None = "unknown"
    git_branch: str = "unknown"
    git_commit_hash: str = "unknown"
    epochs: int = 800
    num_samples: int = 5000
    batch_size: int = 5000
    start_epoch: int = 1
    save_every: int = 800
    k: int = 64  # number of information bits; the encoder emits k complex symbols
    F: int = 10
    num_iteration: int = 6  # Decoder feedback iterations per EQ/DEC pair
    num_of_pair: int = 1  # Independently parameterized EQ/DEC pairs; 1 shares one pair across all iterations
    num_train_enc: int = 100  # Number of iterations for training the encoder in each epoch
    num_train_dec: int = 500  # Number of iterations for training the decoder in each epoch
    train_enc_snr_high: float = 4.0  # dB
    train_enc_snr_low: float = 4.0  # dB
    train_dec_snr_high: float = 4.0  # dB
    train_dec_snr_low: float = 0.5  # dB
    test_snr_high: float = 6  # dB
    test_snr_low: float = 1  # dB
    test_snr_interval: float = 1.0  # dB
    test_data_samples: int = 10_000_000  # Total test samples per SNR across all workers.
    pretrained: str | None = (
        "./trained_models/Turbo_AutoEncoder_k64n64_iter6_pair1_20261027_10_51_27/Turbo_AutoEncoder_20261027_10_51_27_epoch800.pth"  # Path to the pretrained model; None starts from random weights
    )
    is_inference: bool = True  # Skip training entirely and only run inference on "pretrained".
    loss_function: str = "BCE"  # "BCE" or "MSE"
    bce_epsilon: float = 1e-7  # Avoid exact 0/1 probabilities in BCELoss.
    grad_clip_norm: float = 1.0  # Prevent rare exploding gradients from poisoning weights.
    skip_nonfinite_gradients: bool = True
    learning_rate: float = 1e-4
    save_model_name: str = ""
    use_ddp: bool = False
    workstation_GPU_index: int = 0
    max_stuck_counter: int = 49
    encoder_output_power_constraint: str = "per_block"  # "per_symbol" or "per_block" or "none"
    enforce_qpsk_symbols: bool = (
        False  # if true, enforce the encoder output to be in the QPSK constellation; if false, allow the encoder to learn any constellation
    )
    p_array_seed: int = 6
    seed: int = 0


def make(
    config: wandb.sdk.wandb_config.Config,
) -> tuple[Turbo_AutoEncoder, nn.Module, optim.Optimizer, optim.Optimizer, DataLoader]:
    """
    Build the model, loss, optimizers, and test data loader.

    Returns:
        The model and all components required for training and evaluation.

    Raises:
        ValueError: If an inference-only run has no pretrained checkpoint.

    """
    # An inference-only run never updates the weights, so it is only meaningful
    # when a trained checkpoint is supplied.
    if config.is_inference and not config.pretrained:
        raise ValueError("Inference-only runs (is_inference=True) require 'pretrained' to point to a checkpoint.")

    # test dataset
    test_dataset = InfobitDataset(num_samples=config.test_data_samples, k=config.k)
    test_dataloader = make_inference_dataloader(test_dataset, batch_size=config.batch_size, use_ddp=config.use_ddp)

    # Initialize the model
    model = Turbo_AutoEncoder(autoencoder_config=config)

    print("p_array_seed:", config.p_array_seed)
    print("p_array:", generate_permutation_array(k=config.k, seed=config.p_array_seed))

    if config.pretrained:
        device = resolve_device(int(os.environ.get("LOCAL_RANK", config.workstation_GPU_index)))
        checkpoint = torch.load(config.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained model loaded")

    loss_fn = build_loss_fn(config)

    # Make the optimizers
    optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=config.learning_rate)
    optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=config.learning_rate)

    return (
        model,
        loss_fn,
        optimizer_encoder,
        optimizer_decoder,
        test_dataloader,
    )


def model_pipeline(hyperparameters: TurboAEConfig) -> Turbo_AutoEncoder:
    """
    Train and evaluate a model with the supplied hyperparameters.

    When ``is_inference`` is set, training is skipped and only the pretrained
    model is evaluated.

    Returns:
        The trained model, or the pretrained model for an inference-only run.

    """
    # Only rank zero creates an online W&B run. Disabled runs still provide a
    # local wandb.config object to the other DDP workers.
    wandb_mode = "disabled" if hyperparameters.use_ddp and get_rank() != 0 else "online"
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=asdict(hyperparameters),
        name=hyperparameters.run_name,
        group="serial_turbo_autoencoder",
        mode=wandb_mode,
        save_code=True,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        (
            model,
            loss_fn,
            optimizer_encoder,
            optimizer_decoder,
            test_dataloader,
        ) = make(config)
        print(model)

        # and use them to train the model
        trainer = Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
        )
        # Inference-only runs evaluate the pretrained weights loaded in make().
        if config.is_inference:
            print("Inference-only run: skipping training.")
        else:
            trainer.train_TurboAE_AWGN()

        if config.use_ddp:
            # Every worker evaluates its test-data shard. Inference synchronizes
            # model state once and combines statistics at the end of each SNR.
            barrier()
            trainer.inference(test_dataloader=test_dataloader)
            barrier()
            destroy_process_group()
        else:
            trainer.inference(test_dataloader=test_dataloader)

        # finish logging to wandb
        wandb.finish()

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(description="Train the Turbo AutoEncoder")
    parser.add_argument(
        "--p-array-seed",
        type=int,
        default=6,
        help="Seed used to generate the interleaver permutation array.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Global random seed used for this training run.",
    )
    # With store_true, argparse sets the attribute to False by default and to True when the flag appears.
    parser.add_argument(
        "--use-ddp",
        action="store_true",
        help="Enable DistributedDataParallel; launch with torchrun when using this option.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure and run Turbo AutoEncoder training."""
    args = parse_args()

    # update the configuration for the model
    git_branch, git_commit_hash = get_git_metadata()

    config = TurboAEConfig(
        run_name=os.environ.get("SLURM_JOB_NAME"),
        git_branch=git_branch,
        git_commit_hash=git_commit_hash,
        p_array_seed=args.p_array_seed,
        seed=args.seed,
        use_ddp=args.use_ddp,
    )

    if config.use_ddp:
        ddp_setup()

    # Only the process that owns the W&B run needs credentials. Offline and
    # disabled modes intentionally do not contact the W&B service.
    is_primary = not config.use_ddp or get_rank() == 0
    # fetch the WANDB_MODE from the environment, defaulting to "online" if not set.
    wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
    if is_primary and wandb_mode not in {"disabled", "offline"}:
        wandb.login()

    # get the current time for the model name
    time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H_%M_%S")
    if config.use_ddp:
        synchronized_time = [time if get_rank() == 0 else ""]
        broadcast_object_list(synchronized_time, src=0)
        time = synchronized_time[0]
    # defining model save location
    save_model_name = "Turbo_AutoEncoder_" + time
    # save the model name in the config for later use
    config.save_model_name = save_model_name

    # create the folder if it doesn't exist
    if config.use_ddp:
        # agree on a base-seed
        base_seed = broadcast_seed(config.seed)
        setup_seed(base_seed)
        if get_rank() == 0 and not config.is_inference:
            pathlib.Path(f"./trained_models/{save_model_name}").mkdir(exist_ok=True, parents=True)
        barrier()
    else:
        setup_seed(config.seed)
        # create the folder if it doesn't exist; an inference-only run saves nothing
        if not config.is_inference and not pathlib.Path(f"./trained_models/{save_model_name}").exists():
            pathlib.Path(f"./trained_models/{save_model_name}").mkdir(exist_ok=True, parents=True)

    # Build, train and analyze the model with the pipeline
    model_pipeline(config)


if __name__ == "__main__":
    main()
