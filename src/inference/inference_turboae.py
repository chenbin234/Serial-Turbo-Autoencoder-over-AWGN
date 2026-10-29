"""Run TurboAE inference and calculate error statistics."""

from collections.abc import Iterable, Sized
from itertools import chain
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset

import wandb


def make_inference_dataloader(dataset: Dataset[torch.Tensor], batch_size: int, use_ddp: bool = False) -> DataLoader[torch.Tensor]:
    """
    Split test samples across ranks without padding or dropping any samples.

    Unlike a training DistributedSampler, this permits unequal shard lengths,
    including empty shards when there are fewer samples than workers.

    Returns:
        A loader for this rank's samples, or the full dataset without DDP.

    """
    if use_ddp:
        indices = range(dist.get_rank(), len(cast("Sized", dataset)), dist.get_world_size())
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def inference_loop(
    config: wandb.sdk.wandb_config.Config,
    model: torch.nn.Module,
    test_dataloader: Iterable[object],
    device: torch.device,
) -> torch.nn.Module:
    """
    Evaluate local test samples and combine statistics across all DDP workers.

    DDP callers must supply disjoint data shards and call this on every rank.
    Forward passes use the unwrapped model, so unequal batch counts are safe.
    Only rank zero prints and logs the combined results.

    Returns:
        The evaluated, unwrapped model.

    Raises:
        ValueError: If the global dataset is empty.

    """
    use_ddp = bool(getattr(config, "use_ddp", False))
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()

    if use_ddp:
        # Match rank zero's saved model, including BatchNorm running statistics.
        # No DDP forward-time broadcasts are needed after this synchronization.
        with torch.no_grad():
            for tensor in chain(model.parameters(), model.buffers()):
                dist.broadcast(tensor, src=0)
        # Channel noise must use distinct streams on different ranks.
        torch.manual_seed(int(config.seed) + rank)

    inference_snr_db = np.arange(config.test_snr_low, config.test_snr_high + 0.5, config.test_snr_interval)
    bit_error = int(config.k)
    infer_ber_list: list[float] = []
    infer_bler_list: list[float] = []
    if rank == 0:
        print(f"Starting inference on {world_size} worker(s) ...", flush=True)
        wandb.define_metric("custom_step")
        wandb.define_metric("BER v.s. SNR (AWGN)", step_metric="custom_step")
        wandb.define_metric("BLER v.s. SNR (AWGN)", step_metric="custom_step")

    with torch.no_grad():
        for snr_db in inference_snr_db:
            # These accumulators have fixed size regardless of the sample count.
            error_histogram = torch.zeros(bit_error + 1, dtype=torch.int64, device=device)

            for batch_index, data in enumerate(test_dataloader, 1):
                X = cast("torch.Tensor", data).to(device)
                batch_size = X.shape[0]
                predicted_bits = model(X, SNR_db_low=snr_db, SNR_db_high=snr_db)
                predictions = torch.clamp(predicted_bits, 0.0, 1.0)
                predictions_round = torch.round(predictions)

                errors_per_sample = (predictions_round != X).reshape(batch_size, -1).sum(dim=1)
                error_histogram += torch.bincount(errors_per_sample, minlength=bit_error + 1)

                if rank == 0 and batch_index % 1000 == 0:
                    print(f"SNR {snr_db}: rank 0 processed {batch_index} local batches.", flush=True)

            if use_ddp:
                # All ranks participate even when their shard has zero samples.
                dist.all_reduce(error_histogram, op=dist.ReduceOp.SUM)

            histogram = error_histogram.cpu().tolist()
            num_samples = sum(histogram)
            if num_samples == 0:
                raise ValueError("Inference requires at least one test sample across all ranks.")
            # Weight every sample equally, including partial batches and uneven shards.
            bit_errors = sum(errors * count for errors, count in enumerate(histogram))
            block_errors = num_samples - histogram[0]
            ber = bit_errors / (num_samples * bit_error)
            bler = block_errors / num_samples
            infer_ber_list.append(ber)
            infer_bler_list.append(bler)

            if rank == 0:
                print(
                    f"For SNR = {snr_db}, the average BER is {ber}, the average BLER is {bler} ({num_samples} samples across {world_size} workers)",
                    flush=True,
                )
                wandb.log({
                    "custom_step": float(snr_db),
                    "BER v.s. SNR (AWGN)": ber,
                    "BLER v.s. SNR (AWGN)": bler,
                    "Inference samples": num_samples,
                    "Inference bit errors": bit_errors,
                    "Inference block errors": block_errors,
                    "Inference bit error distribution": wandb.Table(
                        data=[[errors, count] for errors, count in enumerate(histogram)],
                        columns=["Bit errors", "Count"],
                    ),
                })

    if rank == 0:
        print("The snr_db values for inference are: ", inference_snr_db)
        print("The average bit error rate is: ", [round(value, 7) for value in infer_ber_list])
        print("The average block error rate is: ", [round(value, 7) for value in infer_bler_list])
        table_ber = wandb.Table(
            data=[[float(snr), ber] for snr, ber in zip(inference_snr_db, infer_ber_list)],
            columns=["SNR", "BER"],
        )
        table_bler = wandb.Table(
            data=[[float(snr), bler] for snr, bler in zip(inference_snr_db, infer_bler_list)],
            columns=["SNR", "BLER"],
        )
        wandb.log({
            "BER": wandb.plot_table(vega_spec_name="BER_new", data_table=table_ber, fields={"x": "SNR", "y": "BER"}),
            "BLER": wandb.plot_table(vega_spec_name="BLER_new", data_table=table_bler, fields={"x": "SNR", "y": "BLER"}),
        })
        print("Finished inference!", flush=True)
    return model
