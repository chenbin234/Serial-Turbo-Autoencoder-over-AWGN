"""Configuration, seeding, DDP, and loss utilities used by TurboAE training."""

import os
import pathlib
import random
import subprocess
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import torch
import torch.distributed
from torch import nn


def get_git_metadata() -> tuple[str, str]:
    """Return the current Git branch and commit hash."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    def run_git_command(args: list[str]) -> str:
        try:
            return subprocess.run(
                args,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "unknown"

    return (
        run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        run_git_command(["git", "rev-parse", "HEAD"]),
    )


class _LossConfig(Protocol):
    """Configuration values required by the loss factory."""

    loss_function: str


def resolve_device(local_rank: int = 0) -> torch.device:
    """
    Select the configured CUDA device, or CPU when CUDA is unavailable.

    Returns:
        The device on which training should run.

    """
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def ddp_setup(force_gloo: bool = False) -> None:
    """Set up distributed training with the selected PyTorch backend."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # NCCL requires CUDA. Keep the configured backend on CUDA and use gloo for CPU DDP.
    backend = "gloo" if force_gloo or not cuda_available else "nccl"
    print(f"Using {backend} backend for distributed training")

    try:
        torch.distributed.init_process_group(backend=backend)
        print(f"Distributed training initialized with {backend} backend")
    except Exception as error:  # noqa: BLE001
        print(f"Failed to initialize with {backend}: {error}")
        if backend == "nccl":
            print("Falling back to gloo backend")
            torch.distributed.init_process_group(backend="gloo")


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def broadcast_seed(base_seed: int | None = None) -> int:
    """
    Select a seed on rank zero and broadcast it to all distributed workers.

    Returns:
        The seed agreed upon by every worker.

    Raises:
        RuntimeError: If the distributed process group has not been initialized.

    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("Call torch.distributed.init_process_group first.")

    world_size = torch.distributed.get_world_size()
    print(f"world_size: {world_size}")
    if world_size <= 1:
        return base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

    rank = torch.distributed.get_rank()
    device = resolve_device(torch.cuda.current_device() if torch.cuda.is_available() else 0)
    if rank == 0:
        seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)
    else:
        seed = 0

    print(f"rank {rank} seed: {seed}")
    seed_tensor = torch.tensor([seed], dtype=torch.long, device=device)
    print(f"rank {rank} seed_tensor: {seed_tensor}")
    torch.distributed.broadcast(seed_tensor, src=0)
    print(f"rank {rank} seed_tensor after broadcast: {seed_tensor}")
    return int(seed_tensor.item())


def build_loss_fn(config: object) -> nn.Module:
    """
    Build the configured training-loss module.

    Returns:
        The selected training-loss module.

    Raises:
        ValueError: If the configured loss name is unsupported.

    """
    loss_config = cast("_LossConfig", config)
    loss_builders: dict[str, Callable[[], nn.Module]] = {
        "BCE": nn.BCELoss,
        "MSE": nn.MSELoss,
    }
    loss_name = loss_config.loss_function

    try:
        return loss_builders[str(loss_name)]()
    except KeyError as error:
        supported = ", ".join(loss_builders)
        raise ValueError(
            f"Invalid training loss {loss_name!r}; expected one of: {supported}.",
        ) from error
