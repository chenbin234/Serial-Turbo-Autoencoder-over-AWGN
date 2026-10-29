"""Serial turbo autoencoder model and its encoder/decoder building blocks."""

from typing import Literal, cast

import torch
from numpy import arange
from numpy.random import mtrand
from torch import nn
from torch.nn import functional

import wandb
from src.channel import RayleighBlockFadingChannel
from src.channel.AWGN_Channel import AWGN_Channel_complex
from src.models import DelayEstimator, MySign, SoftQPSK


def generate_permutation_array(k: int, seed: int = 0) -> torch.Tensor:
    """
    Return the deterministic interleaver permutation for a block of length ``k``.

    Returns:
        A one-dimensional tensor containing the permuted indices.

    """
    if seed == 100:
        p_array = torch.tensor([
            8,
            48,
            56,
            29,
            36,
            22,
            15,
            4,
            63,
            44,
            54,
            31,
            23,
            16,
            9,
            0,
            42,
            61,
            33,
            24,
            51,
            14,
            6,
            41,
            58,
            34,
            27,
            50,
            20,
            7,
            43,
            62,
            35,
            28,
            55,
            17,
            2,
            47,
            10,
            40,
            32,
            25,
            59,
            3,
            52,
            11,
            45,
            18,
            38,
            60,
            1,
            53,
            26,
            46,
            19,
            12,
            37,
            5,
            57,
            30,
            49,
            21,
            13,
            39,
        ])
    else:
        rand_gen = mtrand.RandomState(seed=seed)
        p_array = rand_gen.permutation(arange(k))

        p_array = torch.LongTensor(p_array).view(len(p_array))

    return p_array


def add_padding(input_tensor: torch.Tensor, n_m: int) -> torch.Tensor:
    """
    Pads the input tensor with zeros on the left and right along the last dimension.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, F, n) with dtype torch.float32.
        n_m (int): Number of zeros to pad on both the left and the right sides of the last dimension.

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, F, n + 2*n_M).

    """
    # F.pad expects the padding dimensions in the order: (pad_left, pad_right, pad_top, pad_bottom, ...)
    # Since we need to pad the last dimension (n), we set pad=(n_M, n_M)
    # padded_tensor = F.pad(input_tensor, pad=(n_M, n_M - 1), mode="constant", value=0)
    padded_tensor = functional.pad(
        input_tensor,
        pad=(n_m, n_m),
        mode="constant",
        value=0,
    )

    return padded_tensor


def remove_padding(padded_tensor: torch.Tensor, n_m: int) -> torch.Tensor:
    """
    Removes the padding (n_M zeros) from the left and right along the last dimension.

    Args:
        padded_tensor (torch.Tensor): Padded tensor of shape (batch_size, F, n + 2*n_M).
        n_m (int): Number of padded zeros on each end of the last dimension.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, F, n) after removing padding.

    """
    # We remove the first n_M and last n_M elements along the last dimension.
    # Assumes padded_tensor.shape[-1] >= 2*n_M.

    if n_m > 0:
        return padded_tensor[..., n_m:-n_m]
    return padded_tensor


class CNNBlock(nn.Module):
    """Stack of convolutional layers used by the encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int, padding: str = "same") -> None:
        super().__init__()

        # Define the sequence of convolutional layers
        # input size = (batch_size, in_channels, k), output size = (batch_size, n, k)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
        )

        # Define the output layer
        # input size = (batch_size, n, k), output size = (batch_size, F, k)
        self.output_layer = nn.Conv1d(in_channels=100, out_channels=out_channels, kernel_size=1)
        # self.output_layer = nn.Linear(in_features=100, out_features=F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional stack to ``x``.

        Returns:
            The transformed feature tensor.

        """
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # convert the second and third dimension of the tensor
        # x = x.permute(0, 2, 1)

        # Pass the result through the output layer
        x = self.output_layer(x)

        return x


class Encoder(nn.Module):
    """Encode binary messages into complex-valued channel symbols."""

    def __init__(
        self,
        num_features: int,
        p_array_seed: int,
        add_parity_check_bit: bool = False,
        enforce_qpsk_symbols: bool = False,
        is_soft_qpsk: bool = False,
        soft_qpsk_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if enforce_qpsk_symbols and is_soft_qpsk:
            raise ValueError("enforce_qpsk_symbols and is_soft_qpsk are mutually exclusive; enable at most one.")
        if is_soft_qpsk and not 0.0 <= soft_qpsk_alpha <= 1.0:
            raise ValueError(f"soft_qpsk_alpha must lie in [0, 1], got {soft_qpsk_alpha}.")
        self.p_array_seed = p_array_seed
        self.add_parity_check_bit = add_parity_check_bit
        self.enforce_qpsk_symbols = enforce_qpsk_symbols
        self.is_soft_qpsk = is_soft_qpsk
        self.soft_qpsk_alpha = soft_qpsk_alpha

        self.outer_encoder = CNNBlock(in_channels=1, out_channels=num_features)
        self.inner_encoder = CNNBlock(in_channels=num_features, out_channels=2)

    @staticmethod
    def set_interleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Permute the symbol dimension using ``p_array``.

        Returns:
            The permuted feature tensor.

        """
        # inputs has the shape (batch_size, k, 1)
        inputs = inputs.permute(2, 0, 1)  # output size = (n, batch_size, F)
        res = inputs[p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    @staticmethod
    def power_constraint(x: torch.Tensor) -> torch.Tensor:
        """
        Apply power constraint to the input tensor x.

        This function normalizes each input symbol to have power 1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n) with dtype torch.complex64.

        Returns:
            torch.Tensor: Power-constrained tensor of shape (batch_size, n) with dtype torch.complex64.

        """
        if torch.is_complex(x):
            power = x.real.square() + x.imag.square()
        elif x.dim() > 2 and x.size(-1) == 2:
            power = x.square().sum(dim=-1, keepdim=True)
        else:
            power = x.square()

        eps = torch.finfo(power.dtype).eps
        return x * torch.rsqrt(power.clamp_min(eps))

    @staticmethod
    def add_even_parity_bit(x: torch.Tensor) -> torch.Tensor:
        """
        Adds an even parity check bit along dimension 1 of a binary tensor.

        Args:
            x (torch.Tensor): Binary tensor of shape (batch, k, 1) with values 0 or 1.

        Returns:
            torch.Tensor: Tensor of shape (batch, k+1, 1) with even parity bit appended.

        Raises:
            ValueError: If ``x`` contains values other than zero and one.

        """
        # Ensure it's binary 0/1
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError("Input tensor must contain only 0s and 1s.")

        # Sum along dimension 1 (keepdim=True to maintain shape for broadcasting)
        ones_count = x.sum(dim=1)  # shape: (batch, 1)

        # Even parity bit: 1 if sum is odd, else 0
        parity_bit = (ones_count % 2).to(x.dtype)  # 1 if odd, 0 if even

        # Append parity bit along dim=1
        parity_bit = parity_bit.unsqueeze(1)  # shape: (batch, 1, 1)
        return torch.cat([x, parity_bit], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of binary messages.

        Returns:
            Encoded real-valued in-phase and quadrature components.

        """
        if self.add_parity_check_bit:
            x = Encoder.add_even_parity_bit(x)

        block_length = x.shape[1]  # k
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed)

        x = 2.0 * x - 1.0

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, 1, k)

        # outer encoder input size = (batch_size, 1, k), output size = (batch_size, F, k)
        x = self.outer_encoder(x)

        # quantization to 2 levels
        # x = STEQuantize.apply(x, 1.0, 2)
        x = MySign.apply(x)

        # interleaver
        x = self.set_interleaver(x, p_array)  # output size = (batch_size, F, n)

        # inner encoder
        x = self.inner_encoder(x)  # output size = (batch_size, 2, k)

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, n, 2)

        if self.add_parity_check_bit:
            # input size = (batch_size, n+1, 2), output size = (batch_size, n, 2)
            # simply remove the last element along dimension 1
            x = x[:, :-1, :]

        if self.enforce_qpsk_symbols:
            x = MySign.apply(x)
        elif self.is_soft_qpsk:
            # pull each symbol toward, but not onto, its nearest normalized QPSK point
            x = SoftQPSK.apply(x, self.soft_qpsk_alpha)

        return x


class EqualizerCNNBlock(nn.Module):
    """Convolutional equalizer block."""

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__()

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply equalization to ``x``.

        Returns:
            The equalized feature tensor.

        """
        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        return x


class DecoderCNNBlock(nn.Module):
    """Intermediate convolutional decoder block."""

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__()

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)
        self.cnn_block2 = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode one iteration of feature messages.

        Returns:
            Updated decoder features.

        """
        # input size = (batch_size, F, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DecoderCNNLastBlock(nn.Module):
    """Final convolutional decoder block producing bit probabilities."""

    def __init__(
        self,
        in_channels: int,
        k: int,
        num_features: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.F = num_features
        self.output_size = output_size

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)

        # input size = (batch_size, F, n), output size = (batch_size, F*n)
        self.flatten = nn.Flatten()

        self.decoder_ouput = nn.Linear(
            in_features=num_features * k,
            out_features=output_size,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return decoded bit probabilities for ``x``.

        Returns:
            A tensor of decoded bit probabilities.

        """
        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        x = self.cnn_block1(x)

        # flattten layer
        # input size = (batch_size, M1, n), output size = (batch_size, M1*n)
        x = self.flatten(x)

        # final decoder output
        # input size = (batch_size, M1*n), output size = (batch_size, k)
        x = self.decoder_ouput(x)

        # add sigmoid activation function
        x = self.sigmoid(x)

        # add one dimension to the output (second dimension) to make sure the output has the shape (batch_size, 1, k)
        x = x.unsqueeze(1)

        return x


class Decoder(nn.Module):
    """
    Iteratively equalize and decode received channel samples.

    ``num_of_pair`` sets the number of independently parameterized EQ/DEC pairs.
    Each pair shares its weights across ``num_iteration`` full feedback cycles,
    carrying the feedback into the next pair. The last pair's equalizer is then
    reused once with ``dec_last_block`` to produce the final bit probabilities.
    With ``num_of_pair=1``, all cycles and the final equalization share one EQ.
    """

    def __init__(
        self,
        k: int,
        n: int,
        num_features: int,
        n_m: int,
        upsampling_factor: int,
        num_iteration: int,
        p_array_seed: int,
        add_parity_check_bit: bool,
        num_of_pair: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(num_of_pair, bool) or not isinstance(num_of_pair, int) or num_of_pair < 1:
            raise ValueError("num_of_pair must be a positive integer.")

        self.k = k
        self.n = n
        self.F = num_features
        self.n_M = n_m
        self.N_up = upsampling_factor
        self.num_iteration = num_iteration
        self.num_of_pair = num_of_pair
        self.p_array_seed = p_array_seed
        self.add_parity_check_bit = add_parity_check_bit

        # Each pair owns distinct weights, reused for its feedback iterations.
        self.eq_block_list = nn.ModuleList()
        self.dec_block_list = nn.ModuleList()
        for _ in range(num_of_pair):
            self.eq_block_list.append(
                EqualizerCNNBlock(
                    in_channels=2 + num_features,
                    num_features=num_features,
                )
            )

            self.dec_block_list.append(
                DecoderCNNBlock(
                    in_channels=num_features,
                    num_features=num_features,
                )
            )

        # input size = (batch_size, F, n), output size = (batch_size, 1, k)
        self.dec_last_block = DecoderCNNLastBlock(
            in_channels=num_features,
            k=k,
            num_features=num_features,
            output_size=k,
        )

    @staticmethod
    def interleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Apply an interleaver permutation to feature messages.

        Returns:
            The permuted feature tensor.

        """
        # inputs has the shape (batch_size, F, n)

        inputs = inputs.permute(2, 0, 1)  # output size = (n, batch_size, F)
        res = inputs[p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    @staticmethod
    def deinterleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Invert an interleaver permutation for feature messages.

        Returns:
            The restored feature tensor.

        """
        # inputs has the shape (batch_size, F, n)

        reverse_p_array = torch.argsort(p_array)

        inputs = inputs.permute(2, 0, 1)  # input size = (n, batch_size, F)
        res = inputs[reverse_p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    # pylint: disable-next=too-many-locals
    def forward(self, y_delay_removed: torch.Tensor) -> torch.Tensor:
        """
        Decode a received, delay-corrected signal into bit probabilities.

        Returns:
            A tensor containing the decoded bit probabilities.

        Raises:
            ValueError: If the received tensor has an incompatible shape.

        """
        device = y_delay_removed.device
        y = y_delay_removed.permute(0, 2, 1)  # output size = (batch_size, 2, n)

        block_length = self.n
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed)

        # initiate i_c, i_c_pri is all zeros with the shape (batch_size, F, n)
        i_c_pri = torch.zeros((y.shape[0], self.F, self.n), dtype=torch.float32, device=device)

        # Run every pair's feedback cycles before advancing to the next pair.
        for pair_index in range(self.num_of_pair):
            for _ in range(self.num_iteration):
                # Pad feedback to match the received signal's symbol length.
                y_concatenate = torch.cat((y, i_c_pri), dim=1)

                # Equalize, crop padding, and exchange extrinsic information.
                i_c = self.eq_block_list[pair_index](y_concatenate)  # output size = (batch_size, F, n)
                i_c = i_c - i_c_pri
                i_c = self.deinterleaver(i_c, p_array)

                i_b = self.dec_block_list[pair_index](i_c)
                i_c_pri = i_b - i_c
                i_c_pri = self.interleaver(i_c_pri, p_array)

        # ! last iteration
        # EQ-CNN last block

        # concatenate y with i_c_pri
        # y is of shape (batch_size, n, n+2n_M), i_c_pri shape (batch_size, F, k+2n_M)
        y_concatenate = torch.cat((y, i_c_pri), dim=1)  # output size = (batch_size, n+F, n+2n_M)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        i_c = self.eq_block_list[-1](y_concatenate)

        # update i_c
        i_c = i_c - i_c_pri

        # deinterleaver
        i_c = self.deinterleaver(i_c, p_array)

        # DEC-CNN last block
        # input size = (batch_size, F, k), output size = (batch_size, 1, k)
        y_decoded = self.dec_last_block(i_c)

        # switch the second and third dimension of the tensor, output size = (batch_size, 1, k)
        y_decoded = y_decoded.permute(0, 2, 1)

        return y_decoded


class Turbo_AutoEncoder(nn.Module):
    """End-to-end serial turbo autoencoder for the AWGN channel."""

    def __init__(self, autoencoder_config: wandb.sdk.wandb_config.Config) -> None:
        super().__init__()

        self.k = cast("int", autoencoder_config["k"])
        self.F = cast("int", autoencoder_config["F"])
        self.num_iteration = cast("int", autoencoder_config["num_iteration"])
        self.num_of_pair = cast("int", autoencoder_config.get("num_of_pair", 1))
        self.p_array_seed = cast("int", autoencoder_config["p_array_seed"])
        self.add_parity_check_bit = cast("bool", autoencoder_config["add_parity_check_bit"])
        self.enforce_qpsk_symbols = cast("bool", autoencoder_config["enforce_qpsk_symbols"])
        self.is_soft_qpsk = cast("bool", autoencoder_config["is_soft_qpsk"])
        self.soft_qpsk_alpha = cast("float", autoencoder_config["soft_qpsk_alpha"])
        self.encoder_output_power_constraint = cast("str", autoencoder_config["encoder_output_power_constraint"])

        if self.add_parity_check_bit:
            raise ValueError("The serial AWGN model does not support add_parity_check_bit=True.")

        self.encoder = Encoder(
            num_features=self.F,
            p_array_seed=self.p_array_seed,
            add_parity_check_bit=self.add_parity_check_bit,
            enforce_qpsk_symbols=self.enforce_qpsk_symbols,
            is_soft_qpsk=self.is_soft_qpsk,
            soft_qpsk_alpha=self.soft_qpsk_alpha,
        )
        # The serial encoder emits k I/Q symbols; AWGN needs no padding or upsampling.
        self.decoder = Decoder(
            k=self.k,
            n=self.k,
            num_features=self.F,
            n_m=0,
            upsampling_factor=1,
            num_iteration=self.num_iteration,
            num_of_pair=self.num_of_pair,
            p_array_seed=self.p_array_seed,
            add_parity_check_bit=self.add_parity_check_bit,
        )

    def forward(self, x: torch.Tensor, SNR_db_low: float, SNR_db_high: float) -> torch.Tensor:
        """
        Encode, transmit over AWGN, and decode a batch of binary messages.

        With unit average complex symbol energy, SNR bounds represent Es/N0,
        where N0 = 2 * sigma**2, with variance sigma**2 per I/Q component.

        Returns:
            Decoded bit probabilities with shape (batch_size, k, 1).

        """
        x = self.encoder(x)  # output size = (batch_size, k, 2)

        if self.encoder_output_power_constraint == "per_symbol":
            x = self.encoder.power_constraint(x)
        elif self.encoder_output_power_constraint == "per_block":
            energies = x.square().sum(dim=2, keepdim=True).mean(dim=1, keepdim=True)
            x = x / torch.sqrt(energies.clamp_min(1e-12))

        x = AWGN_Channel_complex(x, SNR_db_low, SNR_db_high)
        x = self.decoder(x)

        return x


class TurboAutoencoderSync(nn.Module):
    """End-to-end serial turbo autoencoder with synchronization block."""

    def __init__(self, autoencoder_config: wandb.sdk.wandb_config.Config) -> None:
        super().__init__()

        self.k = cast("int", autoencoder_config["k"])
        self.n = cast("int", autoencoder_config["n"])
        self.n_M = cast("int", autoencoder_config["n_M"])
        self.F = cast("int", autoencoder_config["F"])
        self.channel = cast("str", autoencoder_config["channel"])
        self.is_delay = cast("bool", autoencoder_config["is_delay"])
        self.num_iteration = cast("int", autoencoder_config["num_iteration"])
        self.num_of_pair = cast("int", autoencoder_config.get("num_of_pair", 1))
        self.p_array_seed = cast("int", autoencoder_config["p_array_seed"])
        self.is_train_sync_block = cast("bool", autoencoder_config["is_train_sync_block"])
        self.independent_delay_per_block = cast("bool", autoencoder_config["independent_delay_per_block"])
        self.nb = cast("int", autoencoder_config["nb"])
        self.tp = cast("int", autoencoder_config["tp"])
        self.ts = cast("int", autoencoder_config["ts"])
        self.N = cast("int", autoencoder_config["N_up"])
        self.dmax = cast("int", autoencoder_config["dmax"])
        self.ns = int(self.n / self.nb)
        self.add_parity_check_bit = cast("bool", autoencoder_config["add_parity_check_bit"])
        self.enforce_qpsk_symbols = cast("bool", autoencoder_config["enforce_qpsk_symbols"])
        self.is_soft_qpsk = cast("bool", autoencoder_config["is_soft_qpsk"])
        self.soft_qpsk_alpha = cast("float", autoencoder_config["soft_qpsk_alpha"])
        self.encoder_output_power_constraint = cast("str", autoencoder_config["encoder_output_power_constraint"])
        self.receiver_resample_method = cast("str", autoencoder_config["receiver_resample_method"])
        self.receiver_resample_sample_phase = cast("float", autoencoder_config["receiver_resample_sample_phase"])
        self.receiver_mmse_regularization = cast("float", autoencoder_config["receiver_mmse_regularization"])
        self.sync_hidden_channels = cast("int", autoencoder_config["sync_hidden_channels"])
        self.sync_num_layers = cast("int", autoencoder_config["sync_num_layers"])
        self.sync_kernel_size = cast("int", autoencoder_config["sync_kernel_size"])
        self.sync_activation = cast("str", autoencoder_config["sync_activation"])
        self.sync_fuse_blocks = cast("bool", autoencoder_config["sync_fuse_blocks"])
        self.sync_separate_q_eps = cast("bool", autoencoder_config["sync_separate_q_eps"])

        self.encoder = Encoder(
            num_features=self.F,
            p_array_seed=self.p_array_seed,
            add_parity_check_bit=self.add_parity_check_bit,
            enforce_qpsk_symbols=self.enforce_qpsk_symbols,
            is_soft_qpsk=self.is_soft_qpsk,
            soft_qpsk_alpha=self.soft_qpsk_alpha,
        )
        if self.is_train_sync_block:
            self.blockfadingchannel = RayleighBlockFadingChannel(
                self.nb,
                self.tp,
                self.ts,
                self.N,
                self.dmax,
                self.is_delay,
                self.receiver_mmse_regularization,
            )
            self.sync_block = DelayEstimator(
                hidden_channels=self.sync_hidden_channels,
                num_layers=self.sync_num_layers,
                kernel_size=self.sync_kernel_size,
                activation=self.sync_activation,
                fuse_blocks=self.sync_fuse_blocks,
                dmax=self.dmax,
                ts=self.ts,
                separate_q_eps=self.sync_separate_q_eps,
            )

        self.decoder = Decoder(
            k=self.k,
            n=self.n,
            num_features=self.F,
            n_m=self.n_M,
            upsampling_factor=self.N,
            num_iteration=self.num_iteration,
            num_of_pair=self.num_of_pair,
            p_array_seed=self.p_array_seed,
            add_parity_check_bit=self.add_parity_check_bit,
        )

    def forward(
        self,
        x: torch.Tensor,
        delay_true: torch.Tensor,
        SNR_db_low: float,
        SNR_db_high: float,
        mode: Literal[
            "encoder",
            "decoder",
            "sync_encoder",
            "sync_block",
            "joint_encoder",
            "joint_decoder",
        ] = "encoder",
        return_sync_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Run the encoder, channel, and the receiver stages selected by ``mode``.

        Returns:
            Decoded bits, predicted delay, or both as selected by ``mode``.

        Raises:
            NotImplementedError: If the configured channel is unsupported.
            ValueError: If ``mode`` is not supported.

        """
        # Retained for API compatibility; joint modes already return both components.
        _ = return_sync_components
        # Transmitter part
        # input size = (batch_size, 1, k), output size = (batch_size, n, 2)
        x = self.encoder(x)
        # Make sure the tensor is contiguous in memory before view_as_complex
        x = x.contiguous()
        # convert x from real to complex tensor, output size = (batch_size, n)
        x = torch.view_as_complex(x)  # output size = (batch_size, n)

        if self.encoder_output_power_constraint == "per_symbol":
            x_transmitted = self.encoder.power_constraint(x)
        elif self.encoder_output_power_constraint == "per_block":
            x_transmitted = normalize_per_fading_block(x, self.nb, self.ns)
        else:
            x_transmitted = x

        if self.channel == "block_fading":
            x, h_true = self.blockfadingchannel.forward(x_transmitted, delay_true, SNR_db_low, SNR_db_high)  # output size = (batch_size, nb, M)

            # normalised each block of the received signal to have unit power before feeding into the sync block
            # energies = torch.mean(torch.abs(x) ** 2, dim=2, keepdim=True)
            # energies = torch.clamp(energies, min=1e-12)
            # x = x / torch.sqrt(energies)  # shape (batch_size, nb, M)

            if mode in {"encoder", "decoder"}:
                x_resampled = self.blockfadingchannel.resample(
                    x_transmitted,
                    h_true,
                    SNR_db_low,
                    SNR_db_high,
                )  # output size = (batch_size, nb, (n / nb) * N)
                # normalised each block of the sampled signal to have unit power before feeding into the decoder block
                energies = torch.mean(torch.abs(x_resampled) ** 2, dim=2, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                x_resampled = x_resampled / torch.sqrt(energies)  # shape (batch_size, nb, (n / nb) * N)
                # convert x_resampled from complex to real tensor, output size = (batch_size, nb, (n / nb) * N, 2)
                x_resampled = torch.view_as_real(x_resampled)  # output size = (batch_size, nb, (n / nb) * N, 2)

                # feed the resampled signal to the decoder
                predicted_bits = self.decoder(x_resampled)
                return predicted_bits
            if mode in {"sync_encoder", "sync_block"}:
                # Feed to the sync block. If block fusion is disabled, reduce
                # per-block estimates back to the shared-delay shape expected by
                # the trainer: (batch_size, 1).
                predicted_delay = self.sync_block(x)
                return predicted_delay
            if mode in {"joint_encoder", "joint_decoder"}:
                # Feed to the sync block. If block fusion is disabled, reduce
                # per-block estimates back to the shared-delay shape expected by
                # the trainer: (batch_size, 1).
                predicted_delay = self.sync_block(x)
                # Receiver-side timing correction must use only the sampled
                # receive waveform and the estimated delay.
                x_resampled = self.blockfadingchannel.receiver_resample(
                    x,
                    predicted_delay,
                    method=self.receiver_resample_method,
                    sample_phase=self.receiver_resample_sample_phase,
                )  # output size = (batch_size, nb, (n / nb) * N)
                # normalised each block of the sampled signal to have unit power before feeding into the decoder block
                energies = torch.mean(torch.abs(x_resampled) ** 2, dim=2, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                x_resampled = x_resampled / torch.sqrt(energies)  # shape (batch_size, nb, (n / nb) * N)

                # convert x_resampled from complex to real tensor, output size = (batch_size, nb, (n / nb) * N, 2)
                x_resampled = torch.view_as_real(x_resampled)  # output size = (batch_size, nb, (n / nb) * N, 2)

                # feed the resampled signal to the decoder
                predicted_bits = self.decoder(x_resampled)

                return predicted_delay, predicted_bits
            raise ValueError(f"Unsupported forward mode: {mode}.")
        raise NotImplementedError


def normalize_per_fading_block(x: torch.Tensor, nb: int, ns: int) -> torch.Tensor:
    """
    Normalize each fading block in ``x`` to unit average power.

    Returns:
        The normalized signal with its original two-dimensional shape.

    """
    x_blocks = x.reshape(x.shape[0], nb, ns)
    energies = torch.mean(torch.abs(x_blocks) ** 2, dim=2, keepdim=True)
    energies = torch.clamp(energies, min=1e-12)
    return (x_blocks / torch.sqrt(energies)).reshape(x.shape[0], -1)
