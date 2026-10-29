"""Serial turbo autoencoder model and its encoder/decoder building blocks."""

from typing import cast

import torch
from numpy import arange
from numpy.random import mtrand
from torch import nn

import wandb
from src.channel.AWGN_Channel import AWGN_Channel_complex
from src.models import MySign


def generate_permutation_array(k: int, seed: int = 0) -> torch.Tensor:
    """
    Return the deterministic interleaver permutation for a block of length ``k``.

    Returns:
        A one-dimensional tensor containing the permuted indices.

    """
    rand_gen = mtrand.RandomState(seed=seed)
    p_array = rand_gen.permutation(arange(k))

    p_array = torch.LongTensor(p_array).view(len(p_array))

    return p_array


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional stack to ``x``.

        Returns:
            The transformed feature tensor.

        """
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # Pass the result through the output layer
        x = self.output_layer(x)

        return x


class Encoder(nn.Module):
    """Encode binary messages into complex-valued channel symbols."""

    def __init__(
        self,
        num_features: int,
        p_array_seed: int,
        enforce_qpsk_symbols: bool = False,
    ) -> None:
        super().__init__()
        self.p_array_seed = p_array_seed
        self.enforce_qpsk_symbols = enforce_qpsk_symbols

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
            x (torch.Tensor): Real I/Q pairs of shape (batch_size, n, 2).

        Returns:
            torch.Tensor: Power-constrained tensor of shape (batch_size, n, 2).

        """
        power = x.square().sum(dim=-1, keepdim=True)

        eps = torch.finfo(power.dtype).eps
        return x * torch.rsqrt(power.clamp_min(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of binary messages.

        Returns:
            Encoded real-valued in-phase and quadrature components.

        """
        block_length = x.shape[1]  # k
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed)

        x = 2.0 * x - 1.0

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, 1, k)

        # outer encoder input size = (batch_size, 1, k), output size = (batch_size, F, k)
        x = self.outer_encoder(x)

        # quantization to 2 levels
        x = MySign.apply(x)

        # interleaver
        x = self.set_interleaver(x, p_array)  # output size = (batch_size, F, n)

        # inner encoder
        x = self.inner_encoder(x)  # output size = (batch_size, 2, k)

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, n, 2)

        if self.enforce_qpsk_symbols:
            x = MySign.apply(x)

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
        num_iteration: int,
        p_array_seed: int,
        num_of_pair: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(num_of_pair, bool) or not isinstance(num_of_pair, int) or num_of_pair < 1:
            raise ValueError("num_of_pair must be a positive integer.")

        self.n = n
        self.F = num_features
        self.num_iteration = num_iteration
        self.num_of_pair = num_of_pair
        self.p_array_seed = p_array_seed

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

    def forward(self, y_received: torch.Tensor) -> torch.Tensor:
        """
        Decode received channel samples of shape (batch_size, n, 2) into bit probabilities.

        Returns:
            A tensor containing the decoded bit probabilities.

        """
        device = y_received.device
        y = y_received.permute(0, 2, 1)  # output size = (batch_size, 2, n)

        block_length = self.n
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed)

        # initiate i_c, i_c_pri is all zeros with the shape (batch_size, F, n)
        i_c_pri = torch.zeros((y.shape[0], self.F, self.n), dtype=torch.float32, device=device)

        # Run every pair's feedback cycles before advancing to the next pair.
        for pair_index in range(self.num_of_pair):
            for _ in range(self.num_iteration):
                # Concatenate the received signal with the feedback.
                y_concatenate = torch.cat((y, i_c_pri), dim=1)

                # Equalize and exchange extrinsic information.
                i_c = self.eq_block_list[pair_index](y_concatenate)  # output size = (batch_size, F, n)
                i_c = i_c - i_c_pri
                i_c = self.deinterleaver(i_c, p_array)

                i_b = self.dec_block_list[pair_index](i_c)
                i_c_pri = i_b - i_c
                i_c_pri = self.interleaver(i_c_pri, p_array)

        # ! last iteration
        # EQ-CNN last block

        # concatenate y with i_c_pri
        # y is of shape (batch_size, 2, n), i_c_pri shape (batch_size, F, n)
        y_concatenate = torch.cat((y, i_c_pri), dim=1)  # output size = (batch_size, 2+F, n)

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
        self.num_of_pair = cast("int", autoencoder_config["num_of_pair"])
        self.p_array_seed = cast("int", autoencoder_config["p_array_seed"])
        self.enforce_qpsk_symbols = cast("bool", autoencoder_config["enforce_qpsk_symbols"])
        self.encoder_output_power_constraint = cast("str", autoencoder_config["encoder_output_power_constraint"])

        self.encoder = Encoder(
            num_features=self.F,
            p_array_seed=self.p_array_seed,
            enforce_qpsk_symbols=self.enforce_qpsk_symbols,
        )
        # The serial encoder emits k I/Q symbols.
        self.decoder = Decoder(
            k=self.k,
            n=self.k,
            num_features=self.F,
            num_iteration=self.num_iteration,
            num_of_pair=self.num_of_pair,
            p_array_seed=self.p_array_seed,
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
