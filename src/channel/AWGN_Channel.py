"""Complex AWGN channel for unit-energy I/Q symbols."""

import torch


def AWGN_Channel_complex(transmitter_signal, SNR_db_low, SNR_db_high):
    """
    Add complex AWGN to unit-energy symbols represented by real I/Q pairs.

    For unequal bounds, sample a noise standard deviation per complex symbol
    and share it between I and Q. The bounds are mixed uniformly in standard
    deviation rather than uniformly in dB.

    Args:
        transmitter_signal (torch.Tensor): Real tensor of shape (batch_size, n, 2),
            with average complex symbol energy E[I**2 + Q**2] = 1.
        SNR_db_low (float): Lower bound of Es/N0 in dB, where N0 = 2 * sigma**2.
        SNR_db_high (float): Upper bound of Es/N0 in dB.

    Returns:
        torch.Tensor: Received I/Q pairs with the same shape as the input.

    """
    # Each real component has variance N0/2 for unit complex symbol energy.
    sigma_low = 10 ** (-SNR_db_low / 20) / (2 ** 0.5)
    sigma_high = 10 ** (-SNR_db_high / 20) / (2 ** 0.5)

    this_sigma = (
        torch.rand_like(transmitter_signal[..., :1]) * (sigma_low - sigma_high)
        + sigma_high
    )
    noise = this_sigma * torch.randn_like(transmitter_signal)

    output_signal = transmitter_signal + noise

    return output_signal
