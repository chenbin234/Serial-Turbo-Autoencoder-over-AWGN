"""
This file contains the implementation of the Turbo AutoEncoder model using PyTorch
Paper: Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels
URL: http://arxiv.org/abs/1911.03038
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy import arange
from numpy.random import mtrand
from torch import nn

# from models.components.interleavers import Interleaver, DeInterleaver
from src.channel.AWGN_Channel import AWGN_Channel_new
from src.models.components.SameShapeConv1d import SameShapeConv1d


def generate_permutation_array(k, seed=0):

    rand_gen = mtrand.RandomState(seed=seed)
    p_array = rand_gen.permutation(arange(k))

    p_array = torch.LongTensor(p_array).view(len(p_array))

    return p_array


# class Interleaver(nn.Module):
#     def __init__(self, p_array):
#         super(Interleaver, self).__init__()
#         self.p_array = torch.LongTensor(p_array).view(len(p_array))

#     def forward(self, inputs):

#         inputs = inputs.permute(1, 0, 2)
#         res = inputs[self.p_array]
#         res = res.permute(1, 0, 2)

#         return res


# class DeInterleaver(nn.Module):
#     def __init__(self, p_array):
#         super(DeInterleaver, self).__init__()

#         self.reverse_p_array = [0 for _ in range(len(p_array))]
#         for idx in range(len(p_array)):
#             self.reverse_p_array[p_array[idx]] = idx

#         self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

#     def forward(self, inputs):
#         inputs = inputs.permute(1, 0, 2)
#         res = inputs[self.reverse_p_array]
#         res = res.permute(1, 0, 2)

#         return res


class Encoder(nn.Module):
    def __init__(self, in_channels, enc_num_unit, kernel_size, enc_num_layer, p_array_seed=0):
        super().__init__()
        self.p_array_seed = p_array_seed
        self.enc_cnn_1 = SameShapeConv1d(
            in_channels=in_channels,
            out_channels=enc_num_unit,
            kernel_size=kernel_size,
            num_layers=enc_num_layer,
            stride=1,
            padding="same",
            bias=True,
        )
        self.enc_linear_1 = nn.Linear(in_features=enc_num_unit, out_features=1)
        self.enc_activation_1 = nn.ELU()

        self.enc_cnn_2 = SameShapeConv1d(
            in_channels=in_channels,
            out_channels=enc_num_unit,
            kernel_size=kernel_size,
            num_layers=enc_num_layer,
            stride=1,
            padding="same",
            bias=True,
        )
        self.enc_linear_2 = nn.Linear(in_features=enc_num_unit, out_features=1)
        self.enc_activation_2 = nn.ELU()

        self.enc_cnn_3 = SameShapeConv1d(
            in_channels=in_channels,
            out_channels=enc_num_unit,
            kernel_size=kernel_size,
            num_layers=enc_num_layer,
            stride=1,
            padding="same",
            bias=True,
        )
        self.enc_linear_3 = nn.Linear(in_features=enc_num_unit, out_features=1)
        self.enc_activation_3 = nn.ELU()

    def set_interleaver(self, inputs):
        # inputs has the shape (batch_size, k, 1)
        k = inputs.shape[1]

        rand_gen = mtrand.RandomState(seed=self.p_array_seed)
        p_array = rand_gen.permutation(arange(k))
        # print(p_array)

        # convert p_array from numpy array to torch tensor
        self.p_array = torch.as_tensor(p_array, dtype=torch.long, device=inputs.device).view(len(p_array))

        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.p_array]
        res = res.permute(1, 0, 2)

        return res

    def power_constraint(self, inputs):
        """
        Apply power constraint to the encoded tensor.

        Args:
            inputs (torch.Tensor): Encoded tensor after applying the Turbo AutoEncoder, with shape (batch_size, k, 3).

        Returns:
            torch.Tensor: Encoded tensor after applying power constraint, with shape (batch_size, k, 3).

        """
        # x has the shape (batch_size, k, 3)
        # Apply power constraint to the encoded tensor
        x_input_norm = (inputs - torch.mean(inputs)) * 1.0 / torch.std(inputs)

        return x_input_norm

    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, k, 1), where k is the length of the bit sequence.

        Returns:
            torch.Tensor: Encoded tensor after applying the Turbo AutoEncoder, with shape (batch_size, k, 3).

        The forward pass performs the following steps:
        1. Convert the input bit sequence from {0, 1} to {-1, 1}.
        2. Encode the systematic bit sequence using the first set of CNN, linear, and activation layers.
        3. Encode the first parity bit sequence using the second set of CNN, linear, and activation layers.
        4. Interleave the systematic bit sequence and encode the second parity bit sequence using the third set of CNN, linear, and activation layers.
        5. Concatenate the systematic and parity bit sequences along the last dimension.
        6. Apply power constraint to the concatenated tensor.

        """
        # x has the shape (batch_size, k, 1) which is the bit sequence to be encoded
        # k is the length of the bit sequence
        # Convert the input bit sequence from {0, 1} to {-1, 1}
        x = 2.0 * x - 1.0

        x_sys = self.enc_cnn_1(x)
        x_sys = self.enc_linear_1(x_sys)
        x_sys = self.enc_activation_1(x_sys)

        x_p1 = self.enc_cnn_2(x)
        x_p1 = self.enc_linear_2(x_p1)
        x_p1 = self.enc_activation_2(x_p1)

        x_sys_interleaved = self.set_interleaver(x)
        x_p2 = self.enc_cnn_3(x_sys_interleaved)
        x_p2 = self.enc_linear_3(x_p2)
        x_p2 = self.enc_activation_3(x_p2)

        # x_tx has the shape (batch_size, k, 3)
        x_tx = torch.cat([x_sys, x_p1, x_p2], dim=2)

        # Apply power constraint to the encoded tensor
        # codes has the shape (batch_size, k, 3)
        codes = self.power_constraint(x_tx)

        return codes


class Decoder(nn.Module):
    def __init__(
        self,
        num_iteration,
        num_iter_ft,
        dec_num_unit,
        kernel_size,
        dec_num_layer,
        p_array_seed=0,
    ):
        super().__init__()
        self.p_array_seed = p_array_seed

        self.num_iter_ft = num_iter_ft
        self.num_iteration = num_iteration

        self.dec1_cnns = nn.ModuleList()
        self.dec2_cnns = nn.ModuleList()
        self.dec1_outputs = nn.ModuleList()
        self.dec2_outputs = nn.ModuleList()

        for idx in range(num_iteration):
            self.dec1_cnns.append(
                SameShapeConv1d(
                    in_channels=2 + num_iter_ft,
                    out_channels=dec_num_unit,
                    kernel_size=kernel_size,
                    num_layers=dec_num_layer,
                    stride=1,
                    padding="same",
                    bias=True,
                )
            )
            self.dec1_outputs.append(nn.Linear(in_features=dec_num_unit, out_features=num_iter_ft))

            self.dec2_cnns.append(
                SameShapeConv1d(
                    in_channels=2 + num_iter_ft,
                    out_channels=dec_num_unit,
                    kernel_size=kernel_size,
                    num_layers=dec_num_layer,
                    stride=1,
                    padding="same",
                    bias=True,
                )
            )

            if idx == num_iteration - 1:
                self.dec2_outputs.append(nn.Linear(in_features=dec_num_unit, out_features=1))
            else:
                self.dec2_outputs.append(nn.Linear(in_features=dec_num_unit, out_features=num_iter_ft))

    def interleaver(self, inputs, p_array):
        # inputs has the shape (batch_size, k, 3)

        inputs = inputs.permute(1, 0, 2)
        res = inputs[p_array]
        res = res.permute(1, 0, 2)

        return res

    def deinterleaver(self, inputs, p_array):
        # inputs has the shape (batch_size, k, 3)

        reverse_p_array = torch.argsort(p_array).to(inputs.device)

        inputs = inputs.permute(1, 0, 2)
        res = inputs[reverse_p_array]
        res = res.permute(1, 0, 2)

        return res

    def forward(self, received):

        batch_size = received.shape[0]
        block_length = received.shape[1]
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed).to(received.device)

        # r_sys = received[:, :, 0].unsqueeze(2)
        # r_sys_interleaved = self.interleaver(r_sys, p_array)
        # r_p1 = received[:, :, 1].unsqueeze(2)
        # r_p2 = received[:, :, 2].unsqueeze(2)

        # prior = torch.zeros(batch_size, block_length, self.num_iter_ft).to(self.device)

        # for i in range(self.num_iteration - 1):

        #     # Decoder 1
        #     dec1_input = torch.cat([r_sys, r_p1, prior], dim=2)
        #     dec1_output = self.dec1_cnns[i](dec1_input)
        #     dec1_output = self.dec1_outputs[i](dec1_output)

        #     dec1_output = dec1_output - prior

        #     dec1_output_interleaved = self.interleaver(dec1_output, p_array)

        #     # Decoder 2
        #     dec2_input = torch.cat(
        #         [r_sys_interleaved, r_p2, dec1_output_interleaved], dim=2
        #     )
        #     dec2_output = self.dec2_cnns[i](dec2_input)
        #     dec2_output = self.dec2_outputs[i](dec2_output)

        #     dec2_output = dec2_output - dec1_output_interleaved

        #     # Update prior
        #     prior = self.deinterleaver(dec2_output, p_array)

        # # Final iteration
        # dec1_input = torch.cat([r_sys, r_p1, prior], dim=2)
        # dec1_output = self.dec1_cnns[self.num_iteration - 1](dec1_input)
        # dec1_output = self.dec1_outputs[self.num_iteration - 1](dec1_output)

        # dec1_output = dec1_output - prior

        # dec1_output_interleaved = self.interleaver(dec1_output, p_array)

        # dec2_input = torch.cat(
        #     [r_sys_interleaved, r_p2, dec1_output_interleaved], dim=2
        # )
        # dec2_output = self.dec2_cnns[self.num_iteration - 1](dec2_input)
        # dec2_output = self.dec2_outputs[self.num_iteration - 1](dec2_output)

        # # Final output, of shape (batch_size, k, 1)
        # final = torch.sigmoid(self.interleaver(dec2_output, p_array))

        # Turbo Decoder
        r_sys = received[:, :, 0].view((batch_size, block_length, 1))
        r_sys_int = self.interleaver(r_sys, p_array)
        r_par1 = received[:, :, 1].view((batch_size, block_length, 1))
        r_par2 = received[:, :, 2].view((batch_size, block_length, 1))

        # num_iteration,
        prior = received.new_zeros((batch_size, block_length, self.num_iter_ft))

        for idx in range(self.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

            x_dec = self.dec1_cnns[idx](x_this_dec)
            x_plr = self.dec1_outputs[idx](x_dec)

            x_plr = x_plr - prior

            x_plr_int = self.interleaver(x_plr, p_array)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

            x_dec = self.dec2_cnns[idx](x_this_dec)

            x_plr = self.dec2_outputs[idx](x_dec)

            x_plr = x_plr - x_plr_int

            prior = self.deinterleaver(x_plr, p_array)

        # last round
        x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

        x_dec = self.dec1_cnns[self.num_iteration - 1](x_this_dec)
        x_plr = self.dec1_outputs[self.num_iteration - 1](x_dec)

        x_plr = x_plr - prior

        x_plr_int = self.interleaver(x_plr, p_array)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

        x_dec = self.dec2_cnns[self.num_iteration - 1](x_this_dec)
        x_plr = self.dec2_outputs[self.num_iteration - 1](x_dec)

        final = torch.sigmoid(self.deinterleaver(x_plr, p_array))

        return final


class Turbo_AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_num_unit,
        kernel_size,
        enc_num_layer,
        num_iteration,
        num_iter_ft,
        dec_num_unit,
        dec_num_layer,
        p_array_seed=0,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, enc_num_unit, kernel_size, enc_num_layer, p_array_seed)
        self.decoder = Decoder(
            num_iteration,
            num_iter_ft,
            dec_num_unit,
            kernel_size,
            dec_num_layer,
            p_array_seed,
        )

    def forward(self, x, SNR_db_low, SNR_db_high):
        x = self.encoder(x)
        x = AWGN_Channel_new(x, SNR_db_low, SNR_db_high)
        x = self.decoder(x)

        return x
