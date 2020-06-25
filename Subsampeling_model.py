import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
import torch.nn.functional as F
from trajectory_initiations import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sampler(nn.Module):
    def __init__(self, resolution, decimation_rate, number_of_decisions, conv_channels,
                 number_of_conv_layers, number_of_linear_layers):
        super().__init__()
        self.resolution = resolution
        self.decimation_rate = decimation_rate
        self.number_of_samples = (resolution ** 2) // decimation_rate
        self.number_of_decisions = number_of_decisions
        self.len_of_decision_vector = self.number_of_samples // number_of_decisions
        # correction if not full numbers
        self.number_of_samples = number_of_decisions * self.len_of_decision_vector

        # self.NN = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
        #                                       num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)

        conv_layers = [
            nn.Sequential(nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1),
                          nn.MaxPool2d(2)) for _ in range(number_of_conv_layers - 2)]
        conv_layers = [nn.Conv2d(in_channels=2, out_channels=conv_channels, kernel_size=3, padding=1)] + conv_layers
        conv_layers = conv_layers + [nn.MaxPool2d(2)] + [nn.Conv2d(in_channels=conv_channels, out_channels=1, kernel_size=3, padding=1)]

        self.convolutions = nn.Sequential(*conv_layers)
        number_of_features = (resolution ** 2) // (4 ** (number_of_conv_layers - 1))
        linear_layers = [
            nn.Linear(in_features=number_of_features // (2 ** i), out_features=number_of_features // (2 ** (i + 1)))
            for i in range(number_of_linear_layers - 1)]
        linear_layers = linear_layers + [
            nn.Linear(in_features=number_of_features // (2 ** (number_of_linear_layers - 1)),
                      out_features=self.len_of_decision_vector*2)]
        self.fully_connected = nn.Sequential(*linear_layers)

    def forward(self, k_space):
        sample_mask = self.convolutions(k_space.permute(0, 3, 1, 2))
        sample_mask = sample_mask.reshape(sample_mask.shape[0], -1)
        sample_vector = self.fully_connected(sample_mask)
        # put the indexes of the vector between -1 and 1 for gird sample
        sample_vector = F.tanh(sample_vector)
        sample_vector = sample_vector.reshape(sample_mask.shape[0],-1,2)
        return sample_vector
