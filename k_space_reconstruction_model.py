import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
from model import SubSamplingLayer
from data.transforms import fft2, ifft2
from trajectory_initiations import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KSpaceReconstruction(nn.Module):
    def __init__(self, number_of_conv_layers, unet_chans,
                 unet_num_pool_layers, unet_drop_prob):
        super().__init__()
        conv_layers = []
        channels = 2
        # The amount of input channels is 2 because of the ifft result
        for _ in range(number_of_conv_layers // 2):
            conv_layers.append(
                nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, bias=False, padding=1))
            conv_layers.append(nn.ReLU())
            channels *= 2
        for _ in range(number_of_conv_layers // 2, number_of_conv_layers):
            conv_layers.append(
                nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, bias=False, padding=1))
            conv_layers.append(nn.ReLU())
            channels //= 2
        self.K_space_reconstruction = nn.Sequential(*conv_layers)
        self.Unet_model = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
                                    num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)

    def forward(self, input_image):
        k_space = fft2(input_image)
        reconstructed_k_space = self.K_space_reconstruction(k_space.permute(0, 3, 1, 2))
        image = ifft2(reconstructed_k_space.permute(0, 2, 3, 1))
        reconstructed_image = self.Unet_model(image.permute(0, 3, 1, 2))
        return reconstructed_image


class SubSamplingModelReconFromKSpace(nn.Module):
    def __init__(self, decimation_rate, resolution, trajectory_learning,
                 subsampling_trajectory, spiral_density, unet_chans,
                 unet_num_pool_layers, unet_drop_prob, number_of_conv_layers):
        super().__init__()
        self.sub_sampling_layer = SubSamplingLayer(decimation_rate, resolution, trajectory_learning,
                                                   subsampling_trajectory, spiral_density)
        self.reconstruction_model = KSpaceReconstruction(number_of_conv_layers=number_of_conv_layers,
                                                         unet_chans=unet_chans,
                                                         unet_num_pool_layers=unet_num_pool_layers,
                                                         unet_drop_prob=unet_drop_prob)

    def forward(self, input_data):
        image_from_sub_sampling = self.sub_sampling_layer(input_data)
        output = self.reconstruction_model(image_from_sub_sampling.squeeze(1))
        return output

    def get_trajectory(self):
        return self.sub_sampling_layer.trajectory

    def get_reconstructed_k_space(self,input_data):
        image_from_sub_sampling = self.sub_sampling_layer(input_data)
        k_space = fft2(image_from_sub_sampling)
        output = self.reconstruction_model.K_space_reconstruction(k_space.squeeze(1).permute(0,3,1,2))
        return output
