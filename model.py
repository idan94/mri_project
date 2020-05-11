import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import utils
from unet_model import UnetModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)

class SubSamplingLayer(nn.Module):

    # This one gives us FULL mask
    def get_init_trajectory_full(self):
        index = 0
        every_point = torch.zeros(self.resolution * self.resolution, 2)
        for i in range(self.resolution):
            for j in range(self.resolution):
                every_point[index] = torch.tensor([i, j])
                index += 1
        every_point = every_point - (0.5 * self.resolution)
        every_point = every_point.to(device)
        return every_point

    def get_init_trajectory_random_uniform(self):
        x = (torch.rand(self.num_measurements, 2) - 0.5) * self.resolution
        return x

    def __init__(self, decimation_rate, resolution, trajectory_learning: bool):
        super().__init__()
        self.decimation_rate = decimation_rate
        self.resolution = resolution
        self.num_measurements = resolution ** 2 // decimation_rate
        self.trajectory = torch.nn.Parameter(self.get_init_trajectory_random_uniform(),
                                             requires_grad=trajectory_learning)

    def forward(self, k_space_input):
        # Fix dimensions for the interpolation
        # We want k space shape will be: (Channels, Batch Size, Resolution, Resolution, 2)
        # When 2(last spot) stands for the Tensor's complex numbers
        k_space_input = k_space_input.permute(0, 1, 4, 2, 3).squeeze(1)
        sub_ksp = utils.bilinear_interpolate_torch_gridsample(k_space_input, self.trajectory)
        output = utils.nufft_adjoint(sub_ksp, self.trajectory, k_space_input.shape, device=sub_ksp.device)
        output = output.unsqueeze(1)
        return output

    def get_trajectory(self):
        return self.trajectory

    def __repr__(self):
        return f'SubSamplingLayer'


class SubSamplingModel(nn.Module):
    def __init__(self, decimation_rate, resolution, trajectory_learning):
        super().__init__()
        self.print = 0
        self.sub_sampling_layer = SubSamplingLayer(decimation_rate, resolution, trajectory_learning)
        self.reconstruction_model = UnetModel(2, 1, 12, 4, 0)

    def forward(self, input_data):
        input_data = self.sub_sampling_layer(input_data)
        output = self.reconstruction_model(input_data.squeeze(1).permute(0, 3, 1, 2))
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

    def return_trajectory_matrix(self):
        map = np.zeros((self.sub_sampling_layer.resolution, self.sub_sampling_layer.resolution))
        clammped_trajectory = np.round(self.sub_sampling_layer.trajectory.detach().numpy()).astype(int)
        map[clammped_trajectory] = 1
        return map
