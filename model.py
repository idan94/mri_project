import numpy as np
import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
from subsample_init import TrajectoryInit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class SubSamplingLayer(nn.Module):
    def __init__(self, decimation_rate, resolution, trajectory_learning: bool, subsampling_trajectory):
        super().__init__()
        self.decimation_rate = decimation_rate
        self.resolution = resolution
        self.trajectory_learning = trajectory_learning
        self.subsampling_trajectory = subsampling_trajectory
        # self.trajectory_learning = trajectory_learning
        self.num_measurements = resolution ** 2 // decimation_rate
        self.trajectory = self.get_init_trajectory()

    def get_init_trajectory(self):
        trajectory = TrajectoryInit(resolution=self.resolution, device=device)

        if self.subsampling_trajectory == 'full':
            trajectory = trajectory.full()
        if self.subsampling_trajectory == 'rows':
            trajectory = trajectory.subsample_random_rows(5)
        if self.subsampling_trajectory == 'colomns':
            trajectory = trajectory.subsample_random_cols(5)
        if self.subsampling_trajectory == 'spiral':
            trajectory = trajectory.spiral(5, 2)  # samples, density

        return torch.nn.Parameter(trajectory,
                                  requires_grad=self.trajectory_learning)

    def forward(self, k_space_input):
        # Fix dimensions for the interpolation
        # We want k space shape will be: (Channels, Batch Size, Resolution, Resolution, 2)
        # When 2(last spot) stands for the Tensor's complex numbers
        k_space_input = k_space_input.permute(0, 1, 4, 2, 3).squeeze(1)
        sub_ksp = interp.bilinear_interpolate_torch_gridsample(k_space_input, self.trajectory)
        output = nufft.nufft_adjoint(sub_ksp, self.trajectory, k_space_input.shape, device=sub_ksp.device)
        output = output.unsqueeze(1)
        return output

    def get_trajectory(self):
        return self.trajectory

    def __repr__(self):
        return f'SubSamplingLayer'


class SubSamplingModel(nn.Module):
    def __init__(self, decimation_rate, resolution, trajectory_learning, subsampling_trajectory, unet_chans,
                 unet_num_pool_layers,
                 unet_drop_prob):
        super().__init__()
        self.sub_sampling_layer = SubSamplingLayer(decimation_rate, resolution, trajectory_learning,
                                                   subsampling_trajectory)
        self.reconstruction_model = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
                                              num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)

    def forward(self, input_data):
        image_from_sub_sampling = self.sub_sampling_layer(input_data)
        output = self.reconstruction_model(image_from_sub_sampling.squeeze(1).permute(0, 3, 1, 2))
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

    def get_trajectory_matrix(self):
        trajectory_matrix = np.zeros((self.resolution, self.resolution))
        trajectory_points = np.round(self.trajectory.detach().numpy()).astype(int)
        trajectory_matrix[trajectory_points] = 1
        return trajectory_matrix
