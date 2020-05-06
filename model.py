import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class SubSamplingLayer(nn.Module):

    # This one gives us FULL mask
    def get_init_trajectory_full(self):
        every_point = torch.zeros(self.resolution * self.resolution, 2)
        for i in range(self.resolution):
            if i >= 140 and i <= 170:
                continue
            for j in range(self.resolution):
                if j >= 120 and j <= 160:
                    continue
                every_point[i * self.resolution + j] = torch.tensor([i, j])
        every_point = every_point - (0.5 * self.resolution)
        every_point = every_point.to(device)

        return every_point

    def __init__(self, decimation_rate, resolution, trajectory_learning: bool):
        super().__init__()
        self.decimation_rate = decimation_rate
        self.resolution = resolution
        # self.trajectory_learning = trajectory_learning
        self.num_measurements = resolution ** 2 // decimation_rate
        self.trajectory = torch.nn.Parameter(self.get_init_trajectory_full(), requires_grad=trajectory_learning)

    def forward(self, k_space_input):
        # plt.figure()
        # plt.imshow(print_complex_kspace_tensor(k_space_input[0].squeeze().detach().cpu()), cmap='gray')
        # plt.title('Given K SPACE')
        # plt.show()
        #
        # full_image = transforms.ifft2(k_space_input)
        # plt.figure()
        # plt.imshow(print_complex_image_tensor(full_image[0].squeeze().detach().cpu()), cmap='gray')
        # plt.title('Image from given K SPACE')
        # plt.show()

        # Fix dimensions for the interpolation
        # We want k space shape will be: (Channels, Batch Size, Resolution, Resolution, 2)
        # When 2(last spot) stands for the Tensor's complex numbers
        k_space_input = k_space_input.permute(0, 1, 4, 2, 3).squeeze(1)
        sub_ksp = interp.bilinear_interpolate_torch_gridsample(k_space_input, self.trajectory)
        output = nufft.nufft_adjoint(sub_ksp, self.trajectory, k_space_input.shape)

        # fixed_output = output.squeeze()[0].detach().cpu()
        # plt.figure()
        # plt.imshow(print_complex_image_tensor(fixed_output), cmap='gray')
        # plt.title('After nufft MASKED K SPACE')
        # plt.show()
        #
        # fixed_output_kspace = transforms.fft2(fixed_output)
        #
        # plt.figure()
        # plt.imshow(print_complex_kspace_tensor(fixed_output_kspace), cmap='gray')
        # plt.title('K space of Output')
        # plt.show()

        # Add channel dimension:
        output = output.unsqueeze(1)

        return output

    def get_trajectory(self):
        return self.trajectory

    def __repr__(self):
        return f'SubSamplingLayer'


class SubSamplingModel(nn.Module):
    def __init__(self, decimation_rate, resolution, trajectory_learning):
        super().__init__()

        self.sub_sampling_layer = SubSamplingLayer(decimation_rate, resolution, trajectory_learning)
        self.reconstruction_model = nn.Sequential(
            nn.Conv2d(2, 6, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(6, 15, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(15, 3, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(3, 1, 5, padding=(2, 2)),
        )

    def forward(self, input_data):
        input_data = self.sub_sampling_layer(input_data)
        output = self.reconstruction_model(input_data.squeeze().permute(0, 3, 1, 2))
        # output = torch.sqrt(output[:, :, 0] ** 2 + output[:, :, 1] ** 2)
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()
