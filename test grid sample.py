import pathlib
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from data.transforms import fft2, ifft2

from common.args import Args
from data.mri_data import SliceData
from dataTransform import DataTransform
from trajectory_initiations import to_trajectory_image
from models.unet.unet_model import UnetModel
from Adversarial import Adversarial
from Subsampeling_model import Sampler
import matplotlib.pyplot as plt
import time
from train_epoch import load_data
from train_epoch import show


def sample_vector(k_space, vector):
    # this function will use grid sample ato nufft to sample the k_space
    # for each sample in k_space batch there will be a sampling vector in vector parameter
    # the vector that is given is with values between -resolution to resolution
    images = torch.zeros_like(k_space)
    k_space = k_space.permute(0,3,1,2)
    for i in range(k_space.shape[0]):
        space = k_space[i].unsqueeze(0)

        # we need to reverse x,y in the sampling vector
        # because grid sample samples y,x
        sampling_vector = torch.zeros_like(vector[i])
        sampling_vector[..., 0], sampling_vector[..., 1] = vector[i][..., 1], vector[i][..., 0]
        # normalize the vector to be in the right range for sampling
        # the values are between -1 and 1
        sampling_vector = sampling_vector.unsqueeze(0).unsqueeze(0)
        normalized_sampling_vector = (sampling_vector + (k_space.shape[2] / 2)) / (k_space.shape[2] - 1)
        normalized_sampling_vector = 2 * normalized_sampling_vector - 1
        sampled_k_space = torch.nn.functional.grid_sample(space, normalized_sampling_vector, mode='bilinear',
                                                 padding_mode='zeros').unsqueeze(2)
        # for the nufft the indexes sould e in the in indexes domain and no normalize to between -1 and 1
        # and so we will use the original sampling vector
        image = nufft.nufft_adjoint(sampled_k_space, vector[i], space.shape, device=k_space.device).squeeze(0)
        images[i] = fft2(image)
    return images


def main():
    torch.manual_seed(0)
    args = Args().parse_args()
    args.data_path = "../" + args.data_path
    index = 3
    train_data_loader, val_data_loader, display_data_loader = load_data(args)
    for k_space, target, f_name, slice in display_data_loader:
        sampling_vector = [[[i, j] for i in range(k_space.shape[1])] for j in range(k_space.shape[2])]
        sampling_vector = torch.tensor(sampling_vector).float()
        sampling_vector = sampling_vector - 0.5 * k_space.shape[1]
        sampling_vector = sampling_vector.reshape(-1, 2)
        sampling_vector = sampling_vector.expand(k_space.shape[0],-1,-1)
        images = sample_vector(k_space,sampling_vector)
        break

    for i in range(images.shape[0]):
        show(ifft2(images[i]))


if __name__ == '__main__':
    main()
