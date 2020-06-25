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
from train_epoch import load_data,show,sample_vector


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
