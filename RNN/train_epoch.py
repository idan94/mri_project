import pathlib
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from common.args import Args
from data.mri_data import SliceData
from dataTransform import DataTransform
from model import SubSamplingModel
from trajectory_initiations import to_trajectory_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_batches(data, number_of_batches, batch_size):
    batches = []
    for _ in range(number_of_batches):
        indexes = np.random.randint(low=0, high=len(data) - 2, size=batch_size)
        batch = [[data[-1]] + data[indexes] for _ in range(number_of_batches)]
        batches += [[sample[0] for sample in batch], [sample[1] for sample in batch]]
    batches = [torch.tensor([batches[i][0] for i in range(len(batches))]),
               torch.tensor([batches[i][1] for i in range(len(batches))])]
    return batches


def adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer):
    for k_space, target, f_name, slice in data:
        sampling_steps = []
        batch_size, channels, length, hight, _ = k_space.shape()
        sampling_mask = torch.zeros(batch_size, length, hight)
        # create the sampling steps
        for _ in range(sampler.number_of_samples):
            temp_sampling_mask = sampler(k_space.permute(0, 3, 1, 2))
            sample = [sampling_mask]
            sampling_mask = sampling_mask + temp_sampling_mask * k_space
            with torch.no_grad:
                sample.append(adversarial(sampling_mask))
            sampling_steps.append(sample)
            # new calculate the loss on the last sample
            # the last sample is calculating the loss function without another sampling again
            # the loss could be L2, L1, PSNR, SSIM or another metric
            sampling_steps.append([sampling_mask, loss_function(reconstructor(sampling_mask), target)])
            batches = create_batches(sampling_steps, number_of_batches=len(sampling_steps) // 10, batch_size=32)

            for batch in batches:
                over_all_optimizer.zero_grad()
                loss = loss_function(adversarial(batch[0]), batch[1])
                loss.backward()
                adversarial_optimizer.step()


def sampler_epoch(sampler, adversarial, reconstructor, data, loss_function, sampler_optimizer, over_all_optimizer):
    for k_space, target, f_name, slice in data:
        batch_size, channels, length, hight, _ = k_space.shape()
        sampling_mask = torch.zeros(batch_size, length, hight)
        for _ in range(sampler.number_of_samples):
            # to zero the grad of the adversarial network and the u_net as well
            over_all_optimizer.zero_grad()
            temp_sampling_mask = sampler(sampling_mask)
            sampling_mask = sampling_mask + temp_sampling_mask * k_space
            loss = adversarial(sampling_mask)
            loss.backward()
            sampler_optimizer.step()


def reconstructor_epochs(sampler, data, reconstructor, loss_function, reconstructor_optimizer, over_all_optimizer,
                         epochs):
    # create the subsampled data
    subsampled_data = []
    for k_space, target, f_name, slice in data:
        batch_size, channels, length, hight, _ = k_space.shape()
        sampling_mask = torch.zeros(batch_size, length, hight)
        for _ in range(sampler.number_of_samples):
            # to zero the grad of the adversarial network and the u_net as well
            with torch.no_grad:
                sampling_mask += sampler(sampling_mask)

        subsampled_data += [sampling_mask * k_space, target]
    subsampled_data = [torch.tensor(subsampled_data[:][0]), torch.tensor(subsampled_data[:][1])]
    # optimize the u-net
    for _ in range(epochs):
        for batch in subsampled_data:
            over_all_optimizer.zero_grad()
            loss = loss_function(reconstructor(batch[0]), batch[1])
            loss.backward()
            reconstructor_optimizer.step()


def train_epoch(sampler, adversarial, reconstructor, data, loss_function,reconstructor_epochs, adversarial_optimizer, sampler_optimizer,
                reconstructor_optimizer, over_all_optimizer):
    # first we train the adversarial using the reconstructor and the sampler
    adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer)
    sampler_epoch(sampler, adversarial, reconstructor, data, loss_function, sampler_optimizer, over_all_optimizer)

    reconstructor_epochs(sampler, data, reconstructor, loss_function, reconstructor_optimizer, over_all_optimizer,
                         reconstructor_epochs)

