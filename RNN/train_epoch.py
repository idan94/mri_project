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

from common.args import Args
from data.mri_data import SliceData
from dataTransform import DataTransform
from model import SubSamplingModel
from trajectory_initiations import to_trajectory_image
from models.unet.unet_model import UnetModel
from RNN.Adversarial import Adversarial
from RNN.Subsampeling_model import Sampler
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show(tenor, title=None):
    plt.imshow(tenor.squeeze().detach().numpy(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def load_data(args):
    train_dataset = SliceData(
        root=args.data_path + '/singlecoil_train',
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=args.sample_rate
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = SliceData(
        root=args.data_path + '/singlecoil_val',
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=args.sample_rate
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    display_dataset = [val_dataset[i] for i in
                       range(0, len(val_dataset), len(val_dataset) // args.display_images)]
    display_data_loader = DataLoader(
        dataset=display_dataset,
        batch_size=args.display_images,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_data_loader, val_data_loader, display_data_loader


def create_batches(data, number_of_batches, batch_size):
    batches = []
    for _ in range(number_of_batches):
        indexes = np.random.randint(low=0, high=len(data) - 1, size=batch_size)
        # each batch will contain the full k-space sampling

        batch = [torch.cat([data[0][-1]] + [data[0][indexes[i]] for i in range(batch_size)],dim=0),
                 torch.cat([data[1][-1]] + [data[1][indexes[i]] for i in range(batch_size)],dim=0)]

        batches.append(batch)

    return batches


# TODO: send log of k-space to sampler and adversarial to refrence from, beacuse the values of the k-space are too low
# TODO: add the k-space to the input of the adversarial DNN
def adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer):
    # TODO: this could be very fast if we use very big batch size of the data loader
    # use different data loader with very big batch size for this, the GPU will do all the processing very fast
    # because the entire batch is processed simultaneously
    for k_space, target, f_name, slice in data:
        # TODO: fix batches
        fixed_k_sapce = torch.sqrt(k_space[:, :, :, 0] ** 2 + k_space[:, :, :, 1] ** 2)
        sampling_steps = [[], []]
        batch_size, channels, length, hight, = k_space.permute(0, 3, 1, 2).shape
        sampled_k_space = torch.zeros(batch_size, length, hight, 2,requires_grad=False)
        sampled_k_space.permute(1, 2, 0, 3)[length // 2][hight // 2][:][:] = 1
        # sample the first pixel in the middle
        sampled_k_space = sampled_k_space * k_space
        # create the sampling steps
        for _ in range(sampler.number_of_samples):
            with torch.no_grad():
                temp_sampling_mask = sampler(sampled_k_space)
            sampling_steps[0].append(sampled_k_space)
            # add the sample from the K-space
            sampled_k_space = sampled_k_space + temp_sampling_mask * k_space
            with torch.no_grad():
                sampling_steps[1].append(adversarial(sampled_k_space))
        # new calculate the loss on the last sample
        # the last sample is calculating the loss function without another sampling again
        # the loss could be L2, L1, PSNR, SSIM or another metric
        sampling_steps[0].append(sampled_k_space)
        # calc the loss on each sample
        with torch.no_grad():
            loss_over_last_sample = [loss_function(reconstructor(sampled_k_space.permute(0, 3, 1, 2)[0].unsqueeze(0)),
                                                   target.unsqueeze(1)[i]).reshape(1,1) for i in range(batch_size)]
        sampling_steps[1].append(torch.cat(loss_over_last_sample, dim=0))
        batches = create_batches(sampling_steps, number_of_batches=len(sampling_steps) // 2, batch_size=2)

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
                # TODO: fix the input should be k-space
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


def train_epoch(sampler, adversarial, reconstructor, data, loss_function, reconstructor_epochs, adversarial_optimizer,
                sampler_optimizer,
                reconstructor_optimizer, over_all_optimizer):
    # first we train the adversarial using the reconstructor and the sampler
    adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer)
    sampler_epoch(sampler, adversarial, reconstructor, data, loss_function, sampler_optimizer, over_all_optimizer)

    reconstructor_epochs(sampler, data, reconstructor, loss_function, reconstructor_optimizer, over_all_optimizer,
                         reconstructor_epochs)


def train(number_of_epochs, reconstructor_lr, sampler_lr, adversarial_lr):
    sampler = Sampler(5, 4, 2, 0)
    adversarial = Adversarial(320, 10, 4, 3)
    reconstructor = UnetModel(in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0)
    adversarial_optimizer = torch.optim.Adam(adversarial.parameters(), lr=adversarial_lr)
    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=sampler_lr)
    reconstructor_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=reconstructor_lr)

    # TODO: check this
    # this will be used to reset the gradients of the entire model
    over_all_optimizer = torch.optim.Adam(
        list(adversarial.parameters()) + list(sampler.parameters()) + list(reconstructor.parameters()))

    args = Args().parse_args()
    args.data_path = '../' + args.data_path
    train_data_loader, val_data_loader, display_data_loader = load_data(args)

    loss_function = nn.MSELoss()

    for _ in range(number_of_epochs):
        train_epoch(sampler, adversarial, reconstructor, train_data_loader, loss_function, reconstructor_epochs,
                    adversarial_optimizer,
                    sampler_optimizer,
                    reconstructor_optimizer, over_all_optimizer)


def main():
    train(1, 0.01, 0.01, 0.01)


if __name__ == '__main__':
    main()
