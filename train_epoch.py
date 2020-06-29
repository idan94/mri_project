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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show(tenor, title=None):
    new_tensor = tenor.clone()
    if len(tenor.shape) > 2:
        new_tensor = new_tensor.permute(2, 0, 1)[0] ** 2 + new_tensor.permute(2, 0, 1)[1] ** 2
    plt.imshow(new_tensor.squeeze().detach().numpy(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def sample_vector(k_space, vector):
    # TODO: make the loop parallel using torch parallel loops
    # this function will use grid sample ato nufft to sample the k_space
    # for each sample in k_space batch there will be a sampling vector in vector parameter
    # the vector that is given is with values between -resolution to resolution
    images = torch.zeros_like(k_space)
    k_space = k_space.permute(0, 3, 1, 2)
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


def load_data(args):
    train_dataset = SliceData(
        root=args.data_path + '/singlecoil_train',
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=0.2
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
        challenge=args.challenge, sample_rate=0.2
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

        batch = [torch.cat([data[0][-1]] + [data[0][indexes[i]] for i in range(batch_size)], dim=0),
                 torch.cat([data[1][-1]] + [data[1][indexes[i]] for i in range(batch_size)], dim=0)]

        batches.append(batch)

    return batches


def K_space_log(k_space):
    # the log of the k_space is done on the abs of each channel of the k_space
    # this is done to scale the range of numbers of the k_space to something  that can fit better to a DNN
    log_of_k_space = torch.zeros_like(k_space)
    log_of_k_space[:, :, :, 0] = torch.log(torch.abs(k_space[:, :, :, 0])) * torch.sign(k_space[:, :, :, 0])
    log_of_k_space[:, :, :, 1] = torch.log(torch.abs(k_space[:, :, :, 1])) * torch.sign(k_space[:, :, :, 1])
    return log_of_k_space


def adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer):
    # TODO: this could be very fast if we use very big batch size of the data loader
    # TODO: change sampling model to few samples instead of one
    # TODO: this loop can be a parallel loop
    # use different data loader with very big batch size for this, the GPU will do all the processing very fast
    # because the entire batch is processed simultaneously
    for k_space, target, f_name, slice in data:
        # TODO: fix batches
        k_space_log = K_space_log(k_space)
        sampling_steps = [[], []]
        batch_size, channels, length, hight, = k_space.permute(0, 3, 1, 2).shape
        sampling_map = torch.zeros(batch_size, length, hight, channels, requires_grad=False)
        sampling_map.permute(1, 2, 0, 3)[length // 2][hight // 2][:][:] = 1
        # sample the first pixel in the middle
        sampled_log_k_space = sampling_map * k_space_log
        # create the sampling steps
        overall_sampling_vector = torch.zeros((batch_size, 0, 2))
        for _ in range(sampler.number_of_decisions):
            with torch.no_grad():
                sampling_vector = sampler(sampled_log_k_space)

            sampling_steps[0].append(torch.cat([sampled_log_k_space, k_space_log], dim=3))
            # add the sample from the K-space
            sampled_log_k_space = sampled_log_k_space + sample_vector(k_space_log, sampling_vector)
            overall_sampling_vector = torch.cat([overall_sampling_vector, sampling_vector], dim=1)
            with torch.no_grad():
                sampling_steps[1].append(adversarial(sampled_log_k_space, k_space_log))
        # new calculate the loss on the last sample
        # the last sample is calculating the loss function without another sampling again
        # the loss could be L2, L1, PSNR, SSIM or another metric
        sampling_steps[0].append(torch.cat([sampled_log_k_space, k_space_log], dim=3))
        # calc the loss on each sample
        sampled_k_space = sample_vector(k_space, overall_sampling_vector)
        with torch.no_grad():
            loss_over_last_sample = [loss_function(reconstructor(sampled_k_space.permute(0, 3, 1, 2)[0].unsqueeze(0)),
                                                   target.unsqueeze(1)[i]).reshape(1, 1) for i in range(batch_size)]
        sampling_steps[1].append(torch.cat(loss_over_last_sample, dim=0))
        batches = create_batches(sampling_steps, number_of_batches=len(sampling_steps) // 2, batch_size=2)

        for batch in batches:
            over_all_optimizer.zero_grad()
            loss = loss_function(adversarial(batch[0][..., :2], batch[0][..., 2:]), batch[1])
            loss.backward()
            adversarial_optimizer.step()


def sampler_epoch(sampler, adversarial, reconstructor, data, loss_function, sampler_optimizer, over_all_optimizer):
    for k_space, target, f_name, slice in data:
        k_space_log = K_space_log(k_space)
        batch_size, length, hight, channels = k_space.shape
        sampling_mask = torch.zeros(batch_size, length, hight, channels, requires_grad=False)
        sampling_mask.permute(1, 2, 0, 3)[length // 2][hight // 2][:][:] = 1
        sampled_log_k_space = sampling_mask * k_space_log
        k_space.requires_grad = True
        for _ in range(sampler.number_of_decisions):
            # to zero the grad of the adversarial network and the u_net as well
            over_all_optimizer.zero_grad()
            # delete the gradients from previous iterations
            k_space_log = k_space_log.detach()
            sampled_log_k_space = sampled_log_k_space.detach()
            sampling_vector = sampler(sampled_log_k_space)
            sampled_log_k_space = sampled_log_k_space + sample_vector(k_space_log, sampling_vector)
            loss = adversarial(sampled_log_k_space, k_space_log)
            # the adversarial NN approx the error of each sample, to minimize it we will take the MSE off
            # the adversarial result over the entire batch
            loss = torch.norm(loss, 2, dim=0)
            loss.backward()
            sampler_optimizer.step()


def reconstructor_epochs(sampler, data, reconstructor, loss_function, reconstructor_optimizer, over_all_optimizer,
                         epochs):
    # create the subsampled data
    # this could be done fast using a very big batch size
    subsampled_data = [[], []]
    for k_space, target, f_name, slice in data:
        batch_size, length, hight, channels = k_space.shape
        sampling_of_the_k_space = torch.zeros(batch_size, length, hight, 1)
        sampling_of_the_k_space.permute(1, 2, 0, 3)[length // 2][hight // 2][:][:] = 1
        k_space_log = K_space_log(k_space)
        sampling_of_the_k_space_log = k_space_log * sampling_of_the_k_space
        over_all_sampling_vector = torch.zeros((batch_size, 0, 2))
        for _ in range(sampler.number_of_decisions):
            # to zero the grad of the adversarial network and the u_net as well
            with torch.no_grad():
                sampling_vector = sampler(sampling_of_the_k_space_log)
                over_all_sampling_vector = torch.cat([over_all_sampling_vector, sampling_vector], dim=1)
                sampling_of_the_k_space_log = sampling_of_the_k_space_log + sample_vector(k_space_log,
                                                                                          sampling_vector)
        subsampled_data[0].append(sample_vector(k_space, over_all_sampling_vector))
        subsampled_data[1].append(target)
    # optimize the u-net
    for _ in range(epochs):
        for subsampled_k_space, target in zip(subsampled_data[0], subsampled_data[1]):
            over_all_optimizer.zero_grad()
            loss = loss_function(reconstructor(subsampled_k_space.permute(0, 3, 1, 2)), target)
            loss.backward()
            reconstructor_optimizer.step()


def train_epoch(sampler, adversarial, reconstructor, data, loss_function, number_of_recon_epochs,
                adversarial_optimizer,
                sampler_optimizer,
                reconstructor_optimizer, over_all_optimizer):
    # first we train the adversarial using the reconstructor and the sampler
    t1 = time.time()
    adversarial_epoch(sampler, adversarial, reconstructor, data, loss_function, adversarial_optimizer,
                      over_all_optimizer)
    sampler_epoch(sampler, adversarial, reconstructor, data, loss_function, sampler_optimizer, over_all_optimizer)
    print(time.time() - t1)

    reconstructor_epochs(sampler, data, reconstructor, loss_function, reconstructor_optimizer, over_all_optimizer,
                         number_of_recon_epochs)


# TODO: add data perallel
# TODO: add visulize
# TODO: add save and load models
def train():
    args = Args().parse_args()

    sampler = Sampler(args.resolution, args.decimation_rate, args.decision_levels, args.sampler_convolution_channels,
                      args.sampler_convolution_layers, args.sampler_linear_layers)

    adversarial = Adversarial(args.resolution, args.adversarial_convolution_channels,
                              args.adversarial_convolution_layers, args.adversarial_linear_layers)

    reconstructor = UnetModel(in_chans=2, out_chans=1, chans=args.reconstruction_unet_chans,
                              num_pool_layers=args.reconstruction_unet_num_pool_layers,
                              drop_prob=args.reconstruction_unet_drop_prob)

    adversarial_optimizer = torch.optim.Adam(adversarial.parameters(), lr=args.adversarial_lr, )
    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=args.sampler_lr)
    reconstructor_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=args.reconstructor_lr)

    # TODO: check this
    # this will be used to reset the gradients of the entire model
    over_all_optimizer = torch.optim.Adam(
        list(adversarial.parameters()) + list(sampler.parameters()) + list(reconstructor.parameters()))

    # TODO: remove this line, each NN needs it's own data loader with it's own sample rate
    args.sample_rate = 0.2
    train_data_loader, val_data_loader, display_data_loader = load_data(args)

    if args.loss_fn == "MSE":
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.L1Loss()

    for _ in range(args.num_epochs):
        train_epoch(sampler, adversarial, reconstructor, train_data_loader, loss_function,
                    args.reconstructor_sub_epochs,
                    adversarial_optimizer,
                    sampler_optimizer,
                    reconstructor_optimizer, over_all_optimizer)
def main():
    train()


if __name__ == '__main__':
    main()
