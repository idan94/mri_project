import torch
import numpy as np
from torch import nn
from data.transforms import fft2 as fft, ifft2 as ifft
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.mri_data import SliceData
from dataTransform import DataTransform
from common.args import Args
from models.unet.unet_model import UnetModel
import matplotlib.pyplot as plt


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
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_data_loader, val_data_loader, display_data_loader


class sampeling_pattern_intersection(nn.Module):
    # this model returns at each run a mask of zeros with one value in the location o be sampled from the K-space
    def __init__(self, resolution, unet_chans, unet_num_pool_layers, unet_drop_prob):
        super().__init__()
        self.resolution = resolution
        self.sampler = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
                                 num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)

    def forward(self, k_space, indexes):
        # need to get the possible indexes for the next sample, these
        # will be given by analytically calculating the derivative and second derivative of the trajectory
        samples = self.sampler(k_space.permute(0,3,1,2))
        selected_samples = torch.zeros_like(samples,requires_grad=True)
        selected_samples[:,:,indexes[:,0],indexes[:,1]] = samples[:,:,indexes[:,0],indexes[:,1]]
        shape = selected_samples.shape
        batch_size,_,_,_ = shape
        selected_samples = selected_samples.reshape(batch_size,-1)
        maximals = [torch.argmax(selected_samples[i,...]) for i in  range(selected_samples.shape[0])]
        masks = torch.zeros_like(selected_samples)
        for i in range(len(maximals)):
            index = maximals[i].detach().item()
            masks[i, index] = selected_samples[i, index]
        masks = masks.reshape(shape)
        return masks


def main():
    args = Args().parse_args()
    train_data_loader, val_data_loader, display_data_loader = load_data(args)
    model = sampeling_pattern_intersection(args.resolution, 5, 2, 0.01)
    for k_space, target, f_name, slice in display_data_loader:
        indexes = torch.randint(0,320,(50,2))

        a = model(k_space,indexes)
        a = torch.norm(a)
        a.backward()
        break

if __name__ == '__main__':
    main()