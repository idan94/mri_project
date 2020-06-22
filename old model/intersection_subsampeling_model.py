import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
from torch.nn import functional as F
from trajectory_initiations import *
from model import SubSamplingLayer
from data.transforms import fft2, ifft2
import matplotlib.pyplot as plt

def show(tensor):
    plt.imshow(tensor.detach().numpy(),cmap='gray')
    plt.show()

class K_space_sub_sampeling(nn.Module):
    def __init__(self, resolution,
                 unet_chans,
                 unet_num_pool_layers, unet_drop_prob):
        super().__init__()
        self.resolution = resolution
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1))
        self.sampeling_unet = UnetModel(in_chans=3, out_chans=1, chans=unet_chans,
                                              num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)
    def forward(self, k_space,mask):
        # the mask is generated without relation to the NN
        # the mask will represent the legal locations for the NN to sample
        # concatenate the k_space and the mask in the channel dimension
        output = self.conv_layer(torch.cat((k_space.permute(0,3,1,2),mask.unsqueeze(1)),dim = 1))
        sampling_mask = self.sampeling_unet(output)
        sampling_mask = sampling_mask * mask.unsqueeze(1)
        # TODO: add soft max over the leagal erea
        result = torch.zeros_like(sampling_mask)

        # save the index of the maximal value for each sample in the batch
        indexes = [torch.argmax(sampling_mask[i]) for i in range(result.shape[0])]
        for i in range(result.shape[0]):
            result[i].flatten()[indexes[i]] = sampling_mask[i].flatten()[indexes[i]]
        return result

def main():
    NN = K_space_sub_sampeling(320,16,4,0)
    k_space = torch.rand((4,320,320,2))
    mask = torch.zeros((4,320,320))
    middle = mask.shape[2] // 2
    mask[:,middle - 20:middle + 20,middle - 20:middle + 20] = 1
    result = NN(k_space,mask)
    loss = torch.norm(result)
    loss.backward()
    plt.imshow(result.squeeze()[0,...].detach().numpy(),cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()



