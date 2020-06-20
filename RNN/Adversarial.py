import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
from trajectory_initiations import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Adversarial(nn.Module):
    def __init__(self, resolution, chancels, number_of_conv_layers,
                 linear_layers):
        super().__init__()
        self.resolution = resolution
        conv_layers = [nn.Sequential(nn.Conv2d(chancels,chancels,3,padding=1),nn.MaxPool2d(2)) for i in range(number_of_conv_layers-2)]
        self.conv_layer = nn.Sequential(nn.Conv2d(2,chancels,3,padding=1)
        ,nn.Sequential(*conv_layers),
        nn.Conv2d(chancels,1,3,padding=1))
        # 6 and not 4 because we need to calc for the convolution without the padding and the maxpooling
        number_of_neuorons = [resolution**2 // ((4**(number_of_conv_layers-2))*(2**i)) for i in range(linear_layers-1)]
        linear_layer = [nn.Linear(number_of_neuorons[i],number_of_neuorons[i+1]) for i in range(linear_layers-2)]
        linear_layer = linear_layer + [nn.Linear(number_of_neuorons[len(number_of_neuorons)-1],1)]
        self.linear = nn.Sequential(*linear_layer)

    def forward(self,k_space):
        out = self.conv_layer(k_space.permute(0,3,1,2))
        return self.linear(out.reshape(out.shape[0],-1))



