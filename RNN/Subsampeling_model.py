import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
import torch.nn.functional as F
from trajectory_initiations import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sampler(nn.Module):
    def __init__(self, number_of_samples, unet_chans,
                 unet_num_pool_layers, unet_drop_prob):
        super().__init__()
        self.number_of_samples = number_of_samples
        self.NN = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
                                              num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)
        # can try sigmoid
        self.sqwish = nn.ReLU()

    def forward(self,k_space):
        sample_mask = self.NN(k_space.permute(0,3,1,2))
        # # do the soft max across the matrices, these are the two last dimensions
        # F.sample_mask = softmax(sample_mask,dim=2)
        # F.sample_mask = softmax(sample_mask, dim=3)
        # leave the only the maximal value in the sampeling mask
        batch_size = k_space.shape[0]
        maximums = [torch.topk(sample_mask[i][...].flatten(),2)[0][1] for i in range(batch_size)]
        sample_mask = sample_mask - torch.cat(
            [torch.topk(sample_mask[i][...].flatten(), 2)[0][1].reshape(1, 1, 1, 1) for i in range(batch_size)])
        sample_mask = self.sqwish(sample_mask)
        return sample_mask.permute(0,2,3,1)


