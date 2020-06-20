import torch
from torch import nn

import pytorch_nufft.interp as interp
import pytorch_nufft.nufft as nufft
from models.unet.unet_model import UnetModel
from trajectory_initiations import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sampler(nn.Module):
    def __init__(self, number_of_samples, unet_chans,
                 unet_num_pool_layers, unet_drop_prob):
        super().__init__()
        self.number_of_samples = number_of_samples
        self.NN = UnetModel(in_chans=2, out_chans=1, chans=unet_chans,
                                              num_pool_layers=unet_num_pool_layers, drop_prob=unet_drop_prob)
        self.max = nn.Softmax()
        # can try sigmoid
        self.sqwish = nn.ReLU()

    def forward(self,k_space):
        sample_mask = self.NN(k_space.permute(0,3,1,2))
        sample_mask = self.max(sample_mask)
        # leave the only the maximal value in the sampeling mask
        batch_size = k_space.shape[0]
        maximums = [torch.topk(sample_mask[i][...],2) for i in range(batch_size)]
        epsilons = [maximums[i][0] - maximums[i][1] for i in range(batch_size)]
        for i in range(batch_size):
            # only the maximal value is epsilon, the second max is 0 and the rest are negative
            sample_mask[i] = sample_mask[i] - maximums[i][0] + epsilons[i]
            # now the maximal value is 1 and the rest are negative or zeros
            sample_mask[i] = sample_mask[i] / epsilons[i]
            # now the maximal is one and the rest are 0
            sample_mask[i] = self.sqwish(sample_mask[i])
        return sample_mask


