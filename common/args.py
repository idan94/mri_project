"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expects resolution of 320')
        self.add_argument('--decision-levels', default=10, type=int, help='number of decisions per epoch')
        self.add_argument('--resume', action='store_true',
                          help='If set, resume the training from a previous model checkpoint'
                               '"--checkpoint" should be set with this')
        self.add_argument('--checkpoint', type=str, default='last_test',
                          help='Output dir name of existing checkpoint. Used along with "--resume"')
        # Data parameters
        self.add_argument('--challenge', default='singlecoil', choices=['singlecoil', 'multicoil'],
                          help='Which challenge')
        self.add_argument('--data-path', default='datasets', type=str, help='Path to the dataset directory')

        # Sampler parameters
        self.add_argument('--sampler-sample-rate', type=float, default=0.3,
                          help='Fraction of total volumes to include to the data set of the sampler DNN')
        self.add_argument('--sampler-batch-size', default=4, type=int, help='Mini batch size for the sampler DNN')
        self.add_argument('--sampler-lr', type=float, default=0.01, help='Learning rate for the sampler DNN')
        self.add_argument('--sampler-lr-decay', type=float, default=0.5,
                          help='the rate for decaying the lr of the sampler DNN')
        self.add_argument('--sampler-lr-decay-every-number-of-epochs', type=int, default=25,
                          help='the number of epochs that will pass for each decaying of the lr for the sampler DNN')
        self.add_argument('--sampler-lr-min-value', type=float, default=0.0001,
                          help='the min learning rate for the sampler DNN')
        self.add_argument('--sampler-convolution-channels', type=int, default=16,
                          help='number of channels in the convolution part of the sampler DNN')
        self.add_argument('--sampler-convolution-layers', type=int, default=4,
                          help='the amount of convectional layers in the sampler DNN')
        self.add_argument('--sampler-linear-layers', type=int, default=2,
                          help='the amount of linear layers in the sampler DNN')

        # Adversarial parameters
        self.add_argument('--adversarial-sample-rate', type=float, default=0.3,
                          help='Fraction of total volumes to include to the data set of the adversarial DNN')
        # in the epoch of the adversarial DNN we first generate data without and gradient calculations
        # this could be done very fast by using the GPU power, as this task is very fitting for parallelization
        # using a big bach size will help us use the full power of the GPU
        # consider even higher batch sizes
        self.add_argument('--adversarial-processing-batch-size', default=400, type=int,
                          help='Mini batch size for the adversarial DNN to generate data, using the sampler DNN')
        self.add_argument('--adversarial-batch-size', default=16, type=int,
                          help='Mini batch size for the adversarial DNN to train')
        self.add_argument('--adversarial-lr', type=float, default=0.01, help='Learning rate for the adversarial DNN')
        self.add_argument('--adversarial-lr-decay', type=float, default=0.5,
                          help='the rate for decaying the lr of the adversarial DNN')
        self.add_argument('--adversarial-lr-decay-every-number-of-epochs', type=int, default=25,
                          help='the number of epochs that will pass for each decaying of the lr for the adversarial DNN')
        self.add_argument('--adversarial-lr-min-value', type=float, default=0.0001,
                          help='the min learning rate for the adversarial DNN')
        self.add_argument('--adversarial-convolution-channels', type=int, default=10,
                          help='the amount of channels in the convolution part of the adversarial DNN')
        self.add_argument('--adversarial-convolution-layers', type=int, default=4,
                          help='the amount of layers in the convolution part of the adversarial DNN')
        self.add_argument('--adversarial-linear-layers', type=int, default=3,
                          help='the amount of layers in the linear part of the adversarial DNN')

        # Reconstructor parameters
        self.add_argument('--reconstructor-sample-rate', type=float, default=0.5,
                          help='Fraction of total volumes to include to the data set of the reconstructor DNN')
        self.add_argument('--reconstructor-batch-size', default=16, type=int,
                          help='Mini batch size for the adversarial DNN')
        self.add_argument('--reconstructor-lr', type=float, default=0.01,
                          help='Learning rate for the reconstructor DNN')
        self.add_argument('--reconstructor-lr-decay', type=float, default=0.5,
                          help='the rate for decaying the lr of the reconstructor DNN')
        self.add_argument('--reconstructor-lr-decay-every-number-of-epochs', type=int, default=25,
                          help='the number of epochs that will pass for each decaying of the lr for the reconstructor')
        self.add_argument('--reconstructor-lr-min-value', type=float, default=0.0001,
                          help='the min learning rate for the reconstructor DNN')
        self.add_argument('--reconstructor-sub-epochs', type=int, default=3,
                          help='the amount of reconstructor epochs that will be done for each adversarial + '
                               'sampler epochs')
        self.add_argument('--reconstruction-unet-chans', type=int, default=32,
                          help='Unet\'s number of output channels of the first convolution layer')
        self.add_argument('--reconstruction-unet-drop-prob', type=int, default=0,
                          help='Unet\'s dropout probability')
        self.add_argument('--reconstruction-unet-num-pool-layers', type=int, default=4,
                          help='Unet\'s number of down-sampling and up-sampling layers')

        # data loading parameters
        self.add_argument('--num-workers', default=8, type=int, help='Number of workers for dataLoaders')
        self.add_argument('--seed', default=32, type=int, help='Seed for random number generators')

        # Optimization parameters
        self.add_argument('--loss-fn', default='L1', choices=['L1', 'MSE'],
                          help='Which loss function to use')

        self.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')

        # Unet(reconstruction) parameters
        self.add_argument('--decimation-rate', default=12, type=int,
                          help='Ratio of k-space points to be sampled. If multiple values are ')

        # Output
        self.add_argument('--output-dir', default='last_test', type=str, help='Path to outputs')
        self.add_argument('--display-images', default=5, type=int,
                          help='Number of images(target+output) to display when test method is called')
        # To-delete
        self.add_argument('--batch-size', type=int, default=4,
                          help='Unet\'s number of down-sampling and up-sampling layers')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
