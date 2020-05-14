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
        self.add_argument('--test-name', type=str, default='temp_test',
                          help='name for the output dir')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expects resolution of 320')

        # Data parameters
        self.add_argument('--challenge', default='singlecoil', choices=['singlecoil', 'multicoil'],
                          help='Which challenge')
        self.add_argument('--data-path', default='singlecoil_val', type=pathlib.Path, help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1,
                          help='Fraction of total volumes to include')
        self.add_argument('--num-workers', default=3, type=int, help='Number of workers for dataLoaders')
        self.add_argument('--seed', default=32, type=int, help='Seed for random number generators')

        # Optimization parameters
        self.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
        self.add_argument('--loss-fn', default='L1', choices=['L1', 'MSE'],
                          help='Which loss function to use')
        self.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
        self.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        # Learning rate
        self.add_argument('--lr-step-size', type=int, default=30,
                          help='Period of learning rate decay')
        self.add_argument('--lr-gamma', type=float, default=0.01,
                          help='Multiplicative factor of learning rate decay')
        self.add_argument('--weight-decay', type=float, default=0.,
                          help='Strength of weight decay regularization')
        self.add_argument('--sub-lr', type=float, default=1e-1, help='learning rate of the sub-sampling layer')

        # Unet(reconstruction) parameters
        self.add_argument('--unet-chans', type=int, default=16,
                          help='Unet\'s number of output channels of the first convolution layer')
        self.add_argument('--unet-drop-prob', type=int, default=0,
                          help='Unet\'s dropout probability')
        self.add_argument('--unet-num-pool-layers', type=int, default=4,
                          help='Unet\'s number of down-sampling and up-sampling layers')
        self.add_argument('--decimation-rate', default=4, type=int,
                          help='Ratio of k-space points to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for each volume.')
        self.add_argument('--subsampling-init', choices=['full', 'rows', 'columns', 'spiral', 'circle'], default='full',
                          type=str,
                          help='From which subsampling mask to start')

        # Output
        self.add_argument('--output-dir', default='last_output', type=str, help='Path to outputs')
        self.add_argument('--display-images', default=5, type=int,
                          help='Number of images(target+output) to display when test method is called')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
