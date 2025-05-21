"""
hlb_cifar10 using torch backend for tinygradA
implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim 
from dataset import get_cifar10_dataloaders
from model import SimpleHLBCNN
import torch.nn.functional as F
import torchvision
from torchvision import transforms


# global defaults
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

batch_size = 1024
bias_scaler = 64
hyp = {
    'opt': {
        'bias_lr':        1.525 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'non_bias_lr':    1.525 / 512,
        'bias_decay':     6.687e-4 * batch_size/bias_scaler,
        'non_bias_decay': 6.687e-4 * batch_size,
        'scaling_factor': 1./9,
        'percent_start': .23,
        'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 10, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
        'train_epochs': 12.1,
        'device': 'cuda',
        'data_location': 'data.pt',
    }
}


hyp = {
    'opt': {
        'bias_lr':        1.525 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'non_bias_lr':    1.525 / 512,
        'bias_decay':     6.687e-4 * batchsize/bias_scaler,
        'non_bias_decay': 6.687e-4 * batchsize,
        'scaling_factor': 1./9,
        'percent_start': .23,
        'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 10, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
        'train_epochs': 12.1,
        'device': 'cuda',
        'data_location': 'data.pt',
    }
}# data loader
if not os.path.exists(hyp['misc']['data_location']):

        transform = transforms.Compose([
            transforms.ToTensor()])

        cifar10      = torchvision.datasets.CIFAR10('cifar10/', download=True,  train=True,  transform=transform)

# helper functions


# train and eval 