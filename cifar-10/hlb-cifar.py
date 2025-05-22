"""
hlb_cifar10 using torch backend for tinygradA
implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""
import os
import copy
import math
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils 
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
# data loader
if not os.path.exists(hyp['misc']['data_location']):

        transform = transforms.Compose([
            transforms.ToTensor()])
        cifar10 = torchvision.datasets.CIFAR10('cifar10/', download=True, train=True, transform=transform)
        cifar10_eval=torchvision.datasets.CIFAR10('cifar10/', download=False, train=False, transform=transform)
        train_dataset = torch.utils.DataLoader(cifar10, batch_size=len(cifar10), drop_last=True, shuffle=True, num_workers=2, persistant_workers=False)
        eval_dataset = torch.utils.DataLoader(cifar10_eval, batch_size=len(cifar10_eval), drop_last=True, shuffle=True, num_workers=2, persistant_workers=False)

        train_dataset_gpu = {}
        eval_dataset_gpu = {}

        train_dataset_gpu['images'], train_dataset_gpu['targets'] = [item.to(device=hyp['misc']['device'], non_blocking=True) for item in next(iter(train_dataset))]
        eval_dataset_gpu['images'],  eval_dataset_gpu['targets']  = [item.to(device=hyp['misc']['device'], non_blocking=True) for item in next(iter(eval_dataset)) ]

        cifar10_std, cifar10_mean = torch.std_mean(train_dataset_gpu['images'], dim=(0,2, 3))

        def batch_nomralize_images(input_images, mean, std):
                return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        
        batch_nomralize_images = partial(batch_nomralize_images, mean = cifar10_mean, std = cifar10_std) # preload with mean and std

        # batch normalize dataset 
        train_dataset_gpu['images'] = batch_nomralize_images(train_dataset_gpu['images'])
        eval_dataset_gpu['images'] = batch_nomralize_images(eval_dataset_gpu['images'])

        data = {
            'train': train_dataset_gpu,
            'eval': eval_dataset_gpu
        }

        ## Convert dataset to FP16 
        data['train']['images'] = data['train']['images'].half().requires_grad_(False)
        data['eval']['images']  = data['eval']['images'].half().requires_grad_(False)

        # Convert this to one-hot to support the usage of cutmix 
        data['train']['targets'] = F.one_hot(data['train']['targets']).half()
        data['eval']['targets'] = F.one_hot(data['eval']['targets']).half()

        torch.save(data, hyp['misc']['data_location'])

else:
    ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)=
    ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
    ## hyp dictionary, then we should be good. :)
    data = torch.load(hyp['misc']['data_location'])

# Pad the GPU training dataset
if hyp['net']['pad_amount'] > 0:
    ## Uncomfortable shorthand, but basically we pad evenly on all _4_ sides with the pad_amount specified in the original dictionary
    data['train']['images'] = F.pad(data['train']['images'], (hyp['net']['pad_amount'],)*4, 'reflect')


# network 
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'], weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'], weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs

class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.temperature = temperature

    def forward(self, x):
        if self.temperature is not None:
            weight = self.weight * self.temperature
        else:
            weight = self.weight
        return x @ weight.T

# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in  = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in,  channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling 

# helper functions


# train and eval 