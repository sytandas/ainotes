"""
hlb_cifar10 using torch backend for tinygradA
implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""
import torch
import torch.nn as nn
import torch.optim as optim 
from dataset import get_cifar10_dataloaders
from model import SimpleHLBCNN
import torch.nn.functional as F

