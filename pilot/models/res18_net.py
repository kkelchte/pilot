"""
Version of ResNet using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False):
    super(Net, self).__init__()
    self.default_image_size=[3,224,224]

    self.network = models.resnet18(pretrained=pretrained)
    self.network.fc = nn.Linear(512, output_size)

  def forward(self, x, train=False, verbose=False):
    if verbose: print x.size()
    if train:
      outputs = self.network.train(x)
    else:
      outputs = self.network.eval(x)
    if verbose: print outputs.size()
    return outputs

