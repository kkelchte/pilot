"""
Version of ResNet using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  # def __init__(self, output_size = 10, pretrained=False):
  def __init__(self, output_size = 10, pretrained=False, **kwargs):
    super(Net, self).__init__()
    self.default_image_size=[3,224,224]
    self.default_feature_size=256*6*6
    self.network = models.alexnet(pretrained=pretrained)
    self.network.classifier[6]=nn.Linear(4096, output_size)

  def forward(self, x, train=False, verbose=False):
    if verbose: print x.size()
    if train:
      self.network.train()
    else:
      self.network.eval()
    outputs = self.network(x)
    if verbose: print outputs.size()
    return outputs

