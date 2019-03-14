"""
Version of Inception using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False):
    super(Net, self).__init__()
    self.default_image_size=[3,299,299]

    self.network = models.inception_v3(pretrained=pretrained)

    # Handle the auxilary net
    num_ftrs = self.network.AuxLogits.fc.in_features
    self.network.AuxLogits.fc = nn.Linear(num_ftrs, output_size)
    # Handle the primary net
    num_ftrs = self.network.fc.in_features
    self.network.fc = nn.Linear(num_ftrs,output_size)


  def forward(self, x, train=False, verbose=False):
    if verbose: print x.size()

    if train:
      self.network.train()
    else:
      self.network.eval()
    try:
      outputs,_ = self.network(x)
    except:
      outputs = self.network(x)
    # outputs,_ = self.network(x)
    if verbose: print outputs.size()
    return outputs

