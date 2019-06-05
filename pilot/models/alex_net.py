"""
Version of ResNet using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  def __init__(self, output_size=10, pretrained=False, feature_extract=False, **kwargs):
    super(Net, self).__init__()
    self.default_image_size=[3,224,224]
    self.network = models.alexnet(pretrained=pretrained)
    self.default_feature_size = self.network.classifier[6].in_features
    if feature_extract:
      for param in self.network.features.parameters(): param.requires_grad = False
    
    self.network.classifier[6]=nn.Linear(self.network.classifier[6].in_features, output_size)
    # modules = list(self.features.children())
    # modules.append()
    self.feature = self.network.features
    self.classify = self.network.classifier

  def forward(self, x, train=False, verbose=False):
    if verbose: print( x.size())
    if train:
      self.network.train()
    else:
      self.network.eval()
    outputs = self.network(x)
    if verbose: print( outputs.size())
    return outputs
