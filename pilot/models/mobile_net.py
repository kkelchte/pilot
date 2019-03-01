"""
Version of ResNet using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

from MobileNetV2 import MobileNetV2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False):
    super(Net, self).__init__()
    self.default_image_size=[3,224,224]
    self.network = MobileNetV2(n_class=output_size)
    if pretrained:
      state_dict = torch.load("/esat/opal/kkelchte/docker_home/tensorflow/log/pytorch-mobilenet/mobilenet_v2.pth.tar", strict=False)
      self.network.load_state_dict(state_dict)

  def forward(self, x, train=False, verbose=False):
    if verbose: print x.size()

    if train:
      self.network.train()
    else:
      self.network.eval()
    outputs = self.network(x)
    if verbose: print outputs.size()
    return outputs

