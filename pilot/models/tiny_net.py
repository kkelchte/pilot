"""
Version of Alexnet with smaller input size and less weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False):
    super(Net, self).__init__()
    self.default_image_size=[3,128,128]
    # first define convolutional and linear operators
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
    self.conv1 = nn.Conv2d(3, 10, (6,6), stride=3)
    self.conv2 = nn.Conv2d(10, 20, 3, stride=2)

    # an affine operation: y = Wx + b
    # in_features, out_features, bias=True
    self.fc1 = nn.Linear(20*20*20, 1024)
    self.fc2 = nn.Linear(1024, output_size)

    
  def forward(self, x, train=False, verbose=False):
    # second: combine different operators in a forward pass.
    # Max pooling over a (2, 2) window
    if verbose: print x.size()
    x = F.relu(self.conv1(x))
    if verbose: print x.size()
    # If the size is a square you can only specify a single number
    x = F.relu(self.conv2(x))
    if verbose: print x.size()
    x = x.view(-1, 20*20*20) #adaptive reshape
    if verbose: print x.size()
    x = F.relu(self.fc1(x))
    if verbose: print x.size()
    x = self.fc2(x)
    if verbose: print x.size()
    return x

  # def num_flat_features(self, x):
  #   size = x.size()[1:]  # all dimensions except the batch dimension
  #   num_features = 1
  #   for s in size:
  #     num_features *= s
  #   return num_features

