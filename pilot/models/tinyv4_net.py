"""
Version of Alexnet with smaller input size and less weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class TinyNet(nn.Module):

  def __init__(self, output_size=10):
    super(TinyNet, self).__init__()
    # first define convolutional and linear operators
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
    self.features=nn.Sequential(
        nn.Conv2d(3, 10, 10, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(10, 20, 5, stride=4),
        nn.ReLU(inplace=True)
      )
    # an affine operation: y = Wx + b
    # in_features, out_features, bias=True
    self.classifier=nn.Sequential(
        nn.Dropout(),
        nn.Linear(20*7*7, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, output_size, bias=False)
      )

  def forward(self, x):
    # second: combine different operators in a forward pass.
    x = self.features(x)
    # print x.shape
    x = x.view(-1, 20*7*7) #adaptive reshape
    x = self.classifier(x)
    return x


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False, **kwargs):
    super(Net, self).__init__()
    if pretrained: raise NotImplementedError
    self.default_image_size=[3,128,128]
    self.default_feature_size=7*7*20
    self.network=TinyNet(output_size=output_size)

    
  def forward(self, x, train=False, verbose=False):
    if train:
      self.network.train()
    else:
      self.network.eval()
    outputs = self.network(x)
    if verbose: print( outputs.size())
    return outputs

    

  # def num_flat_features(self, x):
  #   size = x.size()[1:]  # all dimensions except the batch dimension
  #   num_features = 1
  #   for s in size:
  #     num_features *= s
  #   return num_features

