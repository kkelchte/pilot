#!/usr/bin/python
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

"""
Implementation by work https://arxiv.org/pdf/1709.07492.pdf
Up projection explained in https://arxiv.org/pdf/1606.00373.pdf

@article{Ma2017SparseToDense,
  title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
  author={Ma, Fangchang and Karaman, Sertac},
  booktitle={ICRA},
  year={2018}
}
@article{ma2018self,
  title={Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera},
  author={Ma, Fangchang and Cavalheiro, Guilherme Venturelli and Karaman, Sertac},
  journal={arXiv preprint arXiv:1807.00275},
  year={2018}
}
"""

class Unpool(nn.Module):
  # Unpool: 2*2 unpooling with zero padding
  def __init__(self, num_channels, stride=2):
    super(Unpool, self).__init__()

    self.num_channels = num_channels
    self.stride = stride

    # create kernel [1, 0; 0, 0]
    self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
    self.weights[:,:,0,0] = 1

  def forward(self, x):
    return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
  # Initialize filters with Gaussian random weights
  if isinstance(m, nn.Conv2d):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.ConvTranspose2d):
    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()

class Decoder(nn.Module):
  # Decoder is the base class for all decoders

  names = ['deconv2', 'deconv3', 'upconv', 'upproj']

  def __init__(self):
    super(Decoder, self).__init__()

    self.layer1 = None
    self.layer2 = None
    self.layer3 = None
    self.layer4 = None

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x

class UpProj(Decoder):
  # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

  class UpProjModule(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels):
      super(UpProj.UpProjModule, self).__init__()
      out_channels = in_channels//2
      self.unpool = Unpool(in_channels)
      self.upper_branch = nn.Sequential(collections.OrderedDict([
        ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
        ('batchnorm1', nn.BatchNorm2d(out_channels)),
        ('relu',      nn.ReLU()),
        ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
        ('batchnorm2', nn.BatchNorm2d(out_channels)),
      ]))
      self.bottom_branch = nn.Sequential(collections.OrderedDict([
        ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
        ('batchnorm', nn.BatchNorm2d(out_channels)),
      ]))
      self.relu = nn.ReLU()

    def forward(self, x):
      x = self.unpool(x)
      x1 = self.upper_branch(x)
      x2 = self.bottom_branch(x)
      x = x1 + x2
      x = self.relu(x)
      return x

  def __init__(self, in_channels):
    super(UpProj, self).__init__()
    self.layer1 = self.UpProjModule(in_channels)
    self.layer2 = self.UpProjModule(in_channels//2)
    self.layer3 = self.UpProjModule(in_channels//4)
    self.layer4 = self.UpProjModule(in_channels//8)
    self.conv3 = nn.Conv2d(in_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)
    self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

  def forward(self,x):
    x=super(UpProj, self).forward(x)
    x=self.conv3(x)
    x=self.bilinear(x)
    x=torch.squeeze(x,dim=1)
    return x