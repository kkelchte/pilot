"""
Version of ResNet using torchvision models
TODO: check if indeed gradients are not secretely kept or bn weights updated during forward pass...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False, feature_extract=False, **kwargs):
    super(Net, self).__init__()
    self.default_image_size=[3,224,224]
    self.default_feature_size=[7,7,512]

    self.network = models.resnet18(pretrained=pretrained)

    if feature_extract:
      for param in self.network.parameters(): param.requires_grad = False
    self.network.fc = nn.Linear(512, output_size)
    
    # define separate feature and classify Tensor sequences used by Tools.grad_CAM and Tools.feature_Neighbors
    modules = list(self.network.children())[:-2]
    self.feature = nn.Sequential(*modules)
    
  def classify(self,x):
    """Perform average pooling and fully connected operation on features
    x: ( B x 7 x 7 x 512 )
    returns: ( B x 1)
    """
    pool_output = self.network.avgpool(x)
    if len(pool_output.size()) == 4:
      pool_output = pool_output.view(pool_output.size()[0], -1)
    return self.network.fc(pool_output)

  def forward(self, x, train=False, verbose=False):
    if verbose: print( x.size())

    if train:
      self.network.train()
    else:
      self.network.eval()
    outputs = self.network(x)
    if verbose: print( outputs.size())
    return outputs

