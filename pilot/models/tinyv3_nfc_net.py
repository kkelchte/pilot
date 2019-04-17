"""
Version of Alexnet with smaller input size and less weights

Based on Actor Critic code:
https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinyv3_net

class Net(nn.Module):

  def __init__(self, output_size=10, pretrained=False, n_frames=5, **kwargs):
    super(Net, self).__init__()
    feature_network = tinyv3_net.Net(pretrained=pretrained)
    self.H=1024
    self.n_frames=n_frames
    self.default_feature_size=self.n_frames*feature_network.default_feature_size
    self.default_image_size=feature_network.default_image_size
    self.cnn = feature_network.network.features
    self.classifier=nn.Sequential(
        nn.Linear(self.default_feature_size, self.H),
        nn.ReLU(inplace=True),
        nn.Linear(self.H, output_size)
    )

    # self.linear1 = nn.Linear(self.default_feature_size, self.H)

    # self.linear2 = nn.Linear(self.H, output_size)
  
  def forward(self, x, train=False, verbose=False):
    """
    make a forward pass on the input data x 
    x: the inputs have to be a tensor of 5 dimensions (B, T, C, H, W)
      all data should be on the same device as the network.
    train: defines wether dropout is used
    verbose: print intermediate sizes
    """
    inputs = x
    # first: rearrange data from BxTxCxHxW to BTxCxHxW
    # import pdb; pdb.set_trace()
    batch_size, timesteps, C, H, W = inputs.size()
    if verbose: print("batch_size: {0}, timesteps: {1}, C: {2}, H: {3}, W: {4}.".format(batch_size, timesteps, C, H, W))
    assert(timesteps==self.n_frames)

    c_in = inputs.view(batch_size * self.n_frames, C, H, W)
    if verbose: print("c_in.size: {0}".format(c_in.size()))
    c_out = self.cnn(c_in)
    
    # second: rearrange data for 3D-fully connected layer
    c_out = c_out.view(batch_size,  self.n_frames, -1)
    f_in = c_out.view(batch_size, self.default_feature_size)
    if verbose: print("c_out.size: {0}".format(c_out.size()))
        
    # calculate output over batch and time sequence
    f_out=self.classifier(f_in)
    # f_h = self.linear1(f_in)
    # f_out = self.linear2(f_h)
    # f_out = f_out.view(batch_size, self.n_frames, -1)

    if verbose: print("f_out: {0}".format(f_out.size()))
    return f_out
    # return F.log_softmax(r_out2, dim=1)


  # def num_flat_features(self, x):
  #   size = x.size()[1:]  # all dimensions except the batch dimension
  #   num_features = 1
  #   for s in size:
  #     num_features *= s
  #   return num_features

