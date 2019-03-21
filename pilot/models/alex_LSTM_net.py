"""
Version of Alexnet with smaller input size and less weights

Based on Actor Critic code:
https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import alex_net

class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False,feature_size=100, dropout=0):
    super(Net, self).__init__()
    feature_network = alex_net.Net(pretrained=pretrained)
    self.default_image_size=feature_network.default_image_size
    self.cnn = feature_network.network.features
    self.rnn = nn.LSTM(input_size=feature_network.default_feature_size,
                      hidden_size=64,
                      num_layers=2,
                      batch_first=True,
                      dropout=0)
    self.linear = nn.Linear(64, output_size)

  def forward(self, x, train=False, verbose=False):
    inputs, (hx, cx) = x
    # first: rearrange data from BxTxCxHxW to BTxCxHxW
    batch_size, timesteps, C, H, W = inputs.size()
    print("batch_size: {0}, timesteps: {1}, C: {2}, H: {3}, W: {4}.".format(batch_size, timesteps, C, H, W))
    c_in = inputs.view(batch_size * timesteps, C, H, W)
    print("c_in.size: {0}".format(c_in.size()))
    c_out = self.cnn(c_in)
    
    # second: rearrange data for LSTM
    r_in = c_out.view(batch_size, timesteps, -1)
    print("r_in.size: {0}".format(r_in.size()))
    
    r_out, (h_n, h_c) = self.rnn(r_in, (hx, cx))
    
    print("r_out.size: {0}, h_n: {1}, h_c: {2}".format(r_out.size(), h_n.size(), h_c.size()))
    
    # Take last output value in sequence
    r_out2 = self.linear(r_out[:, -1, :])

    print("r_out2: {0}".format(r_out2.size()))
    return r_out2, (h_n, h_c)
    # return F.log_softmax(r_out2, dim=1)


  # def num_flat_features(self, x):
  #   size = x.size()[1:]  # all dimensions except the batch dimension
  #   num_features = 1
  #   for s in size:
  #     num_features *= s
  #   return num_features

