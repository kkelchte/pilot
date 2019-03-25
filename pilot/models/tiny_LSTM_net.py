"""
Version of Alexnet with smaller input size and less weights

Based on Actor Critic code:
https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiny_net

class Net(nn.Module):

  def __init__(self, output_size = 10, pretrained=False,feature_size=100, dropout=0, hidden_size=64, num_layers=2):
    super(Net, self).__init__()
    feature_network = tiny_net.Net(pretrained=pretrained)
    self.H=hidden_size
    self.L=num_layers
    self.default_image_size=feature_network.default_image_size
    self.cnn = feature_network.network.features
    self.rnn = nn.LSTM(input_size=feature_network.default_feature_size,
                      hidden_size=self.H,
                      num_layers=self.L,
                      batch_first=True,
                      dropout=0)
    self.linear = nn.Linear(self.H, output_size)

  def get_init_state(self,B):
    """
    returns a tuple of numpy zero arrays of size LxBxH
    """
    return (torch.zeros((self.L, B, self.H)), torch.zeros((self.L, B, self.H)))

    
  def forward(self, x, train=False, verbose=False):
    """
    make a forward pass on the input data x 
    x: tuple with (inputs, (hx, cx)) hidden cell states or a tensor in which case initial states are used
      the inputs have to be a tensor of 5 dimensions (B, T, C, H, W)
      all data should be on the same device as the network.
    train: defines wether dropout is used
    verbose: print intermediate sizes

    """
    if isinstance(x, tuple):
      inputs, (hx, cx) = x
    elif isinstance(x, torch.Tensor):
      inputs = x
      hx, cx = self.get_init_state(len(x))
    # first: rearrange data from BxTxCxHxW to BTxCxHxW
    # import pdb; pdb.set_trace()
    batch_size, timesteps, C, H, W = inputs.size()
    if verbose: print("batch_size: {0}, timesteps: {1}, C: {2}, H: {3}, W: {4}.".format(batch_size, timesteps, C, H, W))
    c_in = inputs.view(batch_size * timesteps, C, H, W)
    if verbose: print("c_in.size: {0}".format(c_in.size()))
    c_out = self.cnn(c_in)
    
    # second: rearrange data for LSTM
    r_in = c_out.view(batch_size, timesteps, -1)
    if verbose: print("r_in.size: {0}".format(r_in.size()))
    
    r_out, (h_n, h_c) = self.rnn(r_in, (hx, cx))
    if verbose: print("r_out.size: {0}, h_n: {1}, h_c: {2}".format(r_out.size(), h_n.size(), h_c.size()))
    
    # import pdb; pdb.set_trace()
    f_in = r_out.contiguous().view(batch_size*timesteps, self.H)
    if verbose: print("f_in.size: {0}".format(f_in.size()))
    # calculate output over batch and time sequence
    f_out = self.linear(f_in)
    f_out = f_out.view(batch_size, timesteps, -1)

    if verbose: print("f_out: {0}".format(f_out.size()))
    return f_out, (h_n, h_c)
    # return F.log_softmax(r_out2, dim=1)


  # def num_flat_features(self, x):
  #   size = x.size()[1:]  # all dimensions except the batch dimension
  #   num_features = 1
  #   for s in size:
  #     num_features *= s
  #   return num_features

