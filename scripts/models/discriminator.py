from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import tanh, sigmoid

import nn


def build(input_height, input_width, concat_var):
    """
    Build the discriminator, all weights initialized from scratch
    :param input_width:
    :param input_height: 
    :param concat_var: Theano symbolic tensor variable
    :return: Dictionary that contains the discriminator
    """

    net = {'input': InputLayer((None, 4, input_height, input_width), input_var=concat_var)}
    print "Input: {}".format(net['input'].output_shape[1:])

    net['merge'] = ConvLayer(net['input'], 3, 1, pad=0, flip_filters=False)
    print "merge: {}".format(net['merge'].output_shape[1:])

    net['conv1'] = ConvLayer(net['merge'], 32, 3, pad=1)
    print "conv1: {}".format(net['conv1'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1'], 4)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = ConvLayer(net['pool1'], 64, 3, pad=1)
    print "conv2_1: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = ConvLayer(net['conv2_1'], 64, 3, pad=1)
    print "conv2_2: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = nn.weight_norm(ConvLayer(net['pool2'], 64, 3, pad=1))
    print "conv3_1: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = nn.weight_norm(ConvLayer(net['conv3_1'], 64, 3, pad=1))
    print "conv3_2: {}".format(net['conv3_2'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['fc4'] = DenseLayer(net['pool3'], num_units=100, nonlinearity=tanh)
    print "fc4: {}".format(net['fc4'].output_shape[1:])

    net['fc5'] = DenseLayer(net['fc4'], num_units=2, nonlinearity=tanh)
    print "fc5: {}".format(net['fc5'].output_shape[1:])

    net['prob'] = DenseLayer(net['fc5'], num_units=1, nonlinearity=sigmoid)
    print "prob: {}".format(net['prob'].output_shape[1:])

    return net

## Our Pytorch Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

def build(nn.Module):
  net = nn.Sequential(OrderedDict([
                                    ('merge', nn.Conv2d(4, 3, kernel_size=1,stride = 1, padding = 0)),
                                    ('conv1', nn.Conv2d(3, 32, kernel_size=3,stride = 1, padding = 1)),
                                    ('relu1', nn.ReLU()),
                                    ('pool1',  nn.MaxPool2d(4,4)),

                                    ('conv2_1', nn.Conv2d(32, 64, kernel_size=3,stride = 1, padding = 1)),
                                    ('relu2_1', nn.ReLU()),
                                    ('conv2_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                    ('relu2_2', nn.ReLU()),
                                    ('pool2', nn.MaxPool2d(2,2)),

                                    ('conv3_1', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                    ('relu3_1', nn.ReLU()),
                                    ('weight_norm3_1', nn.utils.weight_norm()), # Look into this later
                                    ('conv3_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                    ('relu3_2', nn.ReLU())
                                    ('weight_norm3_2', nn.utils.weight_norm()), # Look into this later
                                    ('pool3', nn.MaxPool2d(2,2)),

                                    ('fc4', nn.Linear(12288, 100)),
                                    ('tanh4' nn.Tanh()),
                                    ('fc5', nn.Linear(100,2)),
                                    ('tanh5' nn.Tanh()),
                                    ('fc6', nn.Linear(2,1)),
                                    # ('sigmoid6' F.Sigmoid())
                                                              ]))
  return F.Sigmoid(net)

class DiscriminatorNet(nn.Module):
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    
    self.net = nn.Sequential(OrderedDict([
                                            ('merge', nn.Conv2d(4, 3, kernel_size=1,stride = 1, padding = 0)),
                                            ('conv1', nn.Conv2d(3, 32, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu1', nn.ReLU()),
                                            ('pool1',  nn.MaxPool2d(4,4)),

                                            ('conv2_1', nn.Conv2d(32, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu2_1', nn.ReLU()),
                                            ('conv2_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu2_2', nn.ReLU()),
                                            ('pool2', nn.MaxPool2d(2,2)),

                                            ('conv3_1', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu3_1', nn.ReLU()),
                                            ('weight_norm3_1', nn.utils.weight_norm()), # Look into this later
                                            ('conv3_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu3_2', nn.ReLU())
                                            ('weight_norm3_2', nn.utils.weight_norm()), # Look into this later
                                            ('pool3', nn.MaxPool2d(2,2)),

                                            ('fc4', nn.Linear(12288, 100)),
                                            ('tanh4' F.Tanh()),
                                            ('fc5', nn.Linear(100,2)),
                                            ('tanh5' F.Tanh()),
                                            ('fc6', nn.Linear(2,1)),
                                            # ('sigmoid6' F.Sigmoid())
                                                                      ]))

  def forward(self,x):
    x = self.net(x)
    return F.sigmoid(x)



    # -----------------------------------------------------------------------------------------------------------


    # class Discriminator(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()

    #     self.merge = nn.Conv2d(4, 3, kernel_size=1,stride = 1, padding = 0)
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3,stride = 1, padding = 1)        
    #     self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3,stride = 1, padding = 1)
    #     self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)
    #     self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)
    #     self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)

    #     self.pool1 = nn.MaxPool2d(4,4) #64X48X32
    #     self.pool2 = nn.MaxPool2d(2,2)
    #     self.pool3 = nn.MaxPool2d(2,2)

    #     # self.bn1   = nn.BatchNorm2d(32)
    #     # self.bn2   = nn.BatchNorm2d(64)
    #     # self.bn3   = nn.BatchNorm2d(128)
    #     # self.bn4   = nn.BatchNorm2d(256)
    #     # self.bn5   = nn.BatchNorm2d(512)

    #      self.fc4 = nn.Linear(1*1*512, 100)
    #      self.fc5 = nn.Linear(,2)
    #      self.prob = nn.Linear(,1)

    # def forward(self, x):

    #    # Convolution 1 
    #     x = F.relu((self.conv1(self.merge)))
    #    # Pooling 1 
    #     x = self.pool1(x)
    #    # Convolution 2_1 
    #     x = F.relu((self.conv2_1(x)))
    #    # Convolution 2_2
    #     x = F.relu((self.conv2_2(x)))
    #    # Pooling 3 
    #     x = self.pool2(x)
    #    # Convolution 3_1 
    #     x = F.relu((self.conv3_1(x)))
    
    #   # Convolution 5 
    #     x = F.relu((self.conv3_2(x)))
    #    # Pooling 5 
    #     x = self.pool3(x)

    #     # reshape x from 4D to 2D, before reshape, x.size() = N*C*H*W, after reshape, x.size() = N*D
    #     x = x.view(x.size(0), -1)

    #     # x = x.view(-1, 1*1*512)
    #     x = F.Tanh(self.fc4(x))
    #     x = F.Tanh(self.fc5(x))
    #     x = F.Sigmoid(self.prob(x))
                
    #     return x