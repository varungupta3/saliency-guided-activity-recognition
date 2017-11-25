# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer
from layers import RGBtoBGRLayer


def build(inputHeight, inputWidth, input_var):
    """
    Bulid only Convolutional part of the VGG-16 Layer model, all fully connected layers are removed.
    First 3 group of ConvLayers are fixed (not trainable).

    :param input_layer: Input layer of the network.
    :return: Dictionary that contains all layers.
    """

    net = {'input': InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)}
    print "Input: {}".format(net['input'].output_shape[1:])

    net['bgr'] = RGBtoBGRLayer(net['input'])

    net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
    net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
    print "conv1_1: {}".format(net['conv1_1'].output_shape[1:])

    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
    net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
    print "conv1_2: {}".format(net['conv1_2'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
    net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
    print "conv2_1: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
    net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
    print "conv2_2: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
    net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
    print "conv3_1: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
    net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
    print "conv3_2: {}".format(net['conv3_2'].output_shape[1:])

    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
    net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
    print "conv3_3: {}".format(net['conv3_3'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
    net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
    print "conv4_1: {}".format(net['conv4_1'].output_shape[1:])

    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
    net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
    print "conv4_2: {}".format(net['conv4_2'].output_shape[1:])

    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'].add_param(net['conv4_3'].W, net['conv4_3'].W.get_value().shape)
    net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
    print "conv4_3: {}".format(net['conv4_3'].output_shape[1:])

    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    print "pool4: {}".format(net['pool4'].output_shape[1:])

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
    net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
    print "conv5_1: {}".format(net['conv5_1'].output_shape[1:])

    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
    net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
    print "conv5_2: {}".format(net['conv5_2'].output_shape[1:])

    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape)
    net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape)
    print "conv5_3: {}".format(net['conv5_3'].output_shape[1:])

    return net

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Our Pytorch Implementation


import torch
import torch.nn as nn
import torch.nn.functional as F

class build(nn.Module):
  def __init__(self):
    super(build, self).__init__()

    self.net = nn.Sequential(OrderedDict([

              ('bgr', RGBtoBGRLayer()),  # Remember to remove this line as the dataloader will give in NxCxHxW
              ('conv1_1', nn.Conv2d(3, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1', nn.ReLU()),
              ('conv1_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1', nn.ReLU()),
              ('pool1',  nn.MaxPool2d(2,2)),

              ('conv2_1', nn.Conv2d(64, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_1', nn.ReLU()),
              ('conv2_2', nn.Conv2d(128, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_2', nn.ReLU()),
              ('pool2', nn.MaxPool2d(2,2)),

              ('conv3_1', nn.Conv2d(128, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_1', nn.ReLU()),              
              ('conv3_2', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_2', nn.ReLU()),
              ('conv3_3', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_3', nn.ReLU()),
              ('pool3', nn.MaxPool2d(2,2)),  # Remember to make trainable to false until this layer

              ('conv4_1', nn.Conv2d(256, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_1', nn.ReLU()),              
              ('conv4_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_2', nn.ReLU()),
              ('conv4_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_3', nn.ReLU()),
              ('pool4', nn.MaxPool2d(2,2)),

              ('conv5_1', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_1', nn.ReLU()),              
              ('conv5_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_2', nn.ReLU()),
              ('conv5_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_3', nn.ReLU()),
              ('pool5', nn.MaxPool2d(2,2))
              
            ]))

  def forward(self,x):
    x = self.net(x)
    return x