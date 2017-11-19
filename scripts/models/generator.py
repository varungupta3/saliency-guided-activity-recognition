from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import sigmoid
import lasagne
import cPickle
import vgg16
from constants import PATH_TO_VGG16_WEIGHTS


def set_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
    # Set out weights
    vgg16 = cPickle.load(open(path_to_model_weights))
    num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
    lasagne.layers.set_all_param_values(net['conv5_3'], vgg16['param values'][:num_elements_to_set])


def build_encoder(input_height, input_width, input_var):
    encoder = vgg16.build(input_height, input_width, input_var)
    set_pretrained_weights(encoder)
    return encoder


def build_decoder(net):
    net['uconv5_3']= ConvLayer(net['conv5_3'], 512, 3, pad=1)
    print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['upool4'] = Upscale2DLayer(net['uconv5_1'], scale_factor=2)
    print "upool4: {}".format(net['upool4'].output_shape[1:])

    net['uconv4_3'] = ConvLayer(net['upool4'], 512, 3, pad=1)
    print "uconv4_3: {}".format(net['uconv4_3'].output_shape[1:])

    net['uconv4_2'] = ConvLayer(net['uconv4_3'], 512, 3, pad=1)
    print "uconv4_2: {}".format(net['uconv4_2'].output_shape[1:])

    net['uconv4_1'] = ConvLayer(net['uconv4_2'], 512, 3, pad=1)
    print "uconv4_1: {}".format(net['uconv4_1'].output_shape[1:])

    net['upool3'] = Upscale2DLayer(net['uconv4_1'], scale_factor=2)
    print "upool3: {}".format(net['upool3'].output_shape[1:])

    net['uconv3_3'] = ConvLayer(net['upool3'], 256, 3, pad=1)
    print "uconv3_3: {}".format(net['uconv3_3'].output_shape[1:])

    net['uconv3_2'] = ConvLayer(net['uconv3_3'], 256, 3, pad=1)
    print "uconv3_2: {}".format(net['uconv3_2'].output_shape[1:])

    net['uconv3_1'] = ConvLayer(net['uconv3_2'], 256, 3, pad=1)
    print "uconv3_1: {}".format(net['uconv3_1'].output_shape[1:])

    net['upool2'] = Upscale2DLayer(net['uconv3_1'], scale_factor=2)
    print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2_2'] = ConvLayer(net['upool2'], 128, 3, pad=1)
    print "uconv2_2: {}".format(net['uconv2_2'].output_shape[1:])

    net['uconv2_1'] = ConvLayer(net['uconv2_2'], 128, 3, pad=1)
    print "uconv2_1: {}".format(net['uconv2_1'].output_shape[1:])

    net['upool1'] = Upscale2DLayer(net['uconv2_1'], scale_factor=2)
    print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1_2'] = ConvLayer(net['upool1'], 64, 3, pad=1,)
    print "uconv1_2: {}".format(net['uconv1_2'].output_shape[1:])

    net['uconv1_1'] = ConvLayer(net['uconv1_2'], 64, 3, pad=1)
    print "uconv1_1: {}".format(net['uconv1_1'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv1_1'], 1, 1, pad=0,nonlinearity=sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def build(input_height, input_width, input_var):
    encoder = build_encoder(input_height, input_width, input_var)
    generator = build_decoder(encoder)
    return generator




## Our Pytorch Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import cPickle
import vgg16
from constants import PATH_TO_VGG16_WEIGHTS


# Have to remember to do this in train.py. Basically load the pretrained vgg16 model 
def set_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
    # Set out weights
    vgg16 = cPickle.load(open(path_to_model_weights))
    num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
    lasagne.layers.set_all_param_values(net['conv5_3'], vgg16['param values'][:num_elements_to_set])


def build_encoder():
    encoder = vgg16.build()
    set_pretrained_weights(encoder) # builds my vgg16 model that makes sure the first 3 layers i.e 20 parameters are not trainable. In train.py make sure to not backpropagate loss to the 3rd layer.
    return encoder


def build_decoder():

    net = nn.Sequential(OrderedDict([

              ('uconv5_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_1', F.relu()),              
              ('uconv5_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_2', F.relu()),
              ('uconv5_1', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_3', F.relu()),
              
              ('upool4', nn.modules.upsampling.Unpsample(scale_factor=2,mode='nearest')),

              ('uconv4_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_1', F.relu()),              
              ('uconv4_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_2', F.relu()),
              ('uconv4_1', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_3', F.relu()),
              
              ('upool3', nn.modules.upsampling.Unpsample(scale_factor=2,mode='nearest')),

              ('uconv3_3', nn.Conv2d(512, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_1', F.relu()),              
              ('uconv3_2', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_2', F.relu()),
              ('uconv3_1', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_3', F.relu()),
              
              ('upool2', nn.modules.upsampling.Unpsample(scale_factor=2,mode='nearest')), 

              ('uconv2_2', nn.Conv2d(256, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_1', F.relu()),
              ('uconv2_1', nn.Conv2d(128, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_2', F.relu()),
              ('upool1', nn.modules.upsampling.Unpsample(scale_factor=2,mode='nearest')),

              ('uconv1_2', nn.Conv2d(128, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1', F.relu()),
              ('uconv1_1', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1', F.relu()),
              ('output', nn.Conv2d(64, 1, kernel_size=3,stride = 1, padding = 1))             
              
            ]))

    return net


def build():
    encoder = build_encoder()
    generator = build_decoder(encoder)
    return generator


