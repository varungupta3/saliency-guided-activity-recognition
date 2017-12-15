import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as trained_models
from collections import OrderedDict
import numpy as np
from constants import *
from torch.autograd import Variable
import pdb

# class CNNFeatureExtractor(nn.Module):
#   def __init__(self):
#     super(CNNFeatureExtractor, self).__init__()

#     # The network architecture for the encoder using VGG16 model weights pretrained on ImageNet data from the pytorch model zoo.
#     vgg16 = trained_models.vgg16(pretrained=True)
#     # Settting the Pretrained weights of the convolutional layers. vgg16.features.state_dict().keys() will give the 26 parameters (weights and biases of the conv)
#     model = vgg16.features 
#     # Extracting the First 3 convolutional layer and their parameters (weights and bias) to build our encoder model.
#     encoder_fixed = nn.Sequential(*list(model.children())[:-14])
#     # Fixing the paramters (weights and biases) of the first three layers
#     for param in encoder_fixed.parameters():
#       param.requires_grad = False
#     # Generating the rest of the encoder model by extracting the pretrained paramters(training allowed).
#     encoder_trainable = nn.Sequential(*list(model.children())[-14:-1])
#     # Combining the two encoder setups to generate the complete architecture with trainable and not trainable parameters.
#     total_layers = list(encoder_fixed.children())
#     total_layers.extend(list(encoder_trainable.children()))

#     # The complete encoder architecture
#     self.encoder = nn.Sequential(*total_layers)

#     def forward(self,x):
#     # Constructing the Encoder network using VGG16 with pretrained weights

#         x = self.encoder(x)
#         return x


def weight_tensor_from_np(weight):
  return Variable(torch.from_numpy(weight).cuda())

def bias_tensor_from_np(bias):
  return Variable(torch.from_numpy(bias).cuda().view(1,bias.shape[0],1,1))

class CNNFeatureExtractor(nn.Module):
  def __init__(self):
    super(CNNFeatureExtractor, self).__init__()

    # self.relu = F.relu()
    # self.maxpool = nn.MaxPool2d(2,2,return_indices=True)
    # self.maxunpool = nn.MaxUnpool2d(1)

    # ---------------------------------------------------------------------------------------------------------
    #                               ENCODER ARCHITECTURE
    # ---------------------------------------------------------------------------------------------------------
  
  def forward(self, x):
                         # Conv 1_1
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add(bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                         # Conv 1_2
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add(bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                         # Pool 1
    x = F.max_pool2d(x,2)

                        # Conv 2_1
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 2_2
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]),stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Pool 2
    x = F.max_pool2d(x,2)

                        # Conv 3_1
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 3_2
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 3_3
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Pool 3
    x = F.max_pool2d(x,2)

                        # Conv 4_1
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 4_2
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 4_3
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Pool 4
    x = F.max_pool2d(x,2)

                        # Conv 5_1
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 5_2
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)
                        # Conv 5_3
    x = F.conv2d(x, weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()]), stride = 1, padding = 1)
    x = x.add( bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()]))
    x = F.relu(x)

    return x
    
    #pre-define weights and biases
gen_weights = np.load(genWeightsPath)
gen_weight_list_split = [int(i.split('_')[1]) for i in gen_weights.keys()]
gen_weight_list_split.sort(reverse=True)
gen_weight_list_order = ["arr_" + str(i) for i in gen_weight_list_split]