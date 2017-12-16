import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as trained_models
from collections import OrderedDict
from constants import *
import numpy as np
import pdb
from torch.autograd import Variable

def weight_tensor_from_np(weight,gradflag = True):
  return torch.nn.Parameter(torch.from_numpy(weight).cuda(),requires_grad=gradflag)

def bias_tensor_from_np(bias,gradflag = True):
  return torch.nn.Parameter(torch.from_numpy(bias).cuda().view(1,bias.shape[0],1,1),requires_grad=gradflag)



class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    gen_weights = np.load(genWeightsPath)
    gen_weight_list_split = [int(i.split('_')[1]) for i in gen_weights.keys()]
    gen_weight_list_split.sort(reverse=True)
    gen_weight_list_order = ["arr_" + str(i) for i in gen_weight_list_split]
                            # Conv 1_1
    self.W1_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b1_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
                            # Conv 1_2
    self.W1_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b1_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)

                            # Conv 2_1
    self.W2_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b2_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
                            # Conv 2_2
    self.W2_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b2_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)

                            # Conv 3_1
    self.W3_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b3_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
                            # Conv 3_2
    self.W3_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b3_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
                            # Conv 3_3
    self.W3_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)
    self.b3_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()],False)

                            # Conv 4_1
    self.W4_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b4_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # Conv 4_2
    self.W4_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b4_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # Conv 4_3
    self.W4_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b4_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # Conv 5_1
    self.W5_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b5_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # Conv 5_2
    self.W5_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b5_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # Conv 5_3
    self.W5_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.b5_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])


                            # uConv 5_3
    self.uW5_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub5_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 5_2
    self.uW5_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub5_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 5_1
    self.uW5_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub5_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # uConv 4_3
    self.uW4_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub4_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 4_2
    self.uW4_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub4_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 4_1
    self.uW4_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub4_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # uConv 3_3
    self.uW3_3 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub3_3 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 3_2
    self.uW3_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub3_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 3_1
    self.uW3_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub3_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # uConv 2_2
    self.uW2_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub2_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 2_1
    self.uW2_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub2_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # uConv 1_2
    self.uW1_2 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub1_2 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
                            # uConv 1_1
    self.uW1_1 = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.ub1_1 = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])

                            # Output
    self.op_W = weight_tensor_from_np(gen_weights[gen_weight_list_order.pop()])
    self.op_b = bias_tensor_from_np(gen_weights[gen_weight_list_order.pop()])


    # self.relu = F.relu()
    # self.maxpool = nn.MaxPool2d(2,2,return_indices=True)
    # self.maxunpool = nn.MaxUnpool2d(1)

    # ---------------------------------------------------------------------------------------------------------
    #                               ENCODER ARCHITECTURE
    # ---------------------------------------------------------------------------------------------------------
  
  def forward(self, x):
                         # Conv 1_1
    x = F.conv2d(x, self.W1_1, stride = 1, padding = 1)
    x = x.add(self.b1_1)
    x = F.relu(x)    
                         # Conv 1_2
    x = F.conv2d(x, self.W1_2, stride = 1, padding = 1)
    x = x.add(self.b1_2)
    x = F.relu(x)
                         # Pool 1
    x = F.max_pool2d(x,2)

                        # Conv 2_1
    x = F.conv2d(x, self.W2_1, stride = 1, padding = 1)
    x = x.add(self.b2_1)
    x = F.relu(x)
                        # Conv 2_2
    x = F.conv2d(x,self.W2_2,stride = 1, padding = 1)
    x = x.add(self.b2_2)
    x = F.relu(x)
                        # Pool 2
    x = F.max_pool2d(x,2)

                          # Conv 3_1
    x = F.conv2d(x, self.W3_1, stride = 1, padding = 1)
    x = x.add(self.b3_1)
    x = F.relu(x)
                        # Conv 3_2
    x = F.conv2d(x, self.W3_2, stride = 1, padding = 1)
    x = x.add(self.b3_2)
    x = F.relu(x)
                        # Conv 3_3
    x = F.conv2d(x, self.W3_3, stride = 1, padding = 1)
    x = x.add(self.b3_3)
    x = F.relu(x)
                        # Pool 3
    x = F.max_pool2d(x,2)

                        # Conv 4_1
    x = F.conv2d(x,self.W4_1, stride = 1, padding = 1)
    x = x.add(self.b4_1)
    x = F.relu(x)
                        # Conv 4_2
    x = F.conv2d(x, self.W4_2, stride = 1, padding = 1)
    x = x.add(self.b4_2)
    x = F.relu(x)
                        # Conv 4_3
    x = F.conv2d(x, self.W4_3, stride = 1, padding = 1)
    x = x.add(self.b4_3)
    x = F.relu(x)
                        # Pool 4
    x = F.max_pool2d(x,2)

                        # Conv 5_1
    x = F.conv2d(x, self.W5_1, stride = 1, padding = 1)
    x = x.add(self.b5_1)
    x = F.relu(x)
                        # Conv 5_2
    x = F.conv2d(x, self.W5_2, stride = 1, padding = 1)
    x = x.add(self.b5_2)
    x = F.relu(x)
                        # Conv 5_3
    x = F.conv2d(x, self.W5_3, stride = 1, padding = 1)
    x = x.add(self.b5_3)
    x = F.relu(x)
              
  
    # -------------------------------------------------------------------------------------------------------------
    #                                        DECODER ARCHITECTURE
    # -------------------------------------------------------------------------------------------------------------

                         # uConv 5_3
    x = F.conv2d(x, self.uW5_3, stride = 1, padding = 1)
    x = x.add(self.ub5_3)
    x = F.relu(x)
                         # uConv 5_2
    x = F.conv2d(x, self.uW5_2, stride = 1, padding = 1)
    x = x.add(self.ub5_2)
    x = F.relu(x)
                        # uConv 5_1
    x = F.conv2d(x, self.uW5_1, stride = 1, padding = 1)
    x = x.add(self.ub5_1)
    x = F.relu(x)
                         # Upool 4
    x = F.upsample(x, scale_factor=2)

                        # uConv 4_3
    x = F.conv2d(x, self.uW4_3, stride = 1, padding = 1)
    x = x.add(self.ub4_3)
    x = F.relu(x)
                        # uConv 4_2
    x = F.conv2d(x, self.uW4_2, stride = 1, padding = 1)
    x = x.add(self.ub4_2)
    x = F.relu(x)
                        # uConv 4_1
    x = F.conv2d(x, self.uW4_1, stride = 1, padding = 1)
    x = x.add(self.ub4_1)
    x = F.relu(x)
                        # uPool 3
    x = F.upsample(x, scale_factor=2)

                        # uConv 3_3
    x = F.conv2d(x, self.uW3_3, stride = 1, padding = 1)
    x = x.add(self.ub3_3)
    x = F.relu(x)
                        # uConv 3_2
    x = F.conv2d(x, self.uW3_2, stride = 1, padding = 1)
    x = x.add(self.ub3_2)
    x = F.relu(x)
                        # uConv 3_1
    x = F.conv2d(x, self.uW3_1, stride = 1, padding = 1)
    x = x.add(self.ub3_1)
    x = F.relu(x)
                        # uPool 2
    x = F.upsample(x, scale_factor=2)

                        # uConv 2_2
    x = F.conv2d(x, self.uW2_2, stride = 1, padding = 1)
    x = x.add(self.ub2_2)
    x = F.relu(x)
                        # uConv 2_1
    x = F.conv2d(x, self.uW2_1, stride = 1, padding = 1)
    x = x.add(self.ub2_1)
    x = F.relu(x)
                        # uPool 1
    x = F.upsample(x, scale_factor=2)

                        # uConv 1_2
    x = F.conv2d(x, self.uW1_2, stride = 1, padding = 1)
    x = x.add(self.ub1_2)
    x = F.relu(x)
                        # uConv 1_1
    x = F.conv2d(x, self.uW1_1, stride = 1, padding = 1)
    x = x.add(self.ub1_1)
    x = F.relu(x)

                        # Output to 1 Channel
    x = F.conv2d(x, self.op_W, stride = 1, padding = 0)
    x = x.add(self.op_b)
    x = F.sigmoid(x)

    return x


class Discriminator(nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()

    disc_weights = np.load(discWeightsPath)
    disc_weight_list_split = [int(i.split('_')[1]) for i in disc_weights.keys()]
    disc_weight_list_split.sort(reverse=True)
    disc_weight_list_order = ["arr_" + str(i) for i in disc_weight_list_split]
                             # Conv 1_1
    self.W1_1 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b1_1 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                            # Conv 1_2
    self.W1_2 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b1_2 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])

                            # Conv 2_1
    self.W2_1 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b2_1 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                            # Conv 2_2
    self.W2_2 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b2_2 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])

                            # Conv 3_1
    self.W3_1 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b3_1 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                        # Removing Weight norm layer weights
    disc_weights[disc_weight_list_order.pop()]
                            # Conv 3_2
    self.W3_2 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b3_2 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                        # Removing Weight norm layer weights
    disc_weights[disc_weight_list_order.pop()]

                            # FC 4
    self.W4 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b4 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                            # FC 5
    self.W5 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b5 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
                            # FC 6
    self.W6 = weight_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    self.b6 = bias_tensor_from_np(disc_weights[disc_weight_list_order.pop()])
    
    
  def forward(self, x):
    

                         # Conv 1_1
    x = F.conv2d(x,self.W1_1, stride = 1, padding = 0)
    x = x.add(self.b1_1)
    x = F.relu(x)    
                         # Conv 1_2
    x = F.conv2d(x,self.W1_2, stride = 1, padding = 1)
    x = x.add(self.b1_2)
    x = F.relu(x)
                         # Pool1
    x = F.max_pool2d(x,4)

                         # Conv 2_1
    x = F.conv2d(x,self.W2_1, stride = 1, padding = 1)
    x = x.add(self.b2_1)
    x = F.relu(x)
                         # Conv 2_2
    x = F.conv2d(x,self.W2_2, stride = 1, padding = 1)
    x = x.add(self.b2_2)
    x = F.relu(x)
                         # Pool2
    x = F.max_pool2d(x,2)

                         # Conv 3_1
    x = F.conv2d(x,self.W3_1, stride = 1, padding = 1)
    x = x.add(self.b3_1)
    x = F.relu(x)                         
                         # Conv 3_2
    x = F.conv2d(x,self.W3_2, stride = 1, padding = 1)
    x = x.add(self.b3_2)
    x = F.relu(x)                          
                          # Pool 3
    x = F.max_pool2d(x,2)
    # pdb.set_trace()

    x = x.view(x.size(0), -1)

                          # Fc 4
    x = F.linear(x,torch.t(self.W4),bias=self.b4.view(-1))
    # x = x.add(self.b4)
    x = F.tanh(x)
                          # Fc 5
    x = F.linear(x,torch.t(self.W5),bias=self.b5.view(-1))
    # x = x.add(self.b5)
    x = F.tanh(x)
                          # Fc 6
    x = F.linear(x,torch.t(self.W6),bias=self.b6.view(-1))
    # x = x.add(self.b6)
    x = F.sigmoid(x)
    return x   

#pre-define weights and biases
# gen_weights = np.load(genWeightsPath)
# gen_weight_list_split = [int(i.split('_')[1]) for i in gen_weights.keys()]
# gen_weight_list_split.sort(reverse=True)
# gen_weight_list_order = ["arr_" + str(i) for i in gen_weight_list_split]


# disc_weights = np.load(discWeightsPath)
# disc_weight_list_split = [int(i.split('_')[1]) for i in disc_weights.keys()]
# disc_weight_list_split.sort(reverse=True)
# disc_weight_list_order = ["arr_" + str(i) for i in disc_weight_list_split]
