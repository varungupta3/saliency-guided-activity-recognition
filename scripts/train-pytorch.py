from __future__ import print_function
from __future__ import division

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable

# Other python imports
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cPickle as pickle
import pdb

# Other py files imports
from constants import *
from models import *
from config import *
from Salicon_loader import *


#-------------------------------------------------------------
# Argument parsing and exporting torch to cuda.
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Salicon Data Processing
#-------------------------------------------------------------
                # Data Loaded from pickle files as image containers. 
# print ('Loading training data...')
# with open(TRAIN_DATA_DIR, 'rb') as f:
#     train_data = pickle.load(f)
# print ('-->done!')

# print ('Loading validation data...')    
# with open(VAL_DATA_DIR, 'rb') as f:
#     validation_data = pickle.load(f)
# print ('-->done!')

                # Getting the SALICON image and ground truth fixation maps in the form of a numpy array (list)

# trainimages = []    
# trainmask = []
# testimages = []
# testmask = []

# for idx1 in range(0,len(train_data)):
#     trainimages.append(train_data[idx1].image.data)
#     trainmask.append(train_data[idx1].saliency.data)

# for idx2 in range(0,len(validation_data)):
#     testimages.append(validation_data[idx2].image.data)
#     testmask.append(validation_data[idx2].saliency.data)
    
                #  Saving the relevant data - images and saliency maps and loading them for usage

# np.save('trainimages',trainimages)
# np.save('trainmask',trainmask)
# np.save('testimages',testimages)
# np.save('testmask',testmask)


trainimages = np.load('trainimages.npy')
trainmasks = np.load('trainmask.npy')
testimages = np.load('testimages.npy')
testmasks = np.load('testmask.npy')

    # Load Training Data via dataloader object (Salicon_loader.py). Data for the network is Images and the ground truth label is saliency map.
train_dataloader_obj = Salicon_loader(trainimages,trainmasks)    
trainloader = DataLoader(train_dataloader_obj, batch_size=args.batch_size, shuffle=True, num_workers=2)
  
    # Load Testing Data via dataloader object (Salicon_loader.py).
test_dataloader_obj = Salicon_loader(testimages,testmasks)
testloader = DataLoader(test_dataloader_obj, batch_size=args.test_batchsize, shuffle=False, num_workers=2)

pdb.set_trace()

# Choosing a Network Architecture
# --------------------------------------------------------------

print (" 2.1 --> Building the network with Salgan model")

        # Deploying the generator network model from models.py
g_net = Generator()
        # Deploying the discriminator network model from models.py
d_net = Discriminator()

BCELoss= nn.BCELoss().cuda()
#SalganLoss 
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.wd) 

if args.cuda:
    g_net.cuda()
    d_net.cuda()

 # Exponential learning rate decay
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    decay_factor = (args.min_lr / args.lr)**(epoch//5)
    lr = args.lr * decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    net.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            pdb.set_trace()
        data, target = Variable(data), Variable(target)

        
        output = net(data)
        output = output.squeeze()
        # pdb.set_trace()

        # Prediction for the 100 batch input, reshaped to a (100,36) matrix
        out = output.view(output.size(0),-1)
       
         # For clasification we use the regions in the mask which are 0,1. We extract these corresponding locations from both the predicted mask and the ground truth mask.
        tar = target.view(target.size(0),-1)

        mask = tar.ne(2)
        cls_pred = torch.masked_select(out,mask)
        cls_gt = torch.masked_select(tar,mask)
        # cls_pred = cls_pred.ge(0.5).type(torch.FloatTensor)
       
        # pdb.set_trace()
        loss = loss_fn(cls_pred, cls_gt)
        class_loss.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0]))
        


for iter in range(1, args.epochs+1):
    adjust_learning_rate(optimizer, iter)
    train(iter)
    if iter%args.log_interval == 0:        
        test()