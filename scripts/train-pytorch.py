from __future__ import print_function
from __future__ import division
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np

import data
import pdb
from models import *
from Cifar10_loader import *
from config import *
import math
import matplotlib.pyplot as plt


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# CIFAR-10 Data Processing
#-------------------------------------------------------------
    # Data Loaded as numpy 
traindata,trainmask,trainlabels,trainrows,traincols,trainwidth = data.load_train("../cifar10_transformed/imgs/", "../cifar10_transformed/masks/")
testdata,testmask,testlabels,testrows,testcols,testwidth = data.load_test("../cifar10_transformed/imgs/", "../cifar10_transformed/masks/")

# pdb.set_trace()

    # Load Training Data 
train_dataloader_obj = Cifar10_loader(traindata,trainmask,trainlabels,trainrows,traincols,trainwidth)    
trainloader = DataLoader(train_dataloader_obj, batch_size=args.batch_size, shuffle=True, num_workers=2)
  
    # Load Testing Data
test_dataloader_obj = Cifar10_loader(testdata,testmask,testlabels,testrows,testcols,testwidth)
testloader = DataLoader(test_dataloader_obj, batch_size=args.test_batchsize, shuffle=False, num_workers=2)

# Choosing a Network Architecture
# --------------------------------------------------------------

if args.bce:
    print (" 2.1 --> Building only Generative Model. No adversarial Training. BCE Loss")
    net = generator.build()
    loss_fn= nn.BCELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.wd)
    class_loss = []

elif args.salgan:
    print (" 2.2 --> Adversaraial Training using SALGAN loss")
    
    net = ModelSalgan()
   
    loss_fn1= nn.BCELoss().cuda()
    loss_fn= nn.SmoothL1Loss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.wd)
    regres_loss = []

else:
    raise ValueError("Enter a valid network - BCELoss or SALGAN")

if args.cuda:
    net.cuda()

 # Exponential learning rate decay
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    decay_factor = (args.min_lr / args.lr)**(epoch//5)
    lr = args.lr * decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    net.train()

    if args.bce:        

        for batch_idx, (data, target,_,_,_,_) in enumerate(trainloader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                # pdb.set_trace()
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

    elif args.salgan:  

        


for iter in range(1, args.epochs+1):

    adjust_learning_rate(optimizer, iter)
    train(iter)
    if iter%args.log_interval == 0:        
        test()





# pdb.set_trace()