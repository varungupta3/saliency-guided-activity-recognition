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
from pretrained_lstm_feature_extractor import *
from pretrained_models_pytorch import *
from config import *
# from Salicon_loader import *
from CookingSequenceloader import *

from dataloader_lstm import *
# import EgoNet

#-------------------------------------------------------------
# Argument parsing and exporting torch to cuda.
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PLOT_FLAG = args.plot_saliency

# Salicon Data Processing
#-------------------------------------------------------------

# trainimages = np.load("../../Cookingdata.npy")

trainimages = np.load(trainImagesPath)
trainmasks = np.load(trainMasksPath)
trainactions = np.load(trainActionsPath)
trainobjects = np.load(trainObjectsPath)

actionlist = np.load(trainActionsListPath)
objectlist = np.load(trainObjectsListPath)

# pdb.set_trace()
# testimages = np.load(testImagesPath)
# testmasks = np.load(testMasksPath)

    # Load Training Data via dataloader object (dataloader_lstm.py). Data for the network is Images and the ground truth label is saliency map.
train_dataloader_obj = CookingSequentialLoader(trainimages,trainactions,trainobjects,trainmasks)    
trainloader = DataLoader(train_dataloader_obj, batch_size=args.batch_size, shuffle=False, num_workers=2,drop_last=True)
  
    # Load Testing Data via dataloader object (dataloader_lstm.py).
# test_dataloader_obj = CookingSequentialloader(testimages,testmasks)
# testloader = DataLoader(test_dataloader_obj, batch_size=args.test_batchsize, shuffle=False, num_workers=2)

# pdb.s()

# Choosing a Network Architecture
# --------------------------------------------------------------

print (" 2.1 --> Building the network for LSTM")

lstm = LSTM()
if args.cuda:
    lstm.cuda()

if args.predict_saliency:

            # Deploying the network for saliency generation   
    # model = EgoNet.EgoNet
    # Loading the pretrained weights from the EgoNet Caffe model to the computational graph created by the network above.
    # model.load_state_dict(torch.load('EgoNet.pth')) 

    model = Generator()   
    if args.cuda:
        model.cuda()   

else:
    cnn_feat = CNNFeatureExtractor()
    if args.cuda:
        cnn_feat.cuda()

CrossEntropy = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(cnn_feat.parameters(), lr=args.lr, weight_decay = args.wd)

def plot_images(images, pred_saliency):
    for i in np.random.randint(np.shape(images)[0], size=10):
    # for i in range(32):
        fig = plt.figure(i)
        ax1 = fig.add_subplot(121)
        surf = ax1.imshow(images[i,:,:,:].astype(np.uint8))
        ax1.set_title('Image')
        # ax2 = fig.add_subplot(132)
        # surf = ax2.imshow(true_saliency[i,:,:])
        # ax2.set_title('True Saliency')
        ax3 = fig.add_subplot(122)
        surf = ax3.imshow(pred_saliency[i,:,:])
        ax3.set_title('Predicted Saliency')

        # pred_weighted_image = np.multiply(images[26,:,:,:],(np.expand_dims(pred_saliency[26,:,:],axis=3))).astype(np.uint8)
        # true_weighted_image = np.multiply(images[26,:,:,:],(np.expand_dims(pred_saliency[26,:,:],axis=3)/255.0)).astype(np.uint8)
    plt.show()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    decay_factor = (args.min_lr / args.lr)**(epoch//5)
    lr = args.lr * decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    if args.predict_saliency:
        model.eval()
        lstm.train()
    else:
        cnn_feat.train()
        lstm.train()

    correct_act = 0
    correct_obj = 0

    for batch_idx, (image,action,obj,true_saliency) in enumerate(trainloader):
        if args.cuda:
            image, action, obj, true_saliency = image.cuda(), action.cuda(), obj.cuda(), true_saliency.cuda()
            # pdb.set_trace()
        image = image.squeeze()        
        image, action, obj, true_saliency = Variable(image), Variable(action), Variable(obj), Variable(true_saliency)        
                       
            # Generate the predicted saliency map from the generator network.
        
        if args.predict_saliency:
            pred_saliency = model(image)            
            images = (image.cpu().data.numpy().transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]                
            pred_saliencies = pred_saliency.squeeze().cpu().data.numpy()
                # Element wise multiplication fixation maps
            image_fixated = images*(np.repeat(pred_saliencies[:,:,:,np.newaxis],3,axis=3))
            pdb.set_trace()

            if PLOT_FLAG:
                plot_images(images, pred_saliencies)            

            # Using the true saliency maps stacked with the images to evalute LSTM  
        else:              
            images = (image.cpu().data.numpy().transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]
            true_saliencies= (true_saliency.cpu().data.numpy())#.transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]            
            # image_fixated = (images*(np.repeat(true_saliencies[:,:,:,np.newaxis],3,axis=3)))/255.0
            # pdb.set_trace()

            if PLOT_FLAG:
                plot_images(images, true_saliencies)  

            true_saliency = true_saliency.unsqueeze(1)
                # Passing the masked images to extract features using a CNN for LSTM sequence analysis. Will have 512 channels downsampled by a factor of 5 at the end of this.
            # image_fixated = image_fixated.astype(np.float32)
            
            image_appended = torch.cat((image,true_saliency),1)
            # image_input = Variable(torch.from_numpy(image_appended.transpose([0,3,1,2])).cuda())   

            feat_extracted = cnn_feat(image_appended)
            # pdb.set_trace()
                # Performing global average poooling per channel of the image
            feat_pooled = F.avg_pool2d(feat_extracted,feat_extracted.size()[2:])
                # Reshaping the feature map which is Nx1x1x512 into a vector (Nx512) for passing to LSTM
            feat_vector = feat_pooled.squeeze()
            # pdb.set_trace() 

            feat = feat_vector.unsqueeze(1).float()
                
                # LSTM 
            act_out,obj_out = lstm(feat)

            # Creating a one-hot encoded vector for loss comparison

            # action_one_hot = torch.zeros(len(image),len(actionlist)).cuda()
            # obj_one_hot = torch.zeros(len(image),len(objectlist)).cuda()

            # action_one_hot.scatter_(1,action.data.view(-1,1),1)
            # obj_one_hot.scatter_(1,obj.data.view(-1,1),1)

            lstm_action_loss = CrossEntropy(act_out,action.view(-1))
            lstm_object_loss = CrossEntropy(obj_out,obj.view(-1))

            total_loss = lstm_action_loss + lstm_object_loss

            optimizer.zero_grad()
            # lstm_action_loss.backward(retain_graph=True)
            # lstm_object_loss.backward()
            total_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAction Loss: {:.6f}\t Object Loss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), lstm_action_loss.data[0], lstm_object_loss.data[0]))
            

            pred_act = act_out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_act += pred_act.eq(action.data.view_as(pred_act)).cpu().sum()

            pred_obj = obj_out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_obj += pred_obj.eq(obj.data.view_as(pred_obj)).cpu().sum()

            # pdb.set_trace()       
    print('\nAction Accuracy: {}/{} ({:.0f}%)\t Object  Accuracy: {}/{} ({:.0f}%)\n'
                    .format(correct_act, len(trainloader.dataset), 100. * correct_act / len(trainloader.dataset), correct_obj,
                                                    len(trainloader.dataset), 100. * correct_obj / len(trainloader.dataset)))    

for epoch in range(1, args.epochs+1):
    # adjust_learning_rate(optimizer, epoch)
    train(epoch)
    