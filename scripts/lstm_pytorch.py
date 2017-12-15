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
from Salicon_loader import *
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
# # Salicon Data Processing
# #-------------------------------------------------------------

# trainimages = np.load("../../Cookingdata.npy")

trainimages = np.load(trainImagesPath)
trainmasks = np.load(trainMasksPath)
testimages = np.load(testImagesPath)
testmasks = np.load(testMasksPath)
# pdb.set_trace()
    # Load Training Data via dataloader object (dataloader_lstm.py). Data for the network is Images and the ground truth label is saliency map.
train_dataloader_obj = Salicon_loader(trainimages,trainmasks)    
trainloader = DataLoader(train_dataloader_obj, batch_size=args.batch_size, shuffle=False, num_workers=2)
  
    # Load Testing Data via dataloader object (dataloader_lstm.py).
test_dataloader_obj = Salicon_loader(testimages,testmasks)
testloader = DataLoader(test_dataloader_obj, batch_size=args.test_batchsize, shuffle=False, num_workers=2)

# pdb.s()

# Choosing a Network Architecture
# --------------------------------------------------------------

print (" 2.1 --> Building the network for LSTM")

if args.predict_saliency:

            # Deploying the generator network model using EgoNet
    # The network architecture for the encoder using VGG16 model weights pretrained on ImageNet data from the pytorch model zoo.
    # model = EgoNet.EgoNet
    model = Generator()
    # Loading the pretrained weights from the EgoNet Caffe model to the computational graph created by the network above.
    # model.load_state_dict(torch.load('EgoNet.pth'))       
    pdb.set_trace()
    if args.cuda:
        model.cuda()   

else:
    cnn_feat = CNNFeatureExtractor()
    if args.cuda:
        cnn_feat.cuda()



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



def train(epoch):
    if args.predict_saliency:
        model.eval()
    else:
        cnn_feat.train()
    

    for batch_idx, (image, mask) in enumerate(trainloader):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
            # pdb.set_trace()
        image = image.squeeze()
        # image_masked = image_masked.squeeze()
        image, mask = Variable(image), Variable(mask)        
                       
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

            # Using the true saliency maps to generate fixation maps on the images to evalute LSTM  
        else:              
            images = (image.cpu().data.numpy().transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]
            true_saliencies= (mask.cpu().data.numpy())#.transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]            
            image_fixated = (images*(np.repeat(true_saliencies[:,:,:,np.newaxis],3,axis=3)))/255.0
            # pdb.set_trace()

            if PLOT_FLAG:
                plot_images(images, true_saliencies)  

                # Passing the masked images to extract features using a CNN for LSTM sequence analysis. Will have 512 channels downsampled by a factor of 5 at the end of this.
            image_fixated = image_fixated.astype(np.float32)
            image_input = Variable(torch.from_numpy(image_fixated.transpose([0,3,1,2])).cuda())    
            feat_extracted = cnn_feat(image_input)
            # pdb.set_trace()
                # Performing global average poooling per channel of the image
            feat_pooled = F.avg_pool2d(feat_extracted,feat_extracted.size()[2:])

                # Reshaping the feature map which is Nx1x1x512 into a vector (Nx512) for passing to LSTM
            feat_vector = feat_pooled.squeeze()


            pdb.set_trace()

            # feat_vec = [i.unsqueeze(0) for i in feat_vector]
            feat = feat_vector.unsqueeze(1).float()
                # LSTM Starts here
            lstm = nn.LSTM(512,512).cuda()
                # hidden states
            # initialize the hidden state.
            hidden = (Variable(torch.randn(1, 1, 512).cuda()),Variable(torch.randn(1, 1, 512).cuda()))

            # for i in feat_vec:
            #     out,hidden = lstm(i.view(1, 1, -1))
            out,hidden = lstm(feat,hidden)




            pdb.set_trace()       


for epoch in range(1, args.epochs+1):

 
    train(epoch)
    