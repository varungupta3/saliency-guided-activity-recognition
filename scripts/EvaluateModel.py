# from __future__ import print_function
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

# pdb.set_trace()
# testimages = np.load(testImagesPath)
# testmasks = np.load(testMasksPath)

    # Load Training Data via dataloader object (dataloader_lstm.py). Data for the network is Images and the ground truth label is saliency map.
train_dataloader_obj = Salicon_loader(trainimages,trainmasks)    
trainloader = DataLoader(train_dataloader_obj, batch_size=args.batch_size, shuffle=False, num_workers=2,drop_last=True)
  
    # Load Testing Data via dataloader object (dataloader_lstm.py).
# test_dataloader_obj = CookingSequentialloader(testimages,testmasks)
# testloader = DataLoader(test_dataloader_obj, batch_size=args.test_batchsize, shuffle=False, num_workers=2)



# Choosing a Network Architecture
# --------------------------------------------------------------

print (" Evaluating the SALGAN network")

model = torch.load('../gen_model_final.pt') 
if args.cuda:
    model.cuda()   


def plot_images(images, saliency, imagemaps, count):
    # for i in np.random.randint(np.shape(images)[0], size=10):
    for i in range(32):
        fig = plt.figure(i)
        fig.set_size_inches(18, 8)
        ax1 = fig.add_subplot(121)
        surf = ax1.imshow(images[i,:,:,:].astype(np.uint8))               
        ax1.set_title('Original Sequence')
        
        ax2 = fig.add_subplot(122)
        surf = ax2.imshow(imagemaps[i,:,:,:].astype(np.uint8))
        ax2.set_title('Saliency Applied')

        # ax3 = fig.add_subplot(133)
        # surf = ax3.imshow(saliency[i,:,:])
        # ax3.set_title('Predicted Saliency')

        frame = 'frame{}'.format("%04d"%((count*32)+i)) 
        plt.savefig('../saliency_vid_results/res/'+frame)
        # pdb.set_trace()
    # plt.show()


def eval():
    model.eval()

    
    for batch_idx, (image,true_saliency) in enumerate(trainloader):
        if args.cuda:
            image, true_saliency = image.cuda(), true_saliency.cuda()
            
        image = image.squeeze()        
        image, true_saliency = Variable(image,volatile=True), Variable(true_saliency)        
                       
            # Generate the predicted saliency map from the generator network.        
        pred_saliency = model(image)            

        images = (image.cpu().data.numpy().transpose([0,2,3,1]) + np.array([103.939, 116.779, 123.68]).reshape(1,1,1,3))[:,:,:,::-1]                
        pred_saliencies = pred_saliency.squeeze().cpu().data.numpy()
        true_saliencies= (true_saliency.cpu().data.numpy())
            # Element wise multiplication fixation maps
        images_fixated = images*(np.repeat(pred_saliencies[:,:,:,np.newaxis],3,axis=3))
        # images_fixated = (images*(np.repeat(true_saliencies[:,:,:,np.newaxis],3,axis=3)))
        # pdb.set_trace()
    
        plot_images(images, pred_saliencies, images_fixated, batch_idx)            

            
if __name__ == '__main__':
    eval()
    
