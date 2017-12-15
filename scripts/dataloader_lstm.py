import torch
from torch.utils.data import Dataset
import numpy as np

class dataloader_lstm(Dataset):
    def __init__(self,data,mask):
        
        self.data = data
        self.mask = mask 

        # Tile the mask to match the channel dimensions of the image 
        self.mask = np.repeat(self.mask[:,:,:,np.newaxis],3,axis=3)

        # # Applying the saliency to images to generate fixation maps on the image
        # self.mask_applied = self.data*self.mask

        #reshape to N*C*H*W          
        # self.data = self.data.transpose([0,3,1,2])     
        # self.mask = self.mask.transpose([0,3,1,2])   

       
    def __getitem__(self, index, bgr_mean=np.array([103.939, 116.779, 123.68])):
        #every time the data loader is called, it will input a index, 
        #the getitem function will return the image based on the index
        #the maximum index number is defined in __len__ method below
        #for each calling, you could do the image preprocessing, flipping or cropping

        self.mean_val = bgr_mean
        # self.std = np.std(img,axis=0)
        
        img = self.data[index,:,:,:]
        img = img[::-1,:,:]   
        # use broadcasting to vectorizely normalize image
        img = (img - self.mean_val.reshape(1,3,1,1))#/(self.std.reshape(1,3,1,1))
        
        mask_applied = self.mask[index,:,:]
        # mask = (mask - self.mean_val.reshape(1,3,1,1))

        # Applying the saliency to images to generate fixation maps on the image
        # mask_applied = img*mask
        # mask_applied = (mask_applied - self.mean_val.reshape(1,1,3))

        # img = img.transpose([2,0,1]) 
        # mask_applied = mask_applied.transpose([2,0,1]) 

        # convert numpy array to torch tensor variable
        img = torch.from_numpy(img.astype(np.float32))
        mask_applied = torch.from_numpy(mask_applied.astype(np.float32))        
        
        return img,mask_applied

    def __len__(self):
        #this function define the upper bound of input index
        #it's usually set to the data image number

        return self.data.shape[0]