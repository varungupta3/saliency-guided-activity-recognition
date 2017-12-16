import torch
from torch.utils.data import Dataset
import numpy as np

class Salicon_loader(Dataset):
    def __init__(self,data,mask):
        
        self.data = data

        #reshape to N*C*H*W
        # self.data = self.data.reshape(self.data.shape[0],3,48,48)   
        self.data = self.data.transpose([0,3,1,2])     
        self.mask = mask    

       
    def __getitem__(self, index, bgr_mean=np.array([103.939, 116.779, 123.68])):
        #every time the data loader is called, it will input a index, 
        #the getitem function will return the image based on the index
        #the maximum index number is defined in __len__ method below
        #for each calling, you could do the image preprocessing, flipping or cropping
        
        img = self.data[index,:,:,:]
        img = img[::-1,:,:]
        
        self.mean_val = bgr_mean
        # self.std = np.std(img,axis=0)

        # use broadcasting to vectorizely normalize image
        img = (img - self.mean_val.reshape(1,3,1,1))#/(self.std.reshape(1,3,1,1))
        mask = self.mask[index,:,:]

        # convert numpy array to torch tensor variable
        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))        
        
        return img,mask

    def __len__(self):
        #this function define the upper bound of input index
        #it's usually set to the data image number

        return self.data.shape[0]
        # return 2000