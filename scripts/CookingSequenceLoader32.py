import torch
from torch.utils.data import Dataset
import numpy as np

class CookingSequentialLoader(Dataset):
    def __init__(self,images,actions,objects,saliency):
        
        self.images = images

        #reshape to N*C*H*W          
        self.images = self.images.transpose([0,3,1,2])    

        self.actions = actions        
        self.objects = objects
        self.saliency = saliency
       
    def __getitem__(self, index, bgr_mean=np.array([103.939, 116.779, 123.68])):
        #every time the data loader is called, it will input a index, 
        #the getitem function will return the image based on the index
        #the maximum index number is defined in __len__ method below
        #for each calling, you could do the image preprocessing, flipping or cropping
        
        self.mean_val = bgr_mean
        # self.std = np.std(img,axis=0)
        index = np.random.choice(self.images.shape[0]-32,1)[0]

        img = self.images[index:index+32,:,:,:]
        img = img[:,::-1,:,:]      
        

        # use broadcasting to vectorizely normalize image
        img = (img - self.mean_val.reshape(1,3,1,1))#/(self.std.reshape(1,3,1,1))
        act = self.actions[index:index+32].reshape(-1,)
        obj = self.objects[index:index+32].reshape(-1,)
        saliency = self.saliency[index:index+32,:,:]

        # convert numpy array to torch tensor variable
        img = torch.from_numpy(img.astype(np.float32))
        act = torch.from_numpy(act).type(torch.LongTensor)
        obj = torch.from_numpy(obj).type(torch.LongTensor)
        saliency = torch.from_numpy(saliency.astype(np.float32))        
        
        return img,act,obj,saliency

    def __len__(self):
        #this function define the upper bound of input index
        #it's usually set to the data image number

        # return self.images.shape[0]-32
        return 100


# class SubsetRandomSampler(Sampler):
#     """Samples elements randomly from a given list of indices, without replacement.

#     Arguments:
#         indices (list): a list of indices
#     """

#     def __init__(self, indices):
#         self.indices = indices

#     def __iter__(self):
#         return (self.indices[i] for i in self.indices)

#     def __len__(self):
#         return len(self.indices)
