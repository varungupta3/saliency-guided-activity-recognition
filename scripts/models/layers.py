import lasagne
import numpy as np
import theano


class RGBtoBGRLayer(lasagne.layers.Layer):
    def __init__(self, l_in, bgr_mean=np.array([103.939, 116.779, 123.68]),
                 data_format='bc01', **kwargs):
        """A Layer to normalize and convert images from RGB to BGR
        This layer converts images from RGB to BGR to adapt to Caffe
        that uses OpenCV, which uses BGR. It also subtracts the
        per-pixel mean.
        Parameters
        ----------
        l_in : :class:``lasagne.layers.Layer``
            The incoming layer, typically an
            :class:``lasagne.layers.InputLayer``
        bgr_mean : iterable of 3 ints
            The mean of each channel. By default, the ImageNet
            mean values are used.
        data_format : str
            The format of l_in, either `b01c` (batch, rows, cols,
            channels) or `bc01` (batch, channels, rows, cols)
        """
        super(RGBtoBGRLayer, self).__init__(l_in, **kwargs)
        assert data_format in ['bc01', 'b01c']
        self.l_in = l_in
        floatX = theano.config.floatX
        self.bgr_mean = bgr_mean.astype(floatX)
        self.data_format = data_format

    def get_output_for(self, input_im, **kwargs):
        if self.data_format == 'bc01':
            input_im = input_im[:, ::-1, :, :]
            input_im -= self.bgr_mean[:, np.newaxis, np.newaxis]
        else:
            input_im = input_im[:, :, :, ::-1]
            input_im -= self.bgr_mean
        return input_im



#  Our Pytorch Implementation
import numpy as np
import torch

class dataloader_obj(torch.utils.data.Dataset):
    def __init__(self,data,mean=np.array([103.939, 116.779, 123.68])):
        
        self.data = data
        #reshape to N*C*H*W
        self.data = self.data.reshape(self.data.shape[0],3,32,32)
        self.data = self.data[:,:,::-1,:]
                       
        self.mean_val = mean
        # self.std = param['std']

    def __getitem__(self, index):
        #every time the data loader is called, it will input a index, 
        #the getitem function will return the image based on the index
        #the maximum index number is defined in __len__ method below
        #for each calling, you could do the image preprocessing, flipping or cropping
        img = self.data[index,:,:,:]
        # use broadcasting to vectorizely normalize image
        img = (img - self.mean_val.reshape(1,3,1,1))#/(self.std.reshape(1,3,1,1))
        
        # convert numpy array to torch tensor variable
        img = torch.from_numpy(img.astype(np.float32))
       
        return img

    def __len__(self):
        #this function define the upper bound of input index
        #it's usually set to the data image number
        return data.shape[0] 
