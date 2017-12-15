from constants import *
import cPickle as pickle
import pdb
import numpy as np


# # Salicon Data Processing
# #-------------------------------------------------------------
#                 # Data Loaded from pickle files as image containers. 
print ('Loading training data...')
with open(TRAIN_DATA_DIR, 'rb') as f:
    train_data = pickle.load(f)
print ('-->done!')

print ('Loading validation data...')    
with open(VAL_DATA_DIR, 'rb') as f:
    validation_data = pickle.load(f)
print ('-->done!')
pdb.set_trace()
#                 # Getting the SALICON image and ground truth fixation maps in the form of a numpy array (list)

trainimages = []    
trainmask = []
testimages = []
testmask = []

for idx1 in range(0,len(train_data)):
    trainimages.append(train_data[idx1].image.data)
    trainmask.append(train_data[idx1].saliency.data)

for idx2 in range(0,len(validation_data)):
    testimages.append(validation_data[idx2].image.data)
    testmask.append(validation_data[idx2].saliency.data)
    
                #  Saving the relevant data - images and saliency maps and loading them for usage

np.save(trainImagesPath + 'trainimages',trainimages)
np.save(trainMasksPath + 'trainmask',trainmask)
np.save(testImagesPath + 'testimages',testimages)
np.save(testMasksPath + 'testmask',testmask)
