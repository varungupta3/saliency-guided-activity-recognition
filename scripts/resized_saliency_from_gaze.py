import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pdb
from constants import *

# gazeDataPath = 'Ahmed_American.txt'
gaze_files = glob.glob(gazeDataDir + '*.txt')
gaze_dict = {}

def adjust_gaze(x,y):
    return (x*INPUT_SIZE[1])/ORIG_SIZE[1], (y*INPUT_SIZE[0])/ORIG_SIZE[0]


for gaze_file in gaze_files:
    # A dictionary for each video
    person_recipe = gaze_file.split('/')[-1].split('.')[0]
    gaze_dict[person_recipe] = {}

    with open(gaze_file) as f:
        gaze_data = f.read()
        gaze_data = gaze_data.split('\n')
        gaze_data = gaze_data[34:-1]
        for gaze_info in gaze_data:
            gaze = gaze_info.split('\t')
            x, y = adjust_gaze(float(gaze[3]), float(gaze[4]))
            frame = int(gaze[5])
            gaze_type = gaze[7].split('\r')[0]
            if frame in gaze_dict[person_recipe]:
                gaze_dict[person_recipe][frame].append([x,y,gaze_type])
            else:
                gaze_dict[person_recipe][frame] = [[x,y,gaze_type]]

videoNames = ['Ahmad_American']

trainimages = []    
trainmask = []
trainsaliencyimages = []

noise_params = {'mu':0.7,'sigma':7,'size':[80,80]}

skip_frequency = 6
saliencies = {}

gauss_t = np.linspace(-10, 10, 80)
gauss_bump = np.exp(-0.01*gauss_t**2)
gauss_bump /= np.trapz(gauss_bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = gauss_bump[:, np.newaxis] * gauss_bump[np.newaxis, :]
kernel_scaled = (kernel - np.min(kernel))/(np.max(kernel) - np.min(kernel))

for videoName in videoNames:
    img_files = glob.glob(imageDataDir + videoName + '/*.jpg')
    saliencies[videoName] = {}
    for frame_num, img_file in enumerate(img_files):
        if (frame_num+1) % skip_frequency != 0:
            # print ('Skip check')
            continue

        img = cv2.imread(img_file)

        # Convert to RGB for plt.imshow
        img = img[:,:,::-1]

        img_size = img.shape
        saliency_img = np.zeros([img_size[0],img_size[1],img_size[2]], dtype=np.float64)
        saliency = np.zeros([img_size[0],img_size[1]], dtype=np.float64)
        if frame_num+1 in gaze_dict[videoName]:
            print frame_num+1
            for gaze in gaze_dict[videoName][frame_num+1]:
                x = gaze[0] # Corresponds to column
                y = gaze[1] # Corresponds to row
                gaze_type = gaze[2]
                xmin = max([0, int(x)-int(noise_params['size'][1]/2)])
                xmax = min([img_size[1],int(x)+int(noise_params['size'][1]/2)])
                ymin = max([0, int(y)-int(noise_params['size'][0]/2)])
                ymax = min([img_size[0],int(y)+int(noise_params['size'][0]/2)])

                # Relative coordinates for extracting saliency for edge cases
                dx_left = int(x) - xmin
                dx_right = xmax - int(x)
                dy_up = int(y) - ymin
                dy_down = ymax - int(y)
                y_center = int(noise_params['size'][0]/2)
                x_center = int(noise_params['size'][1]/2)

                # To remove gaze points outside image frame
                if ((xmax-xmin > 0) and (ymax-ymin > 0)):
                    # print (xmin, xmax, ymin, ymax)
                    filter_size = [xmax-xmin, ymax-ymin]
                    # print 'Filter size : ' , filter_size
                    saliency_map = kernel_scaled[y_center-dy_up:y_center+dy_down, x_center-dx_left:x_center+dx_right]
                    # pdb.set_trace()
                    sal_map_im = img[ymin:ymax, xmin:xmax,:] * np.repeat(saliency_map[:,:,np.newaxis],3,axis=2)
                    saliency[ymin:ymax,xmin:xmax] = saliency_map
                    saliency_img[ymin:ymax,xmin:xmax,:] = sal_map_im

            # resized_img = cv2.resize(img, (256,192))
            # resized_saliency_img = cv2.resize(saliency_img, (256,192))
            # resized_saliency = cv2.resize(saliency, (256,192))

            trainimages.append(img)
            trainmask.append(saliency)
            trainsaliencyimages.append(saliency_img) 
            # pdb.set_trace()   

        saliencies[videoName][frame_num+1] = saliency
        # pdb.set_trace()
    # pdb.set_trace()
pdb.set_trace()
np.save(trainImagesPath,trainimages)
np.save(trainMasksPath,trainmask)
np.save(trainSaliencyImagesPath,trainsaliencyimages)
# pdb.set_trace()
# The returned saliencies is a dictionary.
# saliencies['John Doe'][f] gives the saliency map for video 'John Doe' and frame number f which starts from 1.
# Some videos do not have saliency maps for all frames since fixations are not synchronized well with the image frames
