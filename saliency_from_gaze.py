import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pdb
from scipy.ndimage.filters import gaussian_filter

gazeDataDir = '../../datasets/GTEA/gaze/'
# gazeDataPath = 'Ahmed_American.txt'
gaze_files = glob.glob(gazeDataDir + '*.txt')

gaze_dict = {}

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
            x, y = float(gaze[3]), float(gaze[4])
            frame = int(gaze[5])
            gaze_type = gaze[7].split('\r')[0]
            if frame in gaze_dict[person_recipe]:
                gaze_dict[person_recipe][frame].append([x,y,gaze_type])
            else:
                gaze_dict[person_recipe][frame] = [[x,y,gaze_type]]

imageDataDir = '../../datasets/GTEA/'
videoNames = ['Ahmad_American']

noise_params = {'mu':0.6,'sigma':0.3,'size':[100,100]}

skip_frequency = 5

for videoName in videoNames:
    img_files = glob.glob(imageDataDir + videoName + '/*.jpg')
    saliencies[videoName] = {}
    for frame_num, img_file in enumerate(img_files):
        if frame_num % skip_frequency != 0:
            continue

        img = cv2.imread(img_file)

        img_size = img.shape
        saliency = np.zeros([img_size[0],img_size[1]], dtype=np.float64)
        if frame_num+1 in gaze_dict[videoName]:
            for gaze in gaze_dict[videoName][frame_num+1]:
                x = gaze[0] # Corresponds to column
                y = gaze[1] # Corresponds to row
                gaze_type = gaze[2]
                xmin = max([0, int(x)-int(noise_params['size'][1]/2)])
                xmax = min([img_size[1],int(x)+int(noise_params['size'][1]/2)])
                ymin = max([0, int(y)-int(noise_params['size'][0]/2)])
                ymax = min([img_size[0],int(y)+int(noise_params['size'][0]/2)])

                # To remove gaze points outside image frame
                if ((xmax-xmin > 0) and (ymax-ymin > 0)):
                    print (xmin, xmax, ymin, ymax)
                    filter_size = [xmax-xmin, ymax-ymin]
                    print 'Filter size : ' , filter_size
                    sal_noise = gaussian_filter(np.random.randn(filter_size[1],filter_size[0]), 
                        noise_params['sigma'])
                    sal_noise = (sal_noise - np.min(sal_noise))/(np.max(sal_noise) - np.min(sal_noise))
                    print 'Saliency    ; ' , sal_noise.shape
                    # pdb.set_trace()
                    saliency[ymin:ymax,xmin:xmax] = sal_noise

        saliencies[videoName][frame_num+1] = saliency
        # pdb.set_trace()
        
    pdb.set_trace()