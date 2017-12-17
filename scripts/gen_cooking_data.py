import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pdb
from constants import *

recipeDataName = ['Ahmad_American.txt']
labels_files = [labelsDataDir+recipeName for recipeName in recipeDataName]
labels_dict = {}

action1_list = []
action2_list = []

for labels_file in labels_files:
    print(labels_file)
    # A dictionary for each video
    person_recipe = labels_file.split('/')[-1].split('.')[0]
    labels_dict[person_recipe] = {}

    with open(labels_file) as f:
        labels_data = f.read()
        labels_data = labels_data.split('\n')
        for count, label_info in enumerate(labels_data):
            print (count)
            # pdb.set_trace()
            actions = label_info.split('><')
            if len(actions) == 2: 
                action1 = actions[0].split('<')[-1]
                frames = actions[1].split(' ')[-1].split('-')
                left_frame = int(frames[0].split('(')[-1])
                right_frame = int(frames[1].split(')')[0])
                frame_range = np.arange(left_frame, right_frame + 1)
                action2 = actions[1].split('>')[0]
                list_of_action2 = action2.split(',')

                # Append to master list of actions
                if action1 not in action1_list:
                    action1_list.append(action1)
                for act2 in list_of_action2:
                    if act2 not in action2_list:
                        action2_list.append(act2)

                for frame in frame_range:
                    labels_dict[person_recipe][frame] = {}
                    labels_dict[person_recipe][frame]['action1'] = action1
                    labels_dict[person_recipe][frame]['action2'] = list_of_action2
                    labels_dict[person_recipe][frame]['action1_idx'] = [idx+1 for idx in range(len(action1_list)) if action1_list[idx]==action1]
                    labels_dict[person_recipe][frame]['action2_idx'] = [idx+1 for idx in range(len(action2_list)) if action2_list[idx]==list_of_action2[0]]


##########################################################################

gaze_files = [gazeDataDir+recipeName for recipeName in recipeDataName]
gaze_dict = {}

def adjust_gaze(x,y):
    return (x*INPUT_SIZE[1])/ORIG_SIZE[1], (y*INPUT_SIZE[0])/ORIG_SIZE[0]

videoNames = []

for gaze_file in gaze_files:
    # A dictionary for each video
    person_recipe = gaze_file.split('/')[-1].split('.')[0]
    if person_recipe not in videoNames:
        videoNames.append(person_recipe)

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

# videoNames = ['Ahmad_American']

noise_params = {'mu':0.7,'sigma':7,'size':[80,80]}

skip_frequency = 100
cooking_data = {}

gauss_t = np.linspace(-10, 10, 80)
gauss_bump = np.exp(-0.01*gauss_t**2)
gauss_bump /= np.trapz(gauss_bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = gauss_bump[:, np.newaxis] * gauss_bump[np.newaxis, :]
kernel_scaled = (kernel - np.min(kernel))/(np.max(kernel) - np.min(kernel))

numFrames = {}

cooking_frame_idxs = []
cooking_saliency_maps = []
cooking_images = []
cooking_action1 = []
cooking_action2 = []

for videoName in videoNames:
    img_files = glob.glob(imageDataDir + videoName + '/*.jpg')
    # pdb.set_trace()
    cooking_data[videoName] = {}
    # print img_files
    for frame_num, img_file in enumerate(img_files):
        # pdb.set_trace()
        if (frame_num+1) % skip_frequency != 0:
            continue

        cooking_data[videoName][frame_num+1] = {}

        img = cv2.imread(img_file)
        # Convert to RGB
        img = img[:,:,::-1]

        img_size = img.shape
        # saliency_img = np.zeros([img_size[0],img_size[1],img_size[2]], dtype=np.float64)
        saliency = np.zeros([img_size[0],img_size[1]], dtype=np.float64)
        if (frame_num+1) in gaze_dict[videoName]:
            print frame_num+1
            for num_gaze, gaze in enumerate(gaze_dict[videoName][frame_num+1]):
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
                    filter_size = [xmax-xmin, ymax-ymin]
                    saliency_map = kernel_scaled[y_center-dy_up:y_center+dy_down, x_center-dx_left:x_center+dx_right]
                    saliency[ymin:ymax,xmin:xmax] = saliency[ymin:ymax,xmin:xmax] + saliency_map
                    saliency = np.clip(saliency, 0.0, 1.0)

            # saliency_img = img * np.repeat(saliency[:,:,np.newaxis],3,axis=2) 

        cooking_frame_idxs.append(frame_num+1)
        cooking_saliency_maps.append(saliency)
        cooking_images.append(img)

        # cooking_data[videoName][frame_num+1]['saliency'] = saliency
        # cooking_data[videoName][frame_num+1]['image'] = img
        if (frame_num+1) in labels_dict[videoName].keys():
            cooking_action1.append(labels_dict[videoName][frame_num+1]['action1_idx'][0])
            cooking_action2.append(labels_dict[videoName][frame_num+1]['action2_idx'][0])
            # cooking_data[videoName][frame_num+1]['action1'] = labels_dict[videoName][frame_num+1]['action1_idx']
            # cooking_data[videoName][frame_num+1]['action2'] = labels_dict[videoName][frame_num+1]['action2_idx']
        else:
            cooking_action1.append(0)
            cooking_action2.append(0)
            # cooking_data[videoName][frame_num+1]['action1'] = 0
            # cooking_data[videoName][frame_num+1]['action2'] = 0

    numFrames[videoName] = len(img_files)

action1_list.insert(0, 'Nothing')
action2_list.insert(0, 'Nothing')
print 'Actions 1: ', action1_list
print 'Actions 2: ', action2_list

np.save(saveDataPath + 'images.npy', cooking_images)
print('Saved Images')
np.save(saveDataPath + 'saliency_maps.npy', cooking_saliency_maps)
print('Saved Saliency Maps')
np.save(saveDataPath + 'frame_idxs.npy', cooking_frame_idxs)
print('Saved Frame indexers')
np.save(saveDataPath + 'action1.npy', cooking_action1)
np.save(saveDataPath + 'action2.npy', cooking_action2)
print('Saved Actions')
np.save(saveDataPath + 'ordered_action1.npy', action1_list)
np.save(saveDataPath + 'ordered_action2.npy', action2_list)
print('Saved Action Indexers')

pdb.set_trace()
