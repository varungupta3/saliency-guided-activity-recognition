import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pdb
import pickle

labelsDataDir = '../../datasets/GTEA/labels_cleaned/'
# gazeDataPath = 'Ahmed_American.txt'
labels_files = glob.glob(labelsDataDir + '*.txt')
labels_dict = {}

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
                main_action = actions[0].split('<')[-1]
                frames = actions[1].split(' ')[-1].split('-')
                left_frame = int(frames[0].split('(')[-1])
                right_frame = int(frames[1].split(')')[0])
                frame_range = np.arange(left_frame, right_frame + 1)
                action2 = actions[1].split('>')[0]
                list_of_objects = action2.split(',')
                for frame in frame_range:
                    labels_dict[person_recipe][frame] = {}
                    labels_dict[person_recipe][frame]['main_action'] = main_action
                    labels_dict[person_recipe][frame]['other_actions'] = list_of_objects

            # else:
                # pdb.set_trace()

pdb.set_trace()