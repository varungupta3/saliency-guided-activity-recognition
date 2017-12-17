import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pdb
import pickle

labelsDataDir = '../../datasets/GTEA/labels_cleaned/'
# gazeDataPath = 'Ahmed_American.txt'
labels_files = glob.glob(labelsDataDir + '*American.txt')
labels_dict = {}

main_actions_list = []
objects_master_list = []

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

                # Append to master list of actions
                if main_action not in main_actions_list:
                    main_actions_list.append(main_action)
                for objects in list_of_objects:
                    if objects not in objects_master_list:
                        objects_master_list.append(objects)

                for frame in frame_range:
                    labels_dict[person_recipe][frame] = {}
                    labels_dict[person_recipe][frame]['main_action'] = main_action
                    labels_dict[person_recipe][frame]['other_actions'] = list_of_objects
                    labels_dict[person_recipe][frame]['main_action_idx'] = [idx+1 for idx in range(len(main_actions_list)) if main_actions_list[idx]==main_action]
                    labels_dict[person_recipe][frame]['other_action_idx'] = [idx+1 for idx in range(len(objects_master_list)) if objects_master_list[idx]==list_of_objects[0]]

pdb.set_trace()