'''
Author: He Wei
'''

import pickle
import argparse

import os
from utils import DataLoader

import numpy as np

import cv2
from PIL import Image, ImageDraw

green = (0,255,0)
red = (0,0,255)


parser = argparse.ArgumentParser()
# Observed length of the trajectory parameter
parser.add_argument('--obs_length', type=int, default=8,
                    help='Observed length of the trajectory')
# Predicted length of the trajectory parameter
parser.add_argument('--pred_length', type=int, default=12,
                    help='Predicted length of the trajectory')
# Test dataset
parser.add_argument('--visual_dataset', type=int, default=1,
                    help='Dataset to be tested on')

# Model to be loaded
parser.add_argument('--epoch', type=int, default=39,
                    help='Epoch of model to be loaded')

# Parse the parameters
sample_args = parser.parse_args()

'''KITTI Training Setting'''

save_directory = '/home/serene/Documents/KITTIData/GT/save/pixel_data/1'

with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

f = open('/home/serene/Documents/KITTIData/GT/save/pixel_data/1/results.pkl', 'rb')
results = pickle.load(f)

dataset = [sample_args.visual_dataset]
data_loader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, dataset, True, infer=True)
#print(data_loader.data[0][0].shape)


# for j in range(len(data_loader.data[0])):
#
#     sourceFileName = "/home/hesl/PycharmProjects/social-lstm-tf-HW/data/KITTI-17/img1/" + str(j + 1).zfill(6) + ".jpg"
#
#     avatar= cv2.imread(sourceFileName)
#     #drawAvatar= ImageDraw.Draw(avatar)
#     #print(avatar.shape)
#     xSize  = avatar.shape[1]
#     ySize = avatar.shape[0]
#     #print(data_loader.data[0][0][0])
#     for i in range(20):
#          #print(i)
#          y=int(data_loader.data[0][j][i][1]*ySize)
#          x=int(data_loader.data[0][j][i][2]*xSize)
#          cv2.rectangle(avatar, (x  - 2, y  - 2), (x  + 2, y + 2), green,thickness=-1)
#          #drawAvatar.rectangle([(x  - 2, y  - 2), (x  + 2, y + 2)], fill=(255, 100, 0))
#
#     #drawAvatar.rectangle([(466, 139), (91 + 466, 139 + 193.68)])
#     #avatar.show()
#     cv2.imshow("avatar", avatar)
#     cv2.waitKey(0)
print(results[0][1][0][2])

#Each Frame
for k in range(int(len(data_loader.data[0])/(sample_args.obs_length+sample_args.pred_length))):
    #Each
    for j in range(sample_args.obs_length+sample_args.pred_length):

        sourceFileName = "/home/serene/Desktop/papers_docs/Video_collections/KITTI/2DMOT2015/train/KITTI-13/img1/" + str(j + 1+k*(sample_args.obs_length+sample_args.pred_length)).zfill(6) + ".jpg"

        avatar= cv2.imread(sourceFileName)

        xSize  = avatar.shape[1]
        ySize = avatar.shape[0]
        print(sourceFileName)

        for i in range(20):

            #GT
            x=int(results[k][0][j][i][0][1]*xSize)
            y=int(results[k][0][j][i][0][0]*ySize)
            cv2.rectangle(avatar, (x  - 2, y  - 2), (x  + 2, y + 2), green,thickness=-1)

            #Predicted
            xp = int(results[k][1][j][i][1][1] * xSize)
            yp = int(results[k][1][j][i][1][0] * ySize)
            cv2.rectangle(avatar, (xp - 2, yp - 2), (xp + 2, yp + 2), red, thickness=-1)

        cv2.imshow("avatar", avatar)
        cv2.waitKey(0)





print(len(results))
