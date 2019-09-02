'''
Visualization script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 3rd April 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.autograd import Variable
import argparse
import cv2 as cv

import os


def plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory,img_file ,withBackground=False):
    '''
    Parameters
    ==========

    true_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes

    pred_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes

    nodesPresent : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step

    obs_length : Length of observed trajectory

    name : Name of the plot

    withBackground : Include background or not
    '''

    traj_length, numNodes, _ = true_trajs.shape
    # Initialize figure
    # Load the background
    # im = plt.imread('plot/background.png')
    # if withBackground:
    #    implot = plt.imshow(im)

    # width_true = im.shape[0]
    # height_true = im.shape[1]

    # if withBackground:
    #    width = width_true
    #    height = height_true
    # else:

    # width = 1
    # height = 1
    # width = 720
    # height = 576

    width = 640
    height = 480

    traj_data = {}
    nodes = []
    fig = plt.figure(1)
    im = cv.imread(img_file)

    plt.imshow(im)
    # plt.imread('')
    plt.hold(True)

    c = [0.81,0.63,0.66] # cyan for ground truth
    p = [0.66,0.19,0.81] # orangy for prediction

    # c = np.random.rand(3)
    # p = np.random.rand(3)
    for tstep in range(traj_length):
        pred_pos = pred_trajs[tstep, :]
        true_pos = true_trajs[tstep, :]

        for ped in range(numNodes):
            nodes.append(ped)
            if ped not in traj_data and tstep < obs_length:
                traj_data[ped] = [[], []]

            if ped in nodesPresent[tstep]: # and ped == 8:
                traj_data[ped][0].append(true_pos[ped, :])
                traj_data[ped][1].append(pred_pos[ped, :])

        for j in traj_data:
            if j  in [5,6,7,8,9,4]:
          # if j  in [1,3,4,5,7]:
          # if j % 2:
          #     c = np.random.rand(3)
          #     p = np.random.rand(3)
              true_traj_ped = traj_data[j][0]  # List of [x,y] elements
              pred_traj_ped = traj_data[j][1]
              # true_x = [(p[1])*width for p in true_traj_ped]
              # true_y = [(p[0])*height for p in true_traj_ped]
              # pred_x = [(p[1])*width for p in pred_traj_ped]
              # pred_y = [(p[0])*height for p in pred_traj_ped]

              true_y = [(p[0]+1)/2 * height for p in true_traj_ped]
              true_x = [(p[1]+1)/2 * width  for p in true_traj_ped]
              pred_y = [(p[0]+1)/2 * height for p in pred_traj_ped]
              pred_x = [(p[1]+1)/2 * width  for p in pred_traj_ped]
              if len(true_x) and len(true_y):
                    plt.plot(true_x, true_y, color=c, linewidth=0.5 , marker='*')
                    plt.plot(pred_x, pred_y, color=p, linestyle='dashed', linewidth=0.5, marker='+')
                    plt.annotate('p' + str(nodes[j]), xytext=(true_x[0], true_y[0]), xy=(true_x[0], true_y[0]), color='black' , fontsize=13)
                         # arrowprops={'arrowstyle':'-', 'color':'yellow', 'lw':2},
                     # fontsize=9)

# plt.annotate('id', xy=(true_x[0], true_y[1]), xytext=(true_x[0], true_y[1]), textcoords='id_' + str(j))
# if not withBackground:

# plt.ylim(1,0)
# plt.xlim(0,1)
# plt.ylim((480, 640))
# plt.xlim((640, 480))

    # plt.show()
    # if withBackground:
    # plt.savefig(name+'.png') #'plot_with_background/'+
    # else:
    plt.savefig(name+'.png')

    plt.gcf().clear()
    # plt.close('all')
    plt.clf()


def main():
    os.chdir('/home/serene/Documents/copy_srnn_pytorch/')
    # os.chdir('/home/serene/Documents/copy_srnn_pytorch_2/srnn-pytorch-master/obstacle_factor_DotPool/')
    parser = argparse.ArgumentParser()
    # os.chdir('/home/serene/Desktop/srnn-pytorch-master') #'/home/serene/Documents/copy_srnn_pytorch/'

    #/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/log/pixel_data/100e
    # Experiments
    parser.add_argument('--noedges', action='store_true')
    parser.add_argument('--temporal', action='store_false')
    parser.add_argument('--temporal_spatial', action='store_true')
    parser.add_argument('--attention', action='store_true' , default=True)

    parser.add_argument('--test_dataset', type=int, default=0,
                        help='test dataset index')

    # Parse the parameters
    args = parser.parse_args()

    # Check experiment tags
    if not (args.noedges or args.temporal or args.temporal_spatial or args.attention):
        print ('Use one of the experiment tags to enforce model')
        return

    # Save directory
    # save_directory = '/home/siri0005/srnn-pytorch-master/obstacle_factor/save' #'/home/serene/Documents/InVehicleCamera/save_kitti/' #'./save/pixel_data/100e/'
    # save_directory = '/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/save/pixel_data/100e/'
    # save_directory = './save/'
    save_directory = './fine_obstacle/prelu/p_02/save/'
    # save_directory = './save/'
    save_directory += str(args.test_dataset) + '/save_attention/'
    # plot_directory = '/home/siri0005/srnn-pytorch-master/plot/'
    # if args.noedges:
    #     print ('No edge RNNs used')
    #     save_directory += 'save_noedges'
    #     plot_directory += 'plot_noedges'
    # elif args.temporal:
    #     print ('Only temporal edge RNNs used')
    #     save_directory += 'save_temporal'
    #     plot_directory += 'plot_temporal'
    # elif args.temporal_spatial:
    #     print ('Both temporal and spatial edge RNNs used')
    #     save_directory += 'save_temporal_spatial'
    #     plot_directory += 'plot_temporal_spatial'
    # else:

    plot_directory = './fine_obstacle/prelu/p_02/plot/' + 'selected_plots/'  #str(args.test_dataset) + '/'
    # plot_directory = '/home/serene/Documents/copy_srnn_pytorch/fine_obstacle/prelu/p_02/plot/selected_plots'
    print ('Both temporal and spatial edge RNNs used with attention')
    # save_directory += 'save_attention'
    # plot_directory += 'plot_attention'

    f = open(save_directory+'results.pkl', 'rb')
    results = pickle.load(f)

    # print "Enter 0 (or) 1 for without/with background"
    # withBackground = int(input())
    withBackground = 1
    zfill = 3

    # for i in range(len(results)): #26,38,40,51,81,91
    for i in [82]:
        skip = str(int(results[i][5][0][19])).zfill(zfill)
        img_file = '/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/pedestrians/ewap_dataset/seq_eth/map.png'
        # '/home/serene/Documents/copy_srnn_pytorch/data/ucy/zara/zara.png'
        #'/home/serene/Documents/copy_srnn_pytorch/data/ucy/univ/ucy_univ.png'
        #'/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/pedestrians/ewap_dataset/seq_eth/map.png'
        #'/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/pedestrians/ewap_dataset/seq_hotel/map.png'
        #'/home/serene/Documents/video/hotel/frame-{0}.jpg'.format(skip)
        # for j in range(20):
        #     if i == 40:
        # name = plot_directory + str(args.test_dataset) +'/sequence_eth' + str(i)  #/pedestrian_1
        name = plot_directory + '/sequence_eth_' + str(i)
        for k in range(20):
            plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], name, plot_directory, img_file,withBackground)
        if int(skip) >= 999 and zfill < 4:
            zfill = zfill + 1
        elif int(skip) >= 9999 and zfill < 5:
            zfill = zfill + 1
        # print(skip)
        skip = str(int(skip) + 10).zfill(zfill)

if __name__ == '__main__':
    main()
