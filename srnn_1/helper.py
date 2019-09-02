
import numpy as np
import torch
import cv2
from copy import deepcopy as deepcopy
from os import path
from torch.autograd import Variable
import pickle as cpkl
from matplotlib import pyplot as plt
import math
def getVector(pos_list):
    '''
    Gets the vector pointing from second element to first element
    params:
    pos_list : A list of size two containing two (x, y) positions
    '''
    pos_i = pos_list[0]
    pos_j = pos_list[1]

    return np.array(pos_i) - np.array(pos_j)


def getMagnitudeAndDirection(*args):
    '''
    Gets the magnitude and direction of the vector corresponding to positions
    params:
    args: Can be a list of two positions or the two positions themselves (variable-length argument)
    '''
    if len(args) == 1:
        pos_list = args[0]
        pos_i = pos_list[0]
        pos_j = pos_list[1]

        vector = np.array(pos_i) - np.array(pos_j)
        magnitude = np.linalg.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector
        return [magnitude] + direction.tolist()

    elif len(args) == 2:
        pos_i = args[0]
        pos_j = args[1]

        ret = torch.zeros(3)
        vector = pos_i - pos_j
        magnitude = torch.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector

        ret[0] = magnitude
        ret[1:3] = direction
        return ret

    else:
        raise NotImplementedError('getMagnitudeAndDirection: Function signature incorrect')


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    # Exponential to get a positive value for std dev
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = torch.tanh(corr)

    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent):
    '''
    Returns samples from 2D Gaussian defined by the parameters
    params:
    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation
    nodesPresent : a list of nodeIDs present in the frame

    returns:
    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

def human_human_distance(ped_pos , neighbour_pos):
    return np.linalg.norm(np.subtract(ped_pos, neighbour_pos))

def human_obs_distance(ped_pos , obs_map , DS_ptr):
    min_dist = 1
    min_id = -1
    if 2 <= DS_ptr <= 3:
        DS_ptr = 2
    for i in range(len(obs_map[DS_ptr][0])):
        px = deepcopy(obs_map[DS_ptr][0][i])
        tmp = deepcopy(px)
        px[0] = tmp[1]
        px[1] = tmp[0]
        dist = np.linalg.norm(np.subtract(ped_pos, px))
        if min_dist > dist:
            min_dist = dist
            min_id = i
            # and ped_pos[1] < np.max(obs_positions[1]) and ped_pos[0] < np.max(obs_positions[0]) and ped_pos[1] < np.min(obs_positions[1]):
        else:
            continue

    return min_dist , min_id , px

def set_obstacle_zones():
    
    # obs_positions = np.where(mod_map == 0.5)
    # Calculate ped distance to every border pixel then choose the least distance to indicate the obstacle zone that has most of the impact on pedestrian path
    # check how close ped to an obstacle body by threshold-ing distances
    # for i in len(obs_positions):
    # np.linalg.norm(obs_positions[i][i] - )

    x_y_eth = [[-0.45, -0.42], [-0.52, 0.47]] #,[0.97, 0.05]] #,[0.98,-0.47],[ 0.92,0.66]
    ## 0.9624999999999999 0.028124999999999956 : entrance point
    # Important :: Feb 22nd: [0.98, 0.04] center point at entrance: defines target spot for most of the pedestrians (contextual point different from semantic point)
    # removed points :  [0.98, 0.09], [0.93, 0.66], [0.98, -0.48]
    ## Feb 22nd : remove half-way points at the side edges:[0.20, 0.58],[0.28, -0.45]
    ## Feb 23rd center point did not improve result , look at other possible ways to tackle potential target spot

    # 0.68	0.66	0.99	-0.40	-0.63	-0.41	-0.65	-0.62	-0.80	-1.00	-1.00
    # -0.63	-1.00	-0.68	-0.52	-0.49	-1.00	-1.00	-0.06	0.24	0.21	-0.54
    # [[0.27, 0.61], [0.78, 0.58], [0.37, 0.99], [0.70, 0.99]]
    # Feb 27th : restore old points version (4 points on side corners)+re-train
    # restore from 18-jan

    id_vec = range(len(x_y_eth))
    t = (x_y_eth, id_vec)
    raw_data = (t,)

    ## eth-hotel
    # x_y_hotel= [[-0.34,-0.16], [-0.37,0.42], [0.63,-0.24], [-0.33,-0.58], [-0.31,-0.99], [0.99, -0.61], [-0.66,0.59], [-0.66,-0.48], [0.99,0.63], [0.51,0.71], [0.41,0.22]]
    # x_y_hotel = [[0.12, 0.48], [0.35, 0.48], [0.56, 0.47], [0.76, 0.46], [0.53, 0.87], [0.51, 0.19]]
    #Feb 23rd : lowest FDE reached with this set of points = 1.34
    ## [[-0.20, -0.33], [-0.24, 0.63], [-0.61, 0.36], [0.81, -0.17],# removed  [-0.61, 0.99], [0.67, 0.99],[-0.61, -0.31], [0.74, 0.37]
                 # [-0.61, -0.66], [-0.20, 0.14], [-0.17, -0.67], [-0.24, 0.63]]

    x_y_hotel = [[-0.20, -0.33], [-0.24, 0.63], [-0.61, 0.36], [0.81, -0.17],# removed  [-0.61, 0.99], [0.67, 0.99],[-0.61, -0.31], [0.74, 0.37]
                 [-0.61, -0.66], [-0.20, 0.14], [-0.17, -0.67], [-0.24, 0.63]]
    id_vec = range(len(x_y_hotel))
    t = (x_y_hotel, id_vec)
    raw_data = raw_data + (t,)

    # x_y_zara = [[0.30, 0.21], [0.72, 0.26], [0.26, 0.91]]
    # -0.46	0.69	0.67	1.00	-0.41	-0.64	-0.65	-0.40	-1.00	-1.00	-0.82
    # 0.42	-0.63	-0.99	-0.68	-0.51	-0.49	-1.00	-1.00	-0.53	0.20	0.24
    # x_y_zara = [[-0.46,0.42], [0.69,-0.63], [0.67,-0.99], [1,-0.68], [-0.41,-0.51], [-0.64,-0.49], [-0.65,-1], [-0.40,-1], [-1,-0.53], [-1,0.20], [-0.82,0.24]]
    # x_y_zara = [[0.55, -0.01],	[0.33,0.01],	[0.23,-0.23],	[-1.00,-0.20],	[-1.00,-1.00],	[0.24,-0.93],
    #             [0.24,-0.92],	[0.54,-1.00],	[-0.82,0.63]]
    #Feb 23rd , lowest FDE(zara1) = 1.80 , FDE (zara2) = 2.00


    x_y_zara = [[0.45, 0.025], [-0.95, -0.18], [0.67, 0.66], [-0.40, 0.66]]
    # x_y_zara = [[0.02, 0.56], [0.03, 0.34], [-0.21, 0.23],[-0.18,-0.90],
    #              [0.66, - 0.81], [0.66, - 0.13], [0.66, 0.39], [0.66, 0.67], [0.66, 1.00]]

    # no need for origin point , [-0.98, - 0.99] , [-0.98, 0.55],[-0.90, 0.33],[-0.90, 0.25],[-0.18, - 0.79], [-0.18, - 0.99],[-0.33, - 0.89]]
    id_vec = range(len(x_y_zara))
    t = (x_y_zara, id_vec)
    raw_data = raw_data + (t,)

    # [-0.52, -0.78]: center point of green square on the left of image
    ## Feb 22nd neglect some points and modify locations
    x_y_univ = [ [-0.45,0.40],[0.83 , -0.83], [-0.85, -0.05] ]
    # remove contextual point [-0.52,-0.78] center point of plant square
    #removed : [-0.36,-0.56], [-0.62,-0.50] , [-0.4,-0.52], [-0.64,0.40] ,[-0.41, -1], [-0.65, -1],[0.66, -1],[0.99, -0.68],[-1, 0.21], [-0.62, -0.06],,[-0.63, -0.49]
    #New :  [-1,-0.54],[-0.62,-0.05], [-0.80,0.24],[0.70,-0.63]: center points at the shop & plant sqaures
    # place only center points same as old placements

    id_vec = range(len(x_y_univ))
    t = (x_y_univ, id_vec)
    raw_data = raw_data + (t,)

    # traj_obs = open('/home/siri0005/srnn-pytorch-master/srnn-pytorch-master/data/obstacles_loc.cpkl', 'wb')
    traj_obs = open('/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/data/obstacles_loc.cpkl', 'wb')
    cpkl.dump(raw_data, traj_obs)
    traj_obs.close()

def compute_edges(nodes, tstep, edgesPresent):
    '''
    Computes new edgeFeatures at test time
    params:
    nodes : A tensor of shape seq_length x numNodes x 2
    Contains the x, y positions of the nodes (might be incomplete for later time steps)
    tstep : The time-step at which we need to compute edges
    edgesPresent : A list of tuples
    Each tuple has the (nodeID_a, nodeID_b) pair that represents the edge
    (Will have both temporal and spatial edges)

    returns:
    edges : A tensor of shape numNodes x numNodes x 2
    Contains vectors representing the edges
    '''
    numNodes = nodes.size()[1]
    edges = (torch.zeros(numNodes * numNodes, 2)).cuda()
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes[tstep - 1, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[tstep, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent ,H ,  test_dataset):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''

    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length).cuda()
    counter = 0

    if test_dataset == 0:
        width = 640
        height = 480
    else:
        width = 720
        height = 576

    # counter = 0

    for tstep in range(pred_length):
        counter = 0
        for nodeID in assumedNodesPresent:
            if nodeID not in trueNodesPresent[tstep]:
                continue
            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]
            ''''''
            # transform pixel pos into meter

            pos = np.ones(3)
            posp = np.ones(3)
            u = true_pos[0]
            v = true_pos[1]

            # GT transformation
            # u = int(round((true_pos[0] +1) * width/2 ))
            # v = int(round((true_pos[1] +1) * height/2))

            u = int(true_pos[0] * width)
            v = int(true_pos[1] * height)

            if test_dataset == 0 or test_dataset == 1:
                pos[0] = v
                pos[1] = u
            else:
                pos[0] = u
                pos[1] = v

            true_pos_temp = torch.cuda.FloatTensor(np.dot(pos, H.transpose()))
            # true_pos_temp= pos
            xp = true_pos_temp[0] / true_pos_temp[2]
            yp = true_pos_temp[1] / true_pos_temp[2]
            true_pos_temp[0] = xp
            true_pos_temp[1] = yp

            # Prediction Transform
            up = pred_pos[0]
            vp = pred_pos[1]

            # up = int(round((pred_pos[0] + 1) * width / 2))
            # vp = int(round((pred_pos[1] + 1) * height / 2))

            up = int(pred_pos[0] * width)
            vp = int(pred_pos[1] * height)

            if test_dataset == 0 or test_dataset == 1:
                posp[0] = vp
                posp[1] = up
            else:
                posp[0] = up
                posp[1] = vp

            pred_pos_temp = torch.cuda.FloatTensor(np.dot(posp, H.transpose())) #posp
            xp = pred_pos_temp[0] / pred_pos_temp[2]
            yp = pred_pos_temp[1] / pred_pos_temp[2]
            pred_pos_temp[0] = xp
            pred_pos_temp[1] = yp

            x = (pred_pos_temp[0] - true_pos_temp[0])
            y = (pred_pos_temp[1] - true_pos_temp[1])

            # pred_pos - true_pos
            error[tstep] += torch.norm((pred_pos_temp - true_pos_temp),2)  # math.sqrt(torch.sum(torch.pow((pred_pos_temp-true_pos_temp),2)))#torch.norm(torch.FloatTensor(posp).cuda() - torch.FloatTensor(pos).cuda() , p=2) #
            # if tstep > 0:
            #     error[tstep] = error[tstep] / tstep
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter
            # error[tstep] += torch.norm((pred_pos_temp-true_pos_temp),2)#math.sqrt(torch.sum(torch.pow((pred_pos_temp-true_pos_temp),2)))#torch.norm(torch.FloatTensor(posp).cuda() - torch.FloatTensor(pos).cuda() , p=2) #
            # counter += 1
            #
            # if counter != 0:
            #     error[tstep] = error[tstep] / counter

    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent ,H ,  test_dataset):
    '''
    Computes final displacement error
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''

    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0
    if test_dataset == 0:
        width = 640
        height = 480
    else:
        width = 720
        height = 576

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent:
        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        ''''''
        # transform pixel pos into meter
        pos = np.ones(3)
        posp = np.ones(3)

        # GT transformation
        u = true_pos[0]
        v = true_pos[1]

        # u = int(round((true_pos[0] + 1) * width / 2))
        # v = int(round((true_pos[1] + 1) * height / 2))

        u = int(true_pos[0] * width)
        v = int(true_pos[1] * height)

        if test_dataset == 0 or test_dataset == 1:
            pos[0] = v
            pos[1] = u
        else:
            pos[0] = u
            pos[1] = v

        true_pos_temp = torch.cuda.FloatTensor(np.dot(pos, H.transpose())) #pos
        x = true_pos_temp[0] / true_pos_temp[2]
        y = true_pos_temp[1] / true_pos_temp[2]
        true_pos_temp[0] = x
        true_pos_temp[1] = y

        # Prediction Transform
        up = pred_pos[0]
        vp = pred_pos[1]

        # up = int(round((pred_pos[0] + 1) * width / 2))
        # vp = int(round((pred_pos[1] + 1) * height / 2))

        up = int(pred_pos[0] * width)
        vp = int(pred_pos[1] * height)

        if test_dataset == 0 or test_dataset == 1:
            posp[0] = vp
            posp[1] = up
        else:
            posp[0] = up
            posp[1] = vp

        pred_pos_temp = torch.cuda.FloatTensor(np.dot(posp, H.transpose())) #posp
        xp = pred_pos_temp[0] / pred_pos_temp[2]
        yp = pred_pos_temp[1] / pred_pos_temp[2]
        pred_pos_temp[0] = xp
        pred_pos_temp[1] = yp

        #pred_pos - true_pos

        error += torch.norm((pred_pos_temp - true_pos_temp), 2) #math.sqrt(torch.sum(torch.pow((pred_pos_temp-true_pos_temp),2))) #torch.norm(torch.FloatTensor(posp).cuda() - torch.FloatTensor(pos).cuda(), p=2) #
        counter += 1

    if counter != 0:
        error = error / counter #len(assumedNodesPresent)

    return error


def sample_gaussian_2d_batch(outputs, nodesPresent, edgesPresent, nodes_prev_tstep):
    mux, muy, sx, sy, corr = getCoef_train(outputs)

    next_x, next_y = sample_gaussian_2d_train(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent)

    nodes = torch.zeros(outputs.size()[0], 2)
    nodes[:, 0] = next_x
    nodes[:, 1] = next_y

    nodes = Variable(nodes.cuda())

    edges = compute_edges_train(nodes, edgesPresent, nodes_prev_tstep)

    return nodes, edges


def compute_edges_train(nodes, edgesPresent, nodes_prev_tstep):
    numNodes = nodes.size()[0]
    edges = Variable((torch.zeros(numNodes * numNodes, 2)).cuda())
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes_prev_tstep[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


def getCoef_train(outputs):
    mux, muy, sx, sy, corr = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d_train(mux, muy, sx, sy, corr, nodesPresent):
    o_mux, o_muy, o_sx, o_sy, o_corr = mux, muy, sx, sy, corr

    numNodes = mux.size()[0]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]

        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y





