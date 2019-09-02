

# Extension from Social Attention 2018 source code for (Situation-Aware pedestrian trajectory prediction CVWW, Feb 2019)

# @Sirin Haddad - Last update : 2nd Sept 2019
# bugs in original code were fixed for calculating ADE & FDE errors
# resulting in lower gap between ADE and FDE (more consistent results along both error metrics)

import argparse
import os
import pickle
import time
import torch
import numpy as np
from torch.autograd import Variable

from utils import DataLoader
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood
def main():

    for i in [2,3]:
        parser = argparse.ArgumentParser()
        # RNN size
        parser.add_argument('--human_node_rnn_size', type=int, default=256,
                            help='Size of Human Node RNN hidden state')
        parser.add_argument('--obs_node_rnn_size', type=int, default=256,
                            help='Size of Human Node RNN hidden state')
        parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                            help='Size of Human Human Edge RNN hidden state')
        parser.add_argument('--human_obstacle_edge_rnn_size', type=int, default=256,
                            help='Size of Obstacle Node RNN hidden state')

        # Input and output size
        parser.add_argument('--human_node_input_size', type=int, default=2,
                            help='Dimension of the node features')
        parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                            help='Dimension of the edge features')
        parser.add_argument('--human_obstacle_edge_input_size', type=int, default=2,
                            help='Dimension of the node input')

        parser.add_argument('--human_node_output_size', type=int, default=5,
                            help='Dimension of the node output')

        # Embedding size
        parser.add_argument('--human_node_embedding_size', type=int, default=64,
                            help='Embedding size of node features')
        parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                            help='Embedding size of edge features')
        parser.add_argument('--human_obstacle_edge_embedding_size', type=int, default=64,
                            help='Embedding size of edge features')

        # Attention vector dimension
        parser.add_argument('--attention_size', type=int, default=64,
                            help='Attention size')

        # Sequence length
        parser.add_argument('--seq_length', type=int, default=20,
                            help='Sequence length')
        parser.add_argument('--pred_length', type=int, default=12,
                            help='Predicted sequence length')
        # let batch size be bigger for faster training, prev: 8
        # Batch size
        parser.add_argument('--batch_size', type=int, default=24,
                            help='Batch size')

        # Number of epochs
        parser.add_argument('--num_epochs', type=int, default=100,
                            help='number of epochs')

        # Gradient value at which it should be clipped
        parser.add_argument('--grad_clip', type=float, default=10.,
                            help='clip gradients at this value')
        # Lambda regularization parameter (L2)
        parser.add_argument('--lambda_param', type=float, default=0.00005,
                            help='L2 regularization parameter')

        # Learning rate parameter
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='learning rate')
        # Decay rate for the learning rate parameter
        parser.add_argument('--decay_rate', type=float, default=0.99,
                            help='decay rate for the optimizer')

        # Dropout rate
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability')

        # The leave out dataset
        parser.add_argument('--leaveDataset', type=int, default=i,
                            help='The dataset index to be left out in training')

        parser.add_argument('--distance_thresh' ,type=float , default=0.5, # small thresholds will not enable the graph to catch lengthy trajectories(only when pedestrians are really close to each other)
                            help='The max distance allowable to consider an interaction between human and obstacle')
        args = parser.parse_args()

        train(args)

def train(args):
    ## 19th Feb : move training files for 3 && 4 to fine_obstacle under copy_2

    # os.chdir('/home/serene/PycharmProjects/srnn-pytorch-master/')
    # context_factor is an experiment for including potential destinations of pedestrians in the graph.
    # did not yield good improvement
    os.chdir('/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master')
    # os.chdir('/home/siri0005/srnn-pytorch-master')
    # base_dir = '../fine_obstacle/prelu/p_02/'

    base_dir =  '../fine_obstacle/prelu/p_02/'
        # '../MultiNodeAttn_HH/'
    # os.chdir('/home/serene/Documents/KITTIData/GT')

    datasets = [i for i in [0,1,2,3,4]]
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)
    # datasets = [0]
    # args.leaveDataset = 0

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, forcePreProcess=True)

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, args.seq_length + 1)

    # Log directory
    # log_directory = './log/world_data/normalized_01/'
    log_directory = base_dir+'log/'
    log_directory += str(args.leaveDataset) + '/'
    log_directory += 'log_attention/'

    # Logging file
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    # save_directory = './save/world_data/normalized_01/'
    save_directory = base_dir+'save/'
    save_directory += str(args.leaveDataset)+'/'
    save_directory += 'save_attention/'

    # log RELU parameter
    param_log_dir = save_directory + 'param_log.txt'
    param_log = open(param_log_dir , 'w')

    # Open the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    # srnn_model = SRNN(args)

    # net = torch.nn.DataParallel(srnn_model)

    # CUDA_VISIBLE_DEVICES = 1
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)

    # learning_rate = args.learning_rate
    print('Training begin')
    best_val_loss = 100
    best_epoch = 0
    # start_epoch = 0

    # if args.leaveDataset == 3:
    # start_epoch =checkpoint_path(0)

    # ckp = torch.load(checkpoint_path(3))
    # net.load_state_dict(ckp['state_dict'])
    # optimizer.load_state_dict(ckp['optimizer_state_dict'])

    # ckp = torch.load(checkpoint_path(4))
    # net.load_state_dict(ckp['state_dict'])
    # optimizer.load_state_dict(ckp['optimizer_state_dict'])
    # last_epoch = ckp['epoch']
    # rang = range(last_epoch, args.num_epochs, 1)
    rang = range(args.num_epochs)

    # Training
    for epoch in rang:
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0
        # if args.leaveDataset == 2 and epoch == 0:
        #     dataloader.num_batches = dataloader.num_batches + start_epoch
        #     epoch += start_epoch

        # For each batch

        stateful = True  # flag that controls transmission of previous hidden states to current hidden states vectors.
        # Part of statefulness

        for batch in range( dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, _, _, d = dataloader.next_batch(randomUpdate=True) ## shuffling input + stateless lstm

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]],d, args.distance_thresh)

                nodes, edges, nodesPresent, edgesPresent,obsNodes, obsEdges, obsNodesPresent, obsEdgesPresent = stgraph.getSequence() #

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                obsNodes = Variable(torch.from_numpy(obsNodes).float()).cuda()
                obsEdges = Variable(torch.from_numpy(obsEdges).float()).cuda()

                ## Modification : reset hidden and cell states only once after every batch ; keeping states updated during sequences 31st JAN

                # Define hidden states
                numNodes = nodes.size()[1]
                numObsNodes = obsNodes.size()[1]

                # if not stateful:
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()

                ## new update : 25th JAN , let the hidden state transition begin with negative ones
                ## such initialization did not lead to any new learning results, network converged quickly and loss did not decrease below
                # -6
                hidden_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
                hidden_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()

                cell_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
                cell_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                #  _ = \
                outputs, h_node_rnn, h_edge_rnn, cell_node_rnn, cell_edge_rnn,o_h_node_rnn, o_h_edge_rnn, o_cell_node_rnn, o_cell_edge_rnn, _ = \
                    net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                    hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs,
                    obsNodes[:args.seq_length], obsEdges[:args.seq_length], obsNodesPresent[:-1],
                    obsEdgesPresent[:-1]
                        ,hidden_states_obs_node_RNNs, hidden_states_obs_edge_RNNs, cell_states_obs_node_RNNs,
                    cell_states_obs_edge_RNNs)

                # # else:
                # #     if len(nodes) == len(hidden_states_node_RNNs): # no additional nodes introduced in graph
                # #         hidden_states_node_RNNs = Variable(h_node_rnn).cuda()
                # #         cell_states_node_RNNs = Variable(cell_node_rnn).cuda()
                # #     else: # for now number of nodes is only increasing in time as new pedestrians are detected in the scene
                # #         pad_size = len(nodes) - len(hidden_states_node_RNNs)
                #         cell_states_node_RNNs = Variable(np.pad(cell_node_rnn, pad_size)).cuda()
                # if sequence > 0:
                #     hidden_states_node_RNNs = Variable(h_node_rnn).resize(hidden_states_node_RNNs.cpu().size())
                #     hidden_states_edge_RNNs = h_edge_rnn
                #     cell_states_node_RNNs = cell_node_rnn
                #     cell_states_edge_RNNs = cell_edge_rnn
                    # new_num_nodes = h_node_rnn.size()[0] - hidden_states_node_RNNs.size()[0]
                    # if h_node_rnn.size()[0] - hidden_states_node_RNNs.size()[0] >=1:
                    #     np.pad(hidden_states_node_RNNs.cpu() , new_num_nodes, mode='constant').cuda()

                    # hidden_states_obs_node_RNNs = o_h_node_rnn
                    # hidden_states_obs_edge_RNNs = o_h_edge_rnn
                    # cell_states_obs_node_RNNs = o_cell_node_rnn
                    # cell_states_obs_edge_RNNs = o_cell_edge_rnn

                # Zero out the gradients

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.data[0]
                # Compute gradients
                loss.backward(retain_variables=True)
                param_log.write(str(net.alpha.data[0]) + '\n')

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches
        # Log it
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0
        # dataloader.valid_num_batches = dataloader.valid_num_batches + start_epoch
        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)## stateless lstm without shuffling

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                stgraph.readGraph([x[sequence]], d, args.distance_thresh)

                nodes, edges, nodesPresent, edgesPresent,obsNodes, obsEdges, obsNodesPresent, obsEdgesPresent = stgraph.getSequence() #

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                obsNodes = Variable(torch.from_numpy(obsNodes).float()).cuda()
                obsEdges = Variable(torch.from_numpy(obsEdges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()

                numObsNodes = obsNodes.size()[1]
                hidden_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
                hidden_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()

                cell_states_obs_node_RNNs = Variable(torch.zeros(numObsNodes, args.obs_node_rnn_size)).cuda()
                cell_states_obs_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_obstacle_edge_rnn_size)).cuda()
                #

                outputs,  h_node_rnn, h_edge_rnn, cell_node_rnn, cell_edge_rnn,o_h_node_rnn ,o_h_edge_rnn, o_cell_node_rnn, o_cell_edge_rnn,  _= net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1],
                                             edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs
                                             , obsNodes[:args.seq_length], obsEdges[:args.seq_length],
                                             obsNodesPresent[:-1], obsEdgesPresent[:-1]
                                             ,hidden_states_obs_node_RNNs, hidden_states_obs_edge_RNNs,
                                             cell_states_obs_node_RNNs, cell_states_obs_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)

                loss_batch += loss.data[0]

                # Reset the stgraph
                stgraph.reset()

            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        # Record best epoch and best validation loss
        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
        # Log it
        log_file_curve.write(str(loss_epoch) + '\n')
        # Save the model after each epoch

        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    # Record the best epoch and best validation loss overall
    print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
    # Log it
    log_file.write(str(best_epoch) + ',' + str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()
    param_log.close()

if __name__ == '__main__':
    main()
