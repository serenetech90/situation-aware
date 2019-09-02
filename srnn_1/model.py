'''
Author : Anirudh Vemula
Date : 16th March 2017
'''

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import os

class HumanNodeRNN(nn.Module):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanNodeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size*2, self.embedding_size)

        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_temporal, h_spatial_other, h, c):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # Concat both the embeddings
        h_edges = torch.cat((h_temporal, h_spatial_other), 1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        h_edges_embedded = self.dropout(h_edges_embedded)

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), 1)

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)

        return out, h_new, c_new

class HumanObstacleEdgeRNN(nn.Module):
    def __init__(self , args , infer = False):
        super(HumanObstacleEdgeRNN , self).__init__()
        self.args = args
        self.infer = infer

        # define required rnn shapes
        self.rnn_size = args.human_obstacle_edge_rnn_size
        self.embedding_size = args.human_obstacle_edge_embedding_size
        self.input_size = args.human_obstacle_edge_input_size

        # Linear layer to embed input -- For now ...
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        #self.encoder_linear.weight =
        #self.encoder_linear.weight = nn.Parameter(torch.normal(means=torch.arange(-1,0), std=torch.arange(-1,0,0.1)))

        # ReLU and Dropout layers
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)
        ## initialize input to hidden weight connection to random negative values
        #self.cell.weight_ih = self.cell.weight_ih(torch.nn.Parameter(torch.normal(means=torch.arange(-1,0), std=torch.arange(-1,0))))

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new

class HumanHumanEdgeRNN(nn.Module):
    '''
    Class representing the Human-Human Edge RNN in the s-t graph
    '''
    def __init__(self, args, infer=False):
        '''function
        Initializer
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanHumanEdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_human_edge_embedding_size
        self.input_size = args.human_human_edge_input_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class EdgeAttention(nn.Module):
    '''
    Class representing the attention module
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.Linear(1,self.human_human_edge_rnn_size)

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer = nn.Linear(1 , self.human_human_edge_rnn_size)
        self.prelu = torch.nn.PReLU(init=args.alpha).cuda()

    def forward(self,h_temporal, h_spatials):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        # Number of spatial edges
        num_edges = h_spatials.size()[0]
        # Newly introduced this layer in attempt to propose a mechanism that combines the impact of multiple nodes on a single node
        # similar to multiple-Head attention (GAT, 2018) mechanism, we use leaky relu with negative slope (0.2) as we are mapping hidden states of normalized
        # input range [-1,+1], we only use two heads for every node , which means we have two single layers of attention for spatial and temporal edges
        # respectively, different from GAT is that we find it necessary to use both combination types concatenation then averaging the head output to keep
        # compact dimensionality and as we are using a jointly training procedure of two networks (temporal and spatial) we cannot assume independence of
        # temporal from spatial features
        # much like single head dot-product attention , the averaging across two feature spaces has same effect as dot-product normalized by a scaling factor
        logsoft = torch.nn.LogSoftmax().cuda()
        # h_spatials = h_spatials.cpu()
        # h_temporal = h_temporal.cpu()

        # c_spatials = h_spatials.data.cpu()
        # c_temporal = h_temporal.data

        # Embed the temporal edgeRNN hidden state
        # temporal_embed = self.temporal_edge_layer(h_temporal)
        # temporal_embed = temporal_embed.squeeze(0)

        # # Embed the spatial edgeRNN hidden states
        # spatial_embed = self.spatial_edge_layer(h_spatials)

        # # Prev: Dot based attention
        # NEW: concatenate the spatial and temporal edges hidden states related to one node
        # all_edges = self.spatial_edge_layer(all_edges)
        # all_edges = all_edges.squeeze(0)
        # spatial attention and temporal attention layers
        spatial_attn_vec = self.prelu(h_spatials)
        temporal_attn_vec = self.prelu(h_temporal)
        # Variable length
        # self.project2out = nn.Linear(self.human_human_edge_rnn_size,num_edges)
        # temperature = num_edges / np.sqrt(self.attention_size)
        # attn = torch.mul(attn_coef, temperature)
        # print (attn)
        # Softmax
        # @serene : RELU
        # a_~: attention mechanism
        # applied logsoft for normalization to sum to 1
        a_spatial = logsoft(spatial_attn_vec)
        a_temporal = logsoft(temporal_attn_vec)

        # Compute weighted value
        spatial_weighted_value = torch.mul(a_spatial, h_spatials)
        temporal_weighted_value = torch.mul(a_temporal, h_temporal)

        cat_weighted_value = torch.cat((spatial_weighted_value, temporal_weighted_value))

        # compare multiplicative attention with averaging to decide which is more suitable to capture interactions
        joint_weighted_value = torch.mean(cat_weighted_value, 0)

        # self.alpha = self.prelu.weight.data[0]
        # We applied Multi-Head attention concept in our own case (joint edges temporal and spatial):
        # relu-based transformation of received edges , passed through logsoftmax non-linearity, multiplied the resulted vectors with hidden states
        # concatenated both weighted vectors (temporal || spatial) then horizontally averaged (combined) them into 1d vector containing weights obtained
        # from jointly passing the vectors of multiple attention mechanisms applied on neighbors of a given node to indicate a neighboring node importance

        return joint_weighted_value , a_spatial # weighted_value


class SRNN(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SRNN, self).__init__()

        self.args = args
        self.infer = infer

        if self.infer:
            # Test time
            self.seq_length = 1
            self.obs_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length
            self.obs_length = args.seq_length - args.pred_length

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        # self.human_obstacle_edge_rnn_size = args.human_obstacle_edge_rnn_size ## new
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = HumanNodeRNN(args, infer)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(args, infer)
        # self.humanObstacleEdgeRNN_Spatial = HumanObstacleEdgeRNN(args, infer)

        const = 0.2
        self.alpha = Variable(torch.zeros(1)+const , requires_grad=True)

        args.alpha = self.alpha.data[0]
        # Initialize attention module
        self.attn = EdgeAttention(args, infer)

    def forward(self, nodes, edges, nodesPresent, edgesPresent, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs):
                # ,obsNodes, obsEdges , obsNodesPresent, obsEdgesPresent , hidden_states_obs_node_RNNs,
                # hidden_states_obs_edge_RNNs, cell_states_obs_node_RNNs, cell_states_obs_edge_RNNs):
        '''
        Forward pass for the model
        params:
        nodes : input node features
        edges : input edge features
        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame
        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame
        hidden_states_node_RNNs : A tensor of size numNodes x node_rnn_size
        Contains hidden states of the node RNNs
        hidden_states_edge_RNNs : A tensor of size numNodes x numNodes x edge_rnn_size
        Contains hidden states of the edge RNNs

        returns:
        outputs : A tensor of shape seq_length x numNodes x 5
        Contains the predictions for next time-step
        hidden_states_node_RNNs
        hidden_states_edge_RNNs
        '''
        # Get number of nodes
        numNodes = nodes.size()[1] #+ obsNodes.size()[1]

        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length*numNodes, self.output_size)).cuda()

        # Data structure to store attention weights
        attn_weights = [{} for _ in range(self.seq_length)]

        # For each frame
        for framenum in range(self.seq_length):
            # Find the edges present in the current frame
            edgeIDs = edgesPresent[framenum]
            #[(edgesPresent[framenum]) , (obsEdgesPresent[framenum])]

            # Separate temporal and spatial edges
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            spatial_edges = [x for x in edgeIDs if x[0] != x[1]]

            # Find the nodes present in the current frame
            nodeIDs = nodesPresent[framenum]
            # nodeIDs = tuple(map(tuple , np.reshape(np.append(np.array(nodesPresent[framenum]), np.array(obsNodesPresent[framenum])),
            #            [len(nodesPresent[framenum]) + len(obsNodesPresent[framenum]), 1])))

            # Get features of the nodes and edges present
            nodes_current = nodes[framenum]
            edges_current = (edges[framenum]).cuda()
            # edges_current = edges_current.cuda()


            # obs_nodes_current = obsNodes[framenum]
            # obs_edges_current = obsEdges[framenum]

            # Initialize temporary tensors
            hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())
            hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())

            # hidden_states_obs_edges_spatial = Variable(torch.zeros(numObsNodes , self.human_obstacle_edge_rnn_size).cuda())

            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

            # If there are any edges
            # obsedgeIDs = obsEdgesPresent[framenum]
            obs_found = False #len(obsedgeIDs) > 0

            if len(edgeIDs) != 0:
                # Temporal Edges
                if len(temporal_edges) != 0:
                    # Get the temporal edges
                    list_of_temporal_edges = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in edgeIDs if x[0] == x[1]]).cuda())
                    # Get nodes associated with the temporal edges
                    list_of_temporal_nodes = torch.LongTensor([x[0] for x in edgeIDs if x[0] == x[1]]).cuda()

                    # Get the corresponding edge features
                    edges_temporal_start_end = torch.index_select(edges_current, 0, list_of_temporal_edges)
                    # Get the corresponding hidden states
                    hidden_temporal_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges)
                    # Get the corresponding cell states
                    cell_temporal_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges)

                    # Do forward pass through temporaledgeRNN
                    h_temporal, c_temporal = self.humanhumanEdgeRNN_temporal(edges_temporal_start_end, hidden_temporal_start_end, cell_temporal_start_end)

                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[list_of_temporal_edges.data] = h_temporal
                    cell_states_edge_RNNs[list_of_temporal_edges.data] = c_temporal

                    # Store the temporal hidden states obtained in the temporary tensor
                    hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes] = h_temporal

                # Spatial Edges
                if len(spatial_edges) != 0:
                    # Get the spatial edges
                    list_of_spatial_edges = Variable(torch.LongTensor([(x[0])*numNodes + (x[1]) for x in edgeIDs if x[0] != x[1]]).cuda())

                    # # Get the set of obstacles spatial edges
                    # For now let the obstacle spatial edges tensors be of same size as other spatial edges
                    # later only count for the num of obstacle nodes
                    # list_of_obstacle_spatial_edges = Variable(torch.LongTensor([x[0]*numNodes + x[1] for x in edgeIDs if x[0] != x[1]]).cuda())

                    # Get nodes associated with the spatial edges
                    list_of_spatial_nodes = np.array([x[0] for x in edgeIDs if x[0] != x[1]])

                    # Get the corresponding edge features
                    edges_spatial_start_end = torch.index_select(edges_current, 0,list_of_spatial_edges)
                    # Get the corresponding hidden states
                    hidden_spatial_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_spatial_edges)
                    # Get the corresponding cell states
                    cell_spatial_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_spatial_edges)

                    # o_h_spatial = torch.LongTensor()
                    # o_c_spatial = torch.LongTensor()

                    # Do forward pass through spatialedgeRNN
                    h_spatial, c_spatial = self.humanhumanEdgeRNN_spatial(edges_spatial_start_end, hidden_spatial_start_end, cell_spatial_start_end)
                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[
                        list_of_spatial_edges.data] = h_spatial  # np.concatenate(h_spatial , o_h_spatial)
                    cell_states_edge_RNNs[
                        list_of_spatial_edges.data] = c_spatial  # np.concatenate(c_spatial , o_c_spatial)

                    # Add spatial Edges obstacle-human
                    # if obs_found:
                    #     # Add arrays for obstacle RNNs
                    #     list_of_obs_spatial_edges = Variable(
                    #         torch.LongTensor([int(x[0]) * obsNodes.size()[1] + int(x[1]) for x in obsedgeIDs])).cuda()
                    #
                    #     list_of_obs_nodes = np.array([x[0] for x in obsedgeIDs if x[0] != x[1]])
                    #
                    #     obs_edges_spatial = torch.index_select(obs_edges_current, 0, list_of_obs_spatial_edges)
                    #     hidden_obs_spatial = torch.index_select(hidden_states_obs_edge_RNNs, 0,list_of_obs_spatial_edges)
                    #     cell_obs_spatial = torch.index_select(cell_states_obs_edge_RNNs, 0, list_of_obs_spatial_edges)
                    #
                    #     # Do forward pass through obstacle EdgeRNN
                    #     o_h_spatial, o_c_spatial = self.humanObstacleEdgeRNN_Spatial(obs_edges_spatial,hidden_obs_spatial,cell_obs_spatial)
                    #
                    #     hidden_states_obs_edge_RNNs[list_of_obs_spatial_edges.data] = o_h_spatial  # np.concatenate(h_spatial , o_h_spatial)
                    #     cell_states_obs_edge_RNNs[list_of_obs_spatial_edges.data] = o_c_spatial  # np.concatenate(c_spatial , o_c_spatial)

                    # pass it to attention module
                    # For each node
                    for node in range(numNodes):
                        # Get indices of obstacles associated with the same human node at specific frame
                        # o = np.where(obsedgeIDs == node , )[0]
                        # if obs_found:
                        #     o = np.where(list_of_obs_nodes == node)[0]
                        # if len(o) > 0:
                        #     o = torch.LongTensor(o).cuda()

                        # Get the indices of spatial edges associated with this node
                        l = np.where(list_of_spatial_nodes == node)[0]
                        if len(l) == 0:
                            # If the node has no spatial edges, nothing to do
                            continue
                        l = torch.LongTensor(l).cuda()

                        # What are the other nodes with these edges?
                        node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]
                        # If it has spatial edges
                        # Get its corresponding temporal edgeRNN hidden state
                        h_node = hidden_states_nodes_from_edges_temporal[node]

                        # Do forward pass through attention module
                        # Modified : Concatenate hidden states of obstacle and human spatial edgeRNN ==> Done
                        # apply on next run : concatenation of spatial hidden states versus pooling them
                        # to capture the effect of multiple node types that have on single node
                        # if obs_found and len(o) > 0:
                        #     # if o_h_spatial[o].size()[0] == h_spatial[l].size()[0]:
                        #     spatials = torch.cat((h_spatial[l],  o_h_spatial[o]),0)
                        #     hidden_attn_weighted, attn_w = self.attn(h_node.view(1, -1) , spatials)
                        #     ## torch.matmul(torch.transpose(o_h_spatial[o], 1, 0),h_spatial[l]))
                        #     # else:
                        #     #     hidden_attn_weighted, attn_w = self.attn(h_node.view(1, -1),
                        #     #                                  torch.mul(o_h_spatial[o],h_spatial[l]) )
                        # else:
                            # h_spatial.is_leaf = True
                            # h_spatial.requires_grad = True
                        hidden_attn_weighted, attn_w = self.attn(h_node.view(1, -1),h_spatial[l])

                        # Store the attention weights
                        # Storing attn coefficients resulted from mapping each step of related trajectories
                        attn_weights[framenum][node] = (attn_w.data.cpu().numpy(), node_others)

                        # Store the output of attention module in temporary tensor
                        hidden_states_nodes_from_edges_spatial[node] = hidden_attn_weighted #torch.transpose(hidden_attn_weighted , 0, 1)

            # If there are nodes in this frame
            if len(nodeIDs) != 0:

                # Get list of nodes
                list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())

                # Get their node features
                nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes)

                # Get the hidden and cell states of the corresponding nodes
                hidden_nodes_current = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                cell_nodes_current = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)

                # Get the temporal edgeRNN hidden states corresponding to these nodes
                h_temporal_other = hidden_states_nodes_from_edges_temporal[list_of_nodes.data]
                h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                # Do a forward pass through nodeRNN
                outputs[framenum * numNodes + list_of_nodes.data], h_nodes, c_nodes = self.humanNodeRNN(nodes_current_selected, h_temporal_other, h_spatial_other, hidden_nodes_current, cell_nodes_current)

                # Update the hidden and cell states
                hidden_states_node_RNNs[list_of_nodes.data] = h_nodes
                cell_states_node_RNNs[list_of_nodes.data] = c_nodes

        # Reshape the outputs carefully
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs,  attn_weights #hidden_states_obs_node_RNNs , hidden_states_obs_edge_RNNs , cell_states_obs_node_RNNs , cell_states_obs_edge_RNNs
