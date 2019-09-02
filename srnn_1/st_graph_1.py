'''
ST-graph data structure script for the structural RNN implementation
Takes a batch of sequences and generates corresponding ST-graphs

Author : Sirin Haddad
Extension from (social attention 2018 source code)
'''
import numpy as np
# from helper import getVector, getMagnitudeAndDirection , get_obstacle_zones
import helper
import pickle as pkl

class ST_GRAPH():

    def __init__(self, batch_size=50, seq_length=5 ):
        '''
        Initializer function for the ST graph class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        '''
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.ds_ptr = 0

        self.nodes = [{} for i in range(batch_size)]
        self.edges = [{} for i in range(batch_size)]

    def reset(self):
        self.nodes = [{} for i in range(self.batch_size)]
        self.edges = [{} for i in range(self.batch_size)]

    def readGraph(self, source_batch , ds_ptr=0, threshold = 0.8):
        '''
        Main function that constructs the ST graph from the batch data
        params:
        source_batch : List of lists of numpy arrays. Each numpy array corresponds to a frame in the sequence.
        '''

        # @ Serene

        # Add obstacle node (influence pedestrian path)
        # obs_positions = {}
        # with open('/home/serene/Desktop/srnn-pytorch-master/data/eth/hotel/obs_map.pkl', 'rb') as pkl_f:
        #     content = pkl.load(pkl_f)

        # Check obstacles if defined beforehand
        # helper.set_obstacle_zones() #Use only once for outlining environment
        ## TODO: call obs_map only once
        # obs_f = open('/home/serene/Documents/copy_srnn_pytorch_2/srnn-pytorch-master/data/obstacles_loc.cpkl', 'rb')
        # traj_obs = open('./srnn-pytorch-master/data/obstacles_loc.cpkl','rb')
        # obs_f = open('./data/obstacles_loc.cpkl','rb')
        # obs_map = pkl.load(obs_f)
        # if obs_map == ():
        #     helper.set_obstacle_zones()

        for sequence in range(self.batch_size):
            # source_seq is a list of numpy arrays
            # where each numpy array corresponds to a single frame
            source_seq = source_batch[sequence]
            for framenum in range(self.seq_length):
                # Each frame is a numpy array
                # each row in the array is of the form
                # pedID, y, x
                frame = source_seq[framenum]

                # Add nodes
                for ped in range(frame.shape[0]):
                    pedID = frame[ped, 0]
                    # Note: 6-FEB the normalized data of range [-1,1] has (v,u) order
                    x = frame[ped, 2]
                    y = frame[ped, 1]
                    pos = (x, y)

                    if pedID not in self.nodes[sequence]:
                        node_type = 'H'
                        node_id = pedID
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        self.nodes[sequence][pedID] = ST_NODE(node_type, node_id, node_pos_list)
                    else:
                        self.nodes[sequence][pedID].addPosition(pos, framenum)
                        # Add Temporal edge between the node at current time-step
                        # and the node at previous time-step
                        edge_id = (pedID, pedID)## apply on next run
                        if self.nodes[sequence][pedID].node_type == 'H':
                            pos_edge = (self.nodes[sequence][pedID].getPosition(framenum-1), pos)
                            if edge_id not in self.edges[sequence]:
                                edge_type = 'H-H/T'
                                edge_pos_list = {}
                                # ASSUMPTION: Adding temporal edge at the later time-step
                                edge_pos_list[framenum] = pos_edge
                                self.edges[sequence][edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                            else:
                                self.edges[sequence][edge_id].addPosition(pos_edge, framenum)

                # ASSUMPTION:
                # Adding spatial edges between all pairs of pedestrians.
                # TODO: started on 6th April 2018
                # Can be pruned by considering pedestrians who are close to each other
                # adding an edge between two nodes indicate a direct interaction / while it might not necessarily exist
                # @me making fully connected graph is expensive on large scale

                # Add spatial edges
                ## New 6 th April : making fully dynamic LSTM graph
                ## dynamically establish H_H edges based on their proximity

                for ped_in in range(frame.shape[0]):
                    for ped_out in range(ped_in+1, frame.shape[0]):
                        ped_ped_dist = helper.human_human_distance(ped_in , ped_out)
                        if ped_ped_dist <= threshold: # H-H threshold for fully-dynamic graph
                            pedID_in = frame[ped_in, 0]
                            pedID_out = frame[ped_out, 0]
                            pos_in = (frame[ped_in, 1], frame[ped_in, 2])
                            pos_out = (frame[ped_out, 1], frame[ped_out, 2])
                            pos = (pos_in, pos_out)
                            edge_id = (pedID_in, pedID_out)
                            # ASSUMPTION:
                            # Assuming that pedIDs always are in increasing order in the input batch data ?? Resolved in normalized data
                            # Date :
                            if edge_id not in self.edges[sequence]:
                                edge_type = 'H-H/S'
                                edge_pos_list = {}
                                edge_pos_list[framenum] = pos
                                self.edges[sequence][edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                            else:
                                self.edges[sequence][edge_id].addPosition(pos, framenum)

                # For now people found in the vicinity of obstacle area that there is only the spatial edge connecting the two node types(H and O),
                # the vicinity can be determined empirically
                # the obstacle node is not connected temporally as it is static however we want the network to learn the impact of such an obstacle
                # an obstacle doesn't necessarily pose deviational power , depending on the semantics of the scene an obstacle can be a building
                # different pedestrians have different intents, one may want to enter the building , other people may not
                # so it is not deviational influence all the time , it can be gravitational(e.g. train stopping to load passengers)

                


    def printGraph(self):
        '''
        Print function for the graph
        For debugging purposes
        '''
        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print('Printing Nodes')
            print('===============================')
            for node in nodes.values():
                node.printNode()
                print('--------------')

            print
            print('Printing Edges')
            print('===============================')
            for edge in edges.values():
                edge.printEdge()
                print('--------------')

    def getSequence(self):
        '''
        Gets the sequence
        '''
        nodes = self.nodes[0]
        edges = self.edges[0]

        numNodes = len(nodes.keys())

        list_of_nodes = {}

        retNodes = np.zeros((self.seq_length, numNodes, 2))
        retEdges = np.zeros((self.seq_length, numNodes*numNodes, 2))  # Diagonal contains temporal edges
        retNodePresent = [[] for c in range(self.seq_length)]
        retEdgePresent = [[] for c in range(self.seq_length)]

        # retObsNodes = np.zeros((self.seq_length,numNodes, 2))
        # retObsEdges = np.zeros((self.seq_length,numNodes*numNodes, 2))
        # retObsNodePresent = [[] for c in range(self.seq_length)]
        # retObsEdgePresent = [[] for c in range(self.seq_length)]

        for i, ped in enumerate(nodes.keys()):
            list_of_nodes[ped] = i
            pos_list = nodes[ped].node_pos_list
            for framenum in range(self.seq_length):
                if nodes[ped].node_type == 'H':
                    if framenum in pos_list:
                        retNodePresent[framenum].append(i)
                        retNodes[framenum, i, :] = list(pos_list[framenum])
                # elif nodes[ped].node_type == 'O':
                #     if framenum in nodes[ped].obs_node_pos_list:
                        # retObsNodePresent[framenum].append(i)
                        # retObsNodes[framenum, i, :] = list(nodes[ped].obs_node_pos_list[framenum])

        for ped, ped_other in edges.keys():
            i, j = list_of_nodes[ped], list_of_nodes[ped_other]
            edge = edges[(ped, ped_other)]

            if ped == ped_other:
                # Temporal edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j))
                        retEdges[framenum, i*(numNodes) + j, :] = helper.getVector(edge.edge_pos_list[framenum])
            else:
                # Spatial edge
                # Spatial edges are bi-directional and the opposite direction has negated values of positions
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        # if edges.get((ped,ped_other)).edge_type == 'H-H/S':
                            retEdgePresent[framenum].append((i, j))
                            retEdgePresent[framenum].append((j, i))
                            # the position returned is a tuple of tuples

                            retEdges[framenum, i*numNodes + j, :] = helper.getVector(edge.edge_pos_list[framenum])
                            retEdges[framenum, j*numNodes + i, :] = -np.copy(retEdges[framenum, i*(numNodes) + j, :])
                    # elif framenum in edge.ped_obs_edge:
                    #     # if edges.get((ped,ped_other)).edge_type == 'H-O/S':
                    #         retObsEdgePresent[framenum].append((i,j))
                    #         # retObsEdgePresent[framenum].append((j,i))
                    #         retObsEdges[framenum, i*numNodes + j, :] = helper.getVector(edge.ped_obs_edge[framenum])
                            # retObsEdges[framenum, j*numNodes + i, :] = -np.copy(retObsEdges[framenum, i*(numNodes) + j, :])

        return retNodes, retEdges, retNodePresent, retEdgePresent #, retObsNodes, retObsEdges ,retObsNodePresent , retObsEdgePresent


class ST_NODE():

    def __init__(self, node_type, node_id, node_pos_list):
        '''
        Initializer function for the ST node class
        params:
        node_type : Type of the node (Human or Obstacle)
        node_id : Pedestrian ID or the obstacle ID
        node_pos_list : Positions of the entity associated with the node in the sequence
        '''
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list= {}
        self.obs_node_pos_list= {}

        if node_type == 'O':
            self.obs_node_pos_list = node_pos_list
        else:
            self.node_pos_list = node_pos_list


    def getPosition(self, index):
        '''
        Get the position of the node at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.node_pos_list)
        return self.node_pos_list[index]

    def getType(self):
        '''
        Get node type
        '''
        return self.node_type

    def getID(self):
        '''
        Get node ID
        '''
        return self.node_id

    def addPosition(self, pos, index):
        '''
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.node_pos_list)
        self.node_pos_list[index] = pos

    def printNode(self):
        '''
        Print function for the node
        For debugging purposes
        '''
        print('Node type:', self.node_type, 'with ID:', self.node_id, 'with positions:', self.node_pos_list.values(), 'at time-steps:', self.node_pos_list.keys())


class ST_EDGE():

    def __init__(self, edge_type, edge_id, edge_pos_list):
        '''
        Initializer function for the ST edge class
        params:
        edge_type : Type of the edge (Human-Human or Human-Obstacle)
        edge_id : Tuple (or set) of node IDs involved with the edge
        edge_pos_list : Positions of the nodes involved with the edge
        '''
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.ped_obs_edge = {}
        self.edge_pos_list = {}

        if edge_type == 'H_O/S':
            self.ped_obs_edge = edge_pos_list
        else:
            self.edge_pos_list = edge_pos_list

    def getPositions(self, index):
        '''
        Get Positions of the nodes at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.edge_pos_list)
        return self.edge_pos_list[index]

    def getType(self):
        '''
        Get edge type
        '''
        return self.edge_type

    def getID(self):
        '''
        Get edge ID
        '''
        return self.edge_id
    def addObstacleEdge(self , pos,index):
        '''
        After inserting H-H edges spatial/temporal we insert H-O spatial edges,
        by allowing more than one edge type under the same frame index in edge_pos_list
        pos : A tuple (ped(x,y) , distance to obstacle)
        index: time-step
        '''
        if index not in self.ped_obs_edge:
            self.ped_obs_edge[index] = pos
        else:
            list = self.ped_obs_edge[index]
            self.ped_obs_edge[index] = np.append(list , pos)

    def addPosition(self, pos, index):
        '''
        Add a position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.edge_pos_list)
        # print index
        self.edge_pos_list [index] = pos

    def printEdge(self):
        '''
        Print function for the edge
        For debugging purposes
        '''
        print('Edge type:', self.edge_type, 'between nodes:', self.edge_id, 'at time-steps:', self.edge_pos_list.keys())
