import glob
import os
import torch
import random
import copy
import pickle
import numpy as np
import networkx as nx


class Poligras(torch.nn.Module):

    def __init__(self, args):

        super(Poligras, self).__init__()
        self.args = args

        ## set up the MLP structure and learnable parameters in the policy function
        self.interLayer_first = torch.nn.Linear(1024, self.args.hidden_size1)
        self.fully_connected_second = torch.nn.Linear(self.args.hidden_size1, self.args.hidden_size2)
        self.dropout = torch.nn.Dropout(p=self.args.dropout)

        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):
        ## policy function computation steps
        
        temp_feat = torch.nn.functional.relu(self.interLayer_first(x))
        temp_feat =  self.fully_connected_second(temp_feat)
        temp_feat = torch.mm(temp_feat, torch.t(temp_feat))

        temp_feat = self.dropout(temp_feat)
        mask_temp_feat = torch.FloatTensor(np.diag([float('-inf')] * temp_feat.size()[0]))
        temp_feat = temp_feat + mask_temp_feat
        temp_feat = torch.nn.functional.softmax(temp_feat.view(1, -1), dim=1).view(temp_feat.size()[0], -1)
        assert(temp_feat.size()[0] == temp_feat.size()[1])

        return temp_feat



class PoligrasRunner(object):
    #读取数据集
    def __init__(self, args):
        print("------model initializing------\n")
        
        self.args = args
        
        self.init_graphs = {}
        self.node_feats = {}
        
        #读取数据集
        for dataset in self.args.dataset + self.args.test_datasets:
            print(f'loading {dataset}\n')
            #读取图和点
            g_file = open(f'./dataset/{dataset}/{dataset}_graph','rb')
            loaded_graph = pickle.load(g_file)
            g_file.close()
            self.init_graphs[dataset] = loaded_graph['G']
            
            g_file = open(f'./dataset/{dataset}/{dataset}_feat','rb')
            loaded_feat = pickle.load(g_file)
            g_file.close()
            self.node_feats[dataset] = loaded_feat['feat']
            print(f'loaded {dataset}')
        self.model = Poligras(self.args)
       
       
       
        
    #读取数据集的数据，做预处理
    def load_data_of_dataset(self ,dataset):
        #读取数据
        self.init_graph = self.init_graphs[dataset]
        self.node_feat = self.node_feats[dataset]
        #预处理
        #   -建立超节点集合
        init_supernode_dict = {}
        self.node_belonging = {}
        for node in self.init_graph.nodes():
            init_supernode_dict[node] = node
            self.node_belonging[node] = node
        #   -为节点编号
        ij = 0
        self.init_nd_idx = {}
        for node in self.init_graph.nodes():
            self.init_nd_idx[node] = ij
            ij += 1
        #   -超节点初始划分
        self.num_patitions = self.init_graph.number_of_nodes() // self.args.group_size
        h_function = list(range(self.init_graph.number_of_nodes()))
        random.shuffle(h_function)
        
        F_A_dict = {}
        for A in init_supernode_dict:
            F_A = self.init_graph.number_of_nodes()
            for v in init_supernode_dict[A]:
                f_v = self.init_graph.number_of_nodes()
                for u in list(self.init_graph[v]) + [v]:
                    if(f_v < h_function[self.init_nd_idx[int(u)]]):
                        f_v = h_function[self.init_nd_idx[int(u)]]
                if(f_v < F_A):
                    F_A = f_v
            F_A_dict[A] = F_A
        F_A_list = sorted(F_A_dict.items(),key=lambda item:item[1])
        
        init_groupIndex = []
        for i in range(self.num_patitions):
            curr_idx = []
            for j in F_A_list[i*len(F_A_list)/self.num_patitions:(i+1)*len(F_A_list)/self.num_patitions]:
                curr_idx.append(j[0])
            init_groupIndex.append(curr_idx)      
        
        #   -存储目前最好的超节点集
        self.best_supernode_dict = init_supernode_dict
        
        #   -存储初始分区数据
        with open(f'./{dataset}_0_best_temp','wb') as f:
            pickle.dump({'g':self.init_graph,'group_index':init_groupIndex,'superNodes_dict':init_supernode_dict},f)
        f.close()
        print(f'{dataset}数据集初始完成\n')
        
            
            
 
 
    def select_action(self, curr_feat):
        ## to select node pair according to computed selection probability matrix

        curr_probs = self.model(curr_feat) ## compute selection probability matrix 

        curr_action = curr_probs.argmax() ## select node pair with the highest probability
        curr_action_row, curr_action_col = curr_action.item() // curr_probs.size()[0], curr_action.item() % curr_probs.size()[0]

        if(curr_action_row == curr_action_col):
            curr_action_row, curr_action_col = random.sample(range(curr_probs.size()[0]), 2)
        self.model.saved_log_probs.append(torch.log(curr_probs[curr_action_row][curr_action_col]))#

        return curr_action_row, curr_action_col


    def update_graph(self, n1, n2, curr_graph):
        ## to compute the summarization reward for the given node pair, also update the intermediate supergraph if the node pair is truly merged

        curr_reward, graph_modify_dict = 0, {'weight':{}, 'if_true':{}, 'add_edge':{}}## "curr_reward" records the sr of merging n1 & n2; "graph_modify_dict" temporarily stores the modifications of graph when merging two (super)nodes, which will be truly conducted if curr_reward > 0;
        nei_n1, nei_n2 = set(self.curr_graph[n1]), set(self.curr_graph[n2])


        ## consider the cases of n1, n2's neighboring nodes
        for sd in nei_n1 & nei_n2 - set([n1]) - set([n2]):
            if(self.curr_graph[n1][sd]['if_true']):
                if(self.curr_graph[n2][sd]['if_true']):
                    curr_reward += 1
                else:
                    if((self.curr_graph[n1][sd]['weight']+self.curr_graph[n2][sd]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*len(self.superNodes_dict[sd])/2)):
                        curr_reward += (2*self.curr_graph[n2][sd]['weight'] - len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]))
                    else:
                        curr_reward += (1+ len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n1][sd]['weight'])
                        graph_modify_dict['if_true'][(n1,sd)] = False
            else:
                if(curr_graph[n2][sd]['if_true']):
                    if((self.curr_graph[n1][sd]['weight']+self.curr_graph[n2][sd]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*len(self.superNodes_dict[sd])/2)):
                        curr_reward += (2*self.curr_graph[n1][sd]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]))
                        graph_modify_dict['if_true'][(n1,sd)] = True
                    else:
                        curr_reward += (1+ len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n2][sd]['weight'])

            graph_modify_dict['weight'][(n1,sd)] = self.curr_graph[n1][sd]['weight'] + self.curr_graph[n2][sd]['weight']
        

        for sd in nei_n1 - nei_n2 - set([n1]) - set([n2]):
            if(self.curr_graph[n1][sd]['if_true']):
                if(self.curr_graph[n1][sd]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[sd]))/2)):
                    curr_reward += -len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd])
                else:
                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n1][sd]['weight'])
                    graph_modify_dict['if_true'][(n1,sd)] = False


        for sd in nei_n2 - nei_n1 - set([n1]) - set([n2]):
            if(self.curr_graph[n2][sd]['if_true']):
                if(curr_graph[n2][sd]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[sd]))/2)):
                    curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd])
                    graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':True}
                else:
                    curr_reward += (1 + len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n2][sd]['weight'])
                    graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':False}
            else:
                graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':False}


        if(n1 in nei_n2):
            if(self.curr_graph[n1][n2]['if_true']):
                if(n1 in nei_n1):
                    if(self.curr_graph[n1][n1]['if_true']):
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                curr_reward += 2
                            else:
                                if((self.curr_graph[n2][n2]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n1][n1]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (1 + 2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight']) 
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:# n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (1 - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                                
                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                    else:## self.curr_graph[n1][n1]['if_true'] = False
                        if(n2 in nei_n2): 
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                    curr_reward += (1 + 2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                            else:
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                    curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)

                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:## n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                graph_modify_dict['if_true'][(n1,n1)] = True
                                curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                else:## n1 not in self loop
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (1 - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                        else:## curr_graph[n2][n2]['if_true'] = False
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                    
                    else:## n2 not in self loop
                        if(self.curr_graph[n1][n2]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                            curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':True}

                        else:
                            curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':False}

            else:## self.curr_graph[n1][n2]['if_true'] = False
                if(n1 in nei_n1):
                    if(self.curr_graph[n1][n1]['if_true']):
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (1 + 2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            else:
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                    curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        
                        else:## n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                    else:## self.curr_graph[n1][n1]['if_true'] = False
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:## n2 not in self loop
                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                else:## n1 not in self loop
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                                curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                        else:
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                            
                    else:
                        graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':False}

        else:## n1 n2 not connected
            if(n1 in nei_n1):
                if(self.curr_graph[n1][n1]['if_true']):
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += 1
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                        else:
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                                curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                            
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                        
                        graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']


                    else:## n2 not in self loop 
                        if(self.curr_graph[n1][n1]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                            curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                        else:
                            curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                            graph_modify_dict['if_true'][(n1,n1)] = False

                else:## self.curr_graph[n1][n1]['if_true'] = False
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                                graph_modify_dict['if_true'][(n1,n1)] = True

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*curr_graph[n2][n2]['weight'])

                        graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']


            else:## n1 not in self loop
                if(n2 in nei_n2):
                    if(self.curr_graph[n2][n2]['if_true']):
                        if(self.curr_graph[n2][n2]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                            curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':True}
                        else:
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                    else:
                        graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                    


        self.model.rewards.append(curr_reward)
        if(curr_reward > 0):
            ## modify current intermediate supergraph
            for pair in graph_modify_dict['weight']:
                self.curr_graph[pair[0]][pair[1]]['weight'] = graph_modify_dict['weight'][pair]
            for pair in graph_modify_dict['if_true']:
                self.curr_graph[pair[0]][pair[1]]['if_true'] = graph_modify_dict['if_true'][pair]
            for pair in graph_modify_dict['add_edge']:
                self.curr_graph.add_edge(pair[0], pair[1], weight=graph_modify_dict['add_edge'][pair]['toAddWei'], if_true=graph_modify_dict['add_edge'][pair]['ifTrue'])

            self.curr_graph.remove_node(n2)
            
            ## update supernode features
            self.curr_feat[self.init_nd_idx[n1]] += self.curr_feat[self.init_nd_idx[n2]]
            for init_n in self.superNodes_dict[n2]:
                self.node_belonging[init_n] = n1
            self.superNodes_dict[n1] += self.superNodes_dict[n2]
            self.superNodes_dict.pop(n2)

        return curr_reward

#---------------------------------------------------------------------------------------------------------------------------------
    def fit(self):
        print("--------model initializing-----------\n")
        #怎么写捏
        #n个数据集一个接一个训练算一个小循环，要训练m次小循环
        #首先外层循环设置m次
        #再设置数据集循环
        #加载数据集数据，预处理，即load_data_of_dataset
        #调用selection获取节点对，再调用update_graph,计算损失函数更新模型参数
        #超过最好奖励就更新最好奖励，保存模型数据，小于特定奖励就重新分组（照旧）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.max_reward_by_inner_iteration = 0
        
        for epoch in range(10):
            print(f'{epoch}----{epoch}/9 (0~9)\n')
            
            for dataset in self.args.dataset:
                print(f'{dataset}开始训练\n')
                #加载、预处理数据集
                self.load_data_of_dataset(dataset)
                #进入训练模式
                self.model.train() 
                #每个数据集根据设定轮数训练 
                for count in range(self.args.counts):
                    #最大奖励、累计小奖励轮数
                    best,bad_counter = -1000000,0
                    while True:
                        g_file = open(f'./{dataset}_{count}_.best_temp','rb')
                        loaded_compre = pickle.load(g_file)
                        g_file.close()

                        self.curr_graph = loaded_compre['g']
                        self.group_index = loaded_compre['group_index']
                        self.superNodes_dict = loaded_compre['superNodes_dict']
                        self.curr_feat = copy.deepcopy(self.node_feat)
                        
                        #count_reward 本次while循环的累计奖励
                        count_reward, batch_id = 0, 0
                        traverse_time = 0
                        for idx in range(len(self.group_index)):
                            if len(self.group_index[idx]) < 3:
                                continue
                            curr_row,curr_col = self.select_action(self.curr_feat[[self.init_nd_idx[i] for i in self.group_index[idx]]])
                            curr_reward = self.update_graph(self.group_index[idx][curr_row],self.group_index[idx][curr_col],self.curr_graph)
                            if(curr_reward > 0):
                                count_reward += curr_reward
                                self.group_index[idx] = np.delete(self.group_index[idx],curr_col)
                                
                        #计算并更新策略损失
                        policy_loss = 0
                        returns = torch.FloatTensor(self.model.rewards)
                        returns = (returns-max(returns.mean,0))/returns.std()
                        
                        for log_prob, R in zip(self.model.saved_log_probs,returns):
                            policy_loss += -log_prob * R
                        #梯度清零
                        self.optimizer.zero_grad()
                        policy_loss.backward()
                        self.optimizer.step()
                        
                        print(f'count:{count},positive reward:{count_reward}\n')
                        
                        del self.model.rewards[:]
                        del self.model.saved_log_probs[:]
                        
                        # 设置奖励更新规则
                        if count < 5:
                            ratio = 0.001
                        else:
                            ratio = 0.01

                        if count_reward > (1 + ratio) * best:
                            best, bad_counter = count_reward, 0
                            self.best_graph, self.best_currFeat, self.best_groupIndex = self.curr_graph, self.curr_feat, self.group_index
                            self.best_superNodes_dict = self.superNodes_dict
                        else:
                            bad_counter += 1

                        # 判断是否停止当前数据集的训练
                        if bad_counter == self.args.bad_counter:
                            break

                    # 判断是否需要重新分组
                    if best > self.max_reward_by_inner_iter:
                        self.max_reward_by_inner_iter = best
                    elif best < (self.max_reward_by_inner_iter / 3):
                        self.max_reward_by_inner_iter = 0
                        assert self.best_graph.number_of_nodes() == len(self.best_superNodes_dict)

                        self.num_partitions = self.best_graph.number_of_nodes() // self.args.group_size

                        h_function = list(range(self.init_graph.number_of_nodes()))
                        random.shuffle(h_function)

                        F_A_dict = {}
                        for A in self.best_superNodes_dict:
                            F_A = self.init_graph.number_of_nodes()
                            for v in self.best_superNodes_dict[A]:
                                f_v = self.init_graph.number_of_nodes()
                                for u in list(self.init_graph[v]) + [v]:
                                    if h_function[self.init_nd_idx[int(u)]] < f_v:
                                        f_v = h_function[self.init_nd_idx[int(u)]]

                                if f_v < F_A:
                                    F_A = f_v

                            F_A_dict[A] = F_A
                        F_A_list = sorted(F_A_dict.items(), key=lambda item: item[1])

                        self.best_groupIndex = []
                        for i in range(self.num_partitions):
                            curr_idx = []
                            for j in F_A_list[int(i * len(F_A_list) / self.num_partitions): int((i + 1) * len(F_A_list) / self.num_partitions)]:
                                curr_idx.append(j[0])
                            self.best_groupIndex.append(np.array(curr_idx))

                    # 保存当前最好的结果
                    self.node_feat = self.best_currFeat
                    f = open('./{}_{}_.best_temp'.format(self.args.dataset, count + 1), 'wb')
                    pickle.dump({'g': self.best_graph, 'group_index': self.best_groupIndex, 'superNodes_dict': self.best_superNodes_dict}, f)
                    f.close()

                    # 删除之前保存的临时文件
                    files = glob.glob('./{}_{}_.best_temp'.format(self.args.dataset, count))
                    for fil in files:
                        os.remove(fil)

                # 删除该数据集的所有临时文件
                files = glob.glob('./{}_*_.best_temp'.format(self.args.dataset))
                for fil in files:
                    os.remove(fil)
        
            print(f"\n---- Finished training on all datasets for epoch {epoch + 1} ----\n")
            
        print("\n-------Training completed---------.\n")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "trained_model_checkpoint.pth")
        print("模型参数和状态保存成功: trained_model_checkpoint.pth")
                        
                                

#---------------------------------------------------------------------------------------------------------------------------------
    def encode(self):
        ## encode superedges after finishing the graph summarization iterations
        print("\n-------Model encoding---------.\n")

        self.superEdges, self_edge = [], []  ## to store the superedges and the initial self-loop edges on initial nodes
        self.correctionSet_plus, self.correctionSet_minus = [], [] ## to store the correction set edges to add and to delete from the supergraph when restoring the initial graph

        finished_pair, i_dx = {}, 0
        self.superNodes_dict = self.best_superNodes_dict
        for A in self.superNodes_dict:
            iterative_superNode = []
            # print('{}th supernode'.format(i_dx))
            for init_n in self.superNodes_dict[A]:
                for nei_n in self.init_graph[init_n]:
                    iterative_superNode.append(self.node_belonging[nei_n])

            for B in set(iterative_superNode):
                if(A == B):
                    continue
                if((A, B) in finished_pair):
                    continue
                else:
                    finished_pair[(A,B)] = 0
                    finished_pair[(B,A)] = 0
            

                Edge_AB = [] # 0
                Pi_E_AB = []
                for n1 in self.superNodes_dict[A]:
                    for n2 in self.superNodes_dict[B]:
                        if((n1, n2) in self.init_graph.edges()):
                            Edge_AB.append((n1, n2))
                        else:
                            Pi_E_AB.append((n1, n2))

                if(len(Edge_AB) <= (len(self.superNodes_dict[A])*len(self.superNodes_dict[B])/2)):
                    self.correctionSet_plus += Edge_AB
                else:
                    self.superEdges.append((A, B))# += 1#
                    self.correctionSet_minus += Pi_E_AB


            Edge_AA = []
            Pi_E_AA = []
            for n1 in self.superNodes_dict[A]:
                for n2 in self.superNodes_dict[A]:
                    if(n1<n2):
                        if((n1, n2) in self.init_graph.edges()):
                            Edge_AA.append((n1, n2))# += 1
                        else:
                            Pi_E_AA.append((n1, n2))
                ## to store the initial nodes having the self-loop edge 
                if((n1, n1) in self.init_graph.edges()): 
                    self_edge.append(n1)



            if(len(Edge_AA) <= (len(self.superNodes_dict[A])*(len(self.superNodes_dict[A])-1)/4)):
                self.correctionSet_plus += Edge_AA
            else:
                self.superEdges.append((A, A))
                self.correctionSet_minus += Pi_E_AA

            i_dx += 1


        print('==============================\n')

        print('#super edge: ', len(self.superEdges))
        print('correction set size: ', len(self.correctionSet_plus) + len(self.correctionSet_minus))
        print("\n-------SuperNode encoding ended, total reward is {}---------.\n".format(self.init_graph.number_of_edges() - len(self_edge) - len(self.superEdges) - len(self.correctionSet_plus) - len(self.correctionSet_minus)))


        f = open('./{}_graph_summary'.format(self.args.dataset), 'wb')
        pickle.dump({'superNodes_dict': self.superNodes_dict, 'superEdge_list': self.superEdges, 'self_edge_list': self_edge, 'correctionSet_plus_list': self.correctionSet_plus, 'correctionSet_minus_list': self.correctionSet_minus}, f)
        f.close()
#---------------------------------------------------------------------------------------------------------------------------------
    def test(self):
        print("\n-------Model testing---------.\n")
        # 加载模型检查点
        checkpoint = torch.load("trained_model_checkpoint.pth")
        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 切换到评估模式
        self.model.eval()

        total_rewards = 0  # 总奖励
        correct_predictions = 0  # 正确预测数量
        total_predictions = 0  # 总预测数量

        # 遍历每个测试数据集，假设 self.args.test_datasets 包含测试集名称列表
        for dataset in self.args.test_datasets:
            
            self.load_data_of_dataset(dataset)

            # 初始化当前测试数据集的图结构和节点特征
            # 假设 self.init_test_graph 和 self.init_test_node_feat 是测试数据集的图和特征
            self.curr_graph = copy.deepcopy(self.init_graph)
            self.curr_feat = copy.deepcopy(self.node_feat)
            
            # 假设 self.init_test_group_index 是测试数据集的初始节点分组
            self.group_index = copy.deepcopy(self.init_group_index)
            
            correct_predictions, total_predictions = 0, 0  # 初始化测试集的预测计数
            total_rewards = 0  # 初始化总奖励

            # 使用 no_grad 来禁用梯度计算
            with torch.no_grad():
                # 遍历节点组进行测试
                for idx in range(len(self.group_index)):
                    if len(self.group_index[idx]) < 3:
                        continue
                    # 使用 select_action 函数选择要合并的节点对（模拟模型决策）
                    curr_row, curr_col = self.select_action(self.curr_feat[[self.init_nd_idx[i] for i in self.group_index[idx]]])

                    # 通过 update_graph 函数执行测试，获取奖励值
                    curr_reward = self.update_graph(self.group_index[idx][curr_row], self.group_index[idx][curr_col], self.curr_graph)

                    # 计算预测的表现（假设 curr_reward 为正时表示正确预测）
                    if curr_reward > 0:
                        correct_predictions += 1
                    total_predictions += 1

                    total_rewards += curr_reward

            # 输出每个数据集的测试结果
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Dataset: {self.args.dataset}; Total rewards: {total_rewards}; Accuracy: {accuracy:.4f}\n")

        print(f"\n-------Testing completed with overall accuracy: {accuracy:.4f} -------.\n")
