import torch
import pickle
import networkx as nx
import numpy as np
import argparse
from os.path import basename



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run feature generation code.")
    parser.add_argument("--dataset", nargs="?", default="in-2004", help="Dataset name")
    args = parser.parse_args()

    
    interval_size = 1000 # interval size is defined as $|V|/d$, where $|V|$ is the number of nodes and $d$ is final output node embedding size

    g_file = open('./dataset/' + args.dataset + '/' + args.dataset + '_graph', 'rb')
    loaded_graph = pickle.load(g_file)
    g_file.close()
    g = loaded_graph['G']
    num_node = g.number_of_nodes()
    print('# nodes: ', num_node)
    print('# edges: ', g.number_of_edges())


    node_dict = {}#
    nd_id = 0
    for nd in g.nodes():
        node_dict[nd] = nd_id
        nd_id += 1

    num = 0
    feat_list, feat_size = [], num_node//interval_size + 1

    for nd in g.nodes():#range(num_node):
        print('nd', num)
        nei_list = list(g[nd])
        

        curr_feat = [0.0]*feat_size#[0.0]*(min_nei//interval_size)
        #t1 = time.time()
        for nei in nei_list:
            row = node_dict[nei]//interval_size
            curr_feat[row] += 1.0
            
        feat_list.append(curr_feat)
        num += 1

    node_feat = torch.FloatTensor(feat_list)


    f = open(args.dataset + '_feat', 'wb')
    data = {'feat': node_feat}
    pickle.dump(data, f)
    f.close()
