import pickle
from glob import glob
import networkx as nx
import numpy as np
import argparse
from os.path import basename


def load_initial_graph(dataset_name):
    dir = './' + dataset_name + '/' + dataset_name + '.graph-txt'

    G = nx.Graph()

    count = 0
    with open(dir, 'r', encoding='UTF-8') as file:
        line = file.readline()
        while(line is not None and line != ''):
            line_list = line.split(' ')
            for ed in line_list[:-1]:
                if(count != int(ed)):
                    ## There are two atrributes for each edge: "weight" and "if_true". The "weight" is to record 
                    ## the number of initial edges between every two supernodes during the node-mergings, it is 
                    ## initialized as 1 because initially there is only one superedge between every two supernodes; 
                    ## the "if_true" is to record if there is a superedge between the two supernodes.
                    ## These two attributes are designed to efficiently compute the summarization rewards during the 
                    ## node-merging process.
                    G.add_edge(count, int(ed), weight=1, if_true=True) 
            

            count += 1
            line = file.readline()


    print('num nodes', G.number_of_nodes())
    print('num edges', G.number_of_edges())
    
    return G




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run networkx graph generation code.")
    parser.add_argument("--dataset", nargs="?", default="in-2004", help="Dataset name")
    args = parser.parse_args()

    
    g = load_initial_graph(dataset_name=args.dataset)


    f = open(args.dataset + '_graph', 'wb')
    data = {'G': g}
    pickle.dump(data, f)
    f.close()

