from torch_geometric.datasets import TUDataset
from statistics import fmean, stdev, median

def collectStats(n, e, label): # n = collected num_nodes, e = collected num_edges
    # Node information for whole graph:
    max_n = max(n)
    min_n = min(n)
    mean_n = fmean(n)
    median_n = median(n)
    stdev_n = stdev(n)
    
    max_e = max(e)
    min_e = min(e)
    mean_e = fmean(e)
    median_e = median(e)
    stdev_e = stdev(e)

    str_n = f'NUM_NODES: Max = {max_n} | Min = {min_n} | Median = {median_n:.2f} | Mean = {mean_n:.2f} | Stdev = {stdev_n:.2f}'
    str_e = f'NUM_EDGES: Max = {max_e} | Min = {min_e} | Median = {median_e:.2f} | Mean = {mean_e:.2f} | Stdev = {stdev_e:.2f}'

    print(f'{label}-Graphs:')
    print(str_n)
    print(str_e)
    print()

def collectEdges(edges_out):
    n_edges_out = []

    count = 0
    curr_elem = edges_out[0]
    for elem in edges_out:
        if elem == curr_elem:
            count += 1
        else:
            n_edges_out.append(count)
            curr_elem = elem
            count = 1
    
    return n_edges_out
        

def collect(dset):
    n_nodes = [] # list of number of nodes in each graph
    n_edges = [] # list of number of outgoing edges for each node

    for data in dset:
        n_nodes.append(data.num_nodes)
        n_edges.extend(collectEdges(data.edge_index[0]))
    
    return n_nodes, n_edges

dataset = TUDataset("./", "IMDB-BINARY", use_node_attr=True)

dataset_zero = dataset[0:500] # should contain roughly equal amount of 0 and 1 labels since dataset was shuffled
dataset_one = dataset[500:1000]

n0, e0 = collect(dataset_zero)
n1, e1 = collect(dataset_one)

collectStats(n0, e0, 0)
collectStats(n1, e1, 1)