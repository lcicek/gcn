"""
Idea:
Presence of a central node that connects all other clusters (i.e. "star actor" that stars
in otherwise uniquely casted (disjoint) movies) affects prediction.

Result:
Presence of such a central node => Prediction = 1
"""

from main import *

class central_node_graph:
    def __init__(self, num_nodes, degree):
        self.x = torch.zeros((num_nodes, 136))

        # recreate edge_index[2, ...]
        start_nodes = [] # an edge goes from a start node...
        end_nodes = [] # ...to an end node

        for i in range(num_nodes):
            if i == 0:
                self.x[i][num_nodes - 1] = 1 # central node has connection to all others

                for j in range(i, num_nodes-1 + i):
                    start_nodes.append(i)
                    end_nodes.append((j+1) % num_nodes)
            else:
                self.x[i][degree] = 1 # 

                for j in range(degree):
                    start_nodes.append(i)
            
                    group_idx = (i-1) // degree # i-1 because our sub-graphs start at 1, not 0
                    target = degree * group_idx + j + 1
                    
                    if target == i:
                        end_nodes.append(0) # add connection to central node instead of node to self
                    else:
                        end_nodes.append(target)
        

        self.edge_index = torch.tensor([start_nodes, end_nodes])
        self.y = 0

print("Testing custom graph containing num_nodes // degree movies and central actor/node:")
explain(central_node_graph(num_nodes=16, degree=3)) # assert: num_nodes-1 % degree == 0