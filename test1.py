"""
Idea:
Differently sized casts affect predictions, i.e. num_nodes per graph affects prediction.

Result:
High num_nodes => Prediction = 0
Low num_nodes => Prediction = 1 (even though a lot more variance here)
"""

from main import *

class one_movie_graph:
    def __init__(self, num_nodes):
        self.x = torch.zeros((num_nodes, 136))

        # recreate edge_index[2, ...]
        start_nodes = [] # an edge goes from a start node...
        end_nodes = [] # ...to an end node

        for i in range(num_nodes):
            self.x[i][num_nodes - 1] = 1 # all have same degree

            for j in range(i, num_nodes-1 + i): # every node has num_nodes-1 edges
                start_nodes.append(i)
                end_nodes.append((j+1) % num_nodes)
                    

        self.edge_index = torch.tensor([start_nodes, end_nodes])
        self.y = 0 if num_nodes < 50 else 1

explain(one_movie_graph(75))

print("Testing custom graphs containing just one movie:")
print("Graphs with num_nodes 1-20:")
for i in range(2, 20, 1):
    explain(one_movie_graph(i))

print("Graphs with num_nodes 40, 45, 50, ... 120:")
for i in range(40, 120, 5):
    explain(one_movie_graph(i))