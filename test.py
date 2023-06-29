"""
Idea 2:
Low cast predicts 1-Graphs.

Result:
Fixed number of movies but increasing cast:

    | Cast | Pred |
    ______________
    | 2-6 |   1   | 
    | 7+  |  0.5  | (pred is technically 0 or 1 only but at a rate of about 0.5)
    ______________

Interpretation:
Low cast is likely a predictor for 1-Graphs.
"""

from main import *

class CustomGraph:
    def __init__(self, num_nodes=11, degree=5): # default values
        self.num_nodes = num_nodes
        self.x = torch.zeros((num_nodes, 136))
        # recreate edge_index[2, ...]
        self.start_nodes = [] # an edge goes from a start node...
        self.end_nodes = [] # ...to an end node

        for i in range(num_nodes):
            if i == 0:
                self.x[i][num_nodes - 1] = 1 # central node has connection to all others

                for j in range(num_nodes-1):
                    self.start_nodes.append(i)
                    self.end_nodes.append((j+1))
            else:
                self.x[i][degree] = 1

                for j in range(degree):
                    self.start_nodes.append(i)
            
                    group_idx = (i-1) // degree # i-1 because our sub-graphs start at 1, not 0
                    target = degree * group_idx + j + 1
                    
                    if target == i:
                        self.end_nodes.append(0) # add connection to central node instead of node to self
                    else:
                        self.end_nodes.append(target)
        
        self.updateEdgeIndex()
        self.y = 0

    def updateEdgeIndex(self):
        self.edge_index = torch.tensor([self.start_nodes, self.end_nodes])

    def addNodes(self, start, end):
        self.start_nodes.extend(start)
        self.end_nodes.extend(end)

        self.updateEdgeIndex()

    def removeNode(self, node):
        delete_indices = []
        
        for i in range(len(self.start_nodes)):
            if self.start_nodes[i] == node or self.end_nodes[i] == node:
                delete_indices.append(i) # remember edge to delete

                if self.start_nodes[i] == node: # degree of target has to be decremented
                    target = self.end_nodes[i] 
                else:
                    target = self.start_nodes[i]

                degree = self.x[target].nonzero().item() # decrement degree
                self.x[target][degree-1] = 1
                self.x[target][degree] = 0

        delete_indices.reverse()
        for i in delete_indices: # delete edge
            self.start_nodes.pop(i)
            self.end_nodes.pop(i)

        # decrement indices accordingly
        for i in range(len(self.start_nodes)):
            if self.start_nodes[i] > node:
                self.start_nodes[i] -= 1
            
            if self.end_nodes[i] > node:
                self.end_nodes[i] -= 1
        
        # update edge index
        self.updateEdgeIndex()

        # update x
        keep_indices = list(range(0, len(self.x)))
        keep_indices.remove(node)
        self.x = self.x[keep_indices]
    
    def overwrite(self, x, edge_index):
        self.x = x
        self.start_nodes = edge_index[0]
        self.start_nodes = edge_index[1]
        self.edge_index = edge_index

print("Testing custom graph containing num_nodes // degree movies and central actor/node:")
num_movies = 3
degree = 6
graph = CustomGraph(num_nodes=num_movies*degree+1, degree=degree)

