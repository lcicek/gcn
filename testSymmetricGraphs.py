from symmetricGraph import SymmetricGraph
from modifyGraphs import ModifyGraph
from utility import loadDataset, explain

# HOW TO USE:
##########################
# 1. CHOOSE GRAPH:

# Either choose graph from dataset:
dataset = loadDataset()
graph1 = dataset[0]
graph1 = ModifyGraph(graph1)

# Or create own symmetric graph
graph2 = ModifyGraph(SymmetricGraph(2, 5, 1))

#########################
# 2. VISUALIZE THE GRAPH 
# NOTE: it will be saved under "images/graph.png" and overwrite "graph.png"

explain(graph2)

#########################
# 3. MODIFY THE GRAPH AS YOU LIKE:
# NOTE: Please reference the node/actor values seen in graph.png!

graph2.removeEdge(actor1=1, actor2=2) # removes edge from actor1 to actor2
graph2.removeActors([3, 5], visualize=False) # removes actors with values 3 and 5 from the graph
graph2.addActorsToMovie(add=3, actor=1, visualize=False) # adds 3 actors to the movie that actor=1 is in

#########################
# 4. VISUALIZE THE MODIFIED GRAPH:
# NOTE: This will overwrite the previous graph and replace "graph.png". 
#       Best debug this file in VSC and open "graph.png" to the side, to actually see the change.

explain(graph2)

#########################
# 5. OPTIONAL: PRINT THE IMPLICIT MOVIES PRESENT IN THE GRAPH:
# print(graph2.movies)