from symmetricGraph import SymmetricGraph
from graph import Graph
from utility import loadDataset, explain, createNewImageDirectory

""" This file is intended to be used for custom testing:

    Visualize a graph and the model's prediction of it.
    Modify the graph, visualize it again and see how the prediction changes.
"""

# HOW TO USE:

# 1. CHOOSE GRAPH:
# Either choose graph from dataset:
dataset = loadDataset()
graph1 = dataset[0]

# Or create own symmetric graph:
graph2 = SymmetricGraph(2, 5, 1)

# Then create a Graph of a graph of your choice:
graph = Graph(graph2)

#########################
# 2. VISUALIZE THE GRAPH (under "images/graph.png")
# NOTE: Graph visualizations with less than four nodes might be empty.

explain(graph)

#########################
# 3. MODIFY THE GRAPH AS YOU LIKE:
# NOTE: It's easier to modify the graph if you reference the node (/actor) values seen in graph.png.

graph.removeEdge(1, 2)
graph.removeActors([3, 5], visualize=False)
graph.addActorsToMovie(add=3, actor=1, visualize=False)

#########################
# 4. VISUALIZE THE MODIFIED GRAPH:
# NOTE: Best debug this file in VSC and open "graph.png" to the side if you want to see the change.
explain(graph)

# Alternatively, you can specify see_change = True in graph.removeActors() or graph.addActorsToMovie(). 
# The graphs and corresponding prediction values will be saved in a new image directory in "results".
dir = createNewImageDirectory()
graph.addActorsToMovie(5, see_change=True, dir=dir)