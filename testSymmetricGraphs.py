from symmetricGraph import SymmetricGraph
from modifyGraphs import ModifyGraph
from calculateMovie import movie, model
from utility import loadDataset, explain
import copy

# TEST-CASES:
graphs = []

graph = ModifyGraph(SymmetricGraph(3, 8, 1))

graph.addActorsToMovie(1, 1, visualize=False)
graphs.append(copy.deepcopy(graph))
graph.addActorsToMovie(2, 8, visualize=False)
graphs.append(copy.deepcopy(graph))
graph.addActorsToMovie(3, 15, visualize=False)
graphs.append(copy.deepcopy(graph))
graph.addActorsToMovie(4, 22, visualize=False)
graphs.append(copy.deepcopy(graph))

explain(graphs[3])

for graph in graphs:
    pred = model(graph.x, graph.edge_index).item()

    approximation = 0
    for m in graph.movies:
        degree = len(m)-1
        approximation += movie[degree]

    approximation /=  (len(graph.movies))

    print(f"predict = {pred}")
    print(f"approx  = {approximation}")
    print("------------")

"""
graphs.append(sg(4, 3, 1))
graphs.append(sg(2, 4, 1))
graphs.append(sg(8, 5, 1))
graphs.append(sg(10, 6, 1))
graphs.append(sg(4, 7, 1))
graphs.append(sg(5, 8, 1))
graphs.append(sg(6, 9, 1))
graphs.append(sg(7, 10, 1))
graphs.append(sg(9, 11, 1))
graphs.append(sg(3, 12, 1))

for graph in graphs:
    pred = model(graph.x, graph.edge_index).item()
    approximation = movie[graph.cast] # in this case approx = val because all movies have same cast

    print(f"predict = {pred}")
    print(f"approx  = {approximation}")
    print("------------")
"""