from utility import loadDataset, loadModel
from modifyGraphs import ModifyGraph
from calculateMovie import movie
from random import randrange

def approximate(graph):
    messiness, overlap = calculateMessiness(graph)

    if overlap == 0: # one movie graph can be approximated easily
        approx = movie[graph.num_actors]
        if approx > 0.5:
            return 1
        else:
            return 0
    elif overlap == 1: # graph can be approximated if size of cast doesn't vary too much
        approx = 0
        for m in graph.movies:
            approx += movie[len(m)]

        approx /= graph.num_movies

        if approx > 0.5:
            return 1
        else:
            return 0
    elif messiness > 0.7: # messiness predictor for 0
        return 0
    elif messiness < 0.3: # cleanness predictor for 1
        return 1
    else: # if we arrive here, we are uncertain about the prediction
        return 0 if graph.num_actors > 25 else 1

def calculateMessiness(graph):
    # higher overlap => higher messiness
    overlap = findOverlap(graph)
    overlap_weight = overlap / graph.num_movies

    # higher number of movies => slightly higher messiness
    movie_weight = graph.num_movies / 135 # 135 = max num_movies

    # higher cast => higher messiness
    cast_weight = averageCast(graph) / 135

    messiness = overlap_weight + movie_weight + cast_weight

    return messiness, overlap

def averageCast(graph):
    avg_cast = 0
    for m in graph.movies:
        avg_cast += len(m)

    avg_cast /= graph.num_movies

    return avg_cast

def findOverlap(graph): # finds how many actors are in at least two movies
    overlap = 0

    for actor in range(graph.num_actors):
        appears = 0

        for m in graph.movies:
            if actor in m:
                appears += 1
                
                if appears > 1:
                    overlap += 1
                    break
    
    return overlap

model = loadModel()
accuracy = 0
dataset = loadDataset()
for i in range(150):
    graph = dataset[i]
    label, _ = model.evaluate(graph.x, graph.edge_index, batch=None)

    graph = ModifyGraph(graph)
    approx_label = approximate(graph)

    accuracy += (label.item() == approx_label)

for i in range(850, 1000):
    graph = dataset[i]
    label, _ = model.evaluate(graph.x, graph.edge_index, batch=None)
    
    graph = ModifyGraph(graph)
    approx_label = approximate(graph)

    accuracy += (label.item() == approx_label)

accuracy /= 300

print(f"Model can be approximated with probability: {accuracy}")