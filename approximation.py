from utility import loadDataset, loadModel
from modifyGraphs import ModifyGraph
from calculateMovie import movie

def calculateApproximation(graph, label=False):
    messiness, overlap = calculateMessiness(graph)

    if overlap == 0: # one movie graph predictions are fixed, so it's the most important feature if present
        if label:
            return 1 if movie[graph.num_actors] > 0.5 else 0
        else:
            return movie[graph.num_actors]
    elif messiness > 1: # high messiness is predictor for 0
        return 0
    elif messiness < 0.3: # low messiness is predictor for 1
        return 1
    elif overlap == 1: # single central node graph can be roughly approximated 
        approx = 0
        for m in graph.movies:
            approx += movie[len(m)]

        approx /= graph.num_movies

        if label:
            if approx > 0.5:
                approx = 1
            else:
                approx = 0

        return approx
    else: # if all else fails, just used number of actors as a rough estimate
        return 0 if graph.num_actors > 25 else 1

def calculateMessiness(graph):
    # higher overlap => higher messiness
    overlap = findOverlap(graph)
    overlap_weight = overlap / graph.num_movies # always between 0 and 1

    # increase messiness if number of movie and size of cast is high
    movie_weight = graph.num_movies / 136
    cast_weight = averageCast(graph) / 136

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

def calculateAccuracy(graphs, model, tolerance, label=False):
    accuracy = 0
    for i in range(0, 150):
        graph = graphs[i]

        if label:
            pred, _ = model.evaluate(graph.x, graph.edge_index, batch=None)
        else:
            pred = model(graph.x, graph.edge_index).item()

        graph = ModifyGraph(graph)
        approx = calculateApproximation(graph, label=label)

        correct_approx = 1 if abs(pred - approx) <= tolerance else 0
        accuracy += correct_approx
    
    return accuracy

def approximateModel(label=False):
    """ Calculates how accurately the model's prediction is approximated.
    An approximation is considered successful if the difference to the prediction
    is no greater than 0.25. The justification is:
    A prediction of 0.25 can be understood as a prediction of 0.5 with a halfway
    nudge towards 0, i.e. the model is not that sure about the prediction but thinks
    that it's probably a zero.
    Thus, we would consider an approximation between 0 and 0.5 successful.
    Similarly, if a prediction is 0.5 (the model is completely uncertain), then any
    approximation between 0.25 and 0.75 would be considered successful.
    
    [ Note: Decreasing this tolerance to 0.15, decreases the accuracy to 0.63. ]

    With this definition, the approximation is able to achieve an accuracy of 0.77.
    If we don't approximate the model's prediction but, instead, the model's predicted
    label, the accuracy of the approximation increases to 0.83.
    """

    model = loadModel()
    dataset = loadDataset()

    # use unseen test-set graphs:
    zero_graphs = dataset[0:150]
    one_graphs = dataset[850:1000]
    total_graphs = 300

    accuracy = 0
    tolerance = 0.25
    
    accuracy += calculateAccuracy(zero_graphs, model, tolerance=tolerance, label=label)
    accuracy += calculateAccuracy(one_graphs, model, tolerance=tolerance, label=label)
    accuracy /= total_graphs

    print(f"Model can be approximated with probability: {accuracy}")

if __name__ == "__main__":
    approximateModel()