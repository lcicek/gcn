from utility import loadDataset, loadModel
from modifyGraph import ModifyGraph
from calculateMovie import movie

def calculateApproximation(graph, label=False):
    """ Calculates approximation of model's prediction by using these features:
        1. Single movie graphs have fixed predictions (see calculateMovie.py). 
        2. Messiness (i.e. high number of movies and large node entanglement) is a predictor for 0-Graphs.
        3. Cleanness (i.e. lower number of movies and low node entanglement/ clear movie clusters) is a predictor for 1-Graphs.
        3. Graphs with a single central node can be roughly approximated by using the fixed single movie graph values.
        4. There is a tendency for 1-Graphs to have less nodes and 0-Graphs to have more nodes.
    """
    messiness, overlap = calculateMessiness(graph)

    if overlap == 0: # no overlapping actor => graph must contain just one movie => value is fixed
        if label:
            return 1 if movie[graph.num_actors] > 0.5 else 0
        else:
            return movie[graph.num_actors]
    elif messiness > 1:
        return 0
    elif messiness < 0.3:
        return 1
    elif overlap == 1:
        approx = 0
        for m in graph.movies:
            approx += movie[len(m)] # compute sum of fixed movie values

        approx /= graph.num_movies

        if label:
            if approx > 0.5:
                approx = 1
            else:
                approx = 0

        return approx
    else: # if all else fails, just use number of actors as a rough estimate
        return 0 if graph.num_actors > 50 else 1

def calculateMessiness(graph):
    """ If a graph is messy, it means it has a large number of movies with a lot of
        entangled nodes, i.e. the overlap of actors between movies is high.
        This function computes the overlap, factors in the amount of movies and size
        of cast and returns a messiness value of around 0 or 1 (even though it can be as
        high as 3).
    """

    # higher overlap => higher messiness
    overlap = findOverlap(graph)
    overlap_weight = overlap / graph.num_movies # min = 0, max = 1

    # increase messiness if number of movie and size of cast is high
    movie_weight = graph.num_movies / 136
    cast_weight = averageCast(graph) / 136

    messiness = overlap_weight + movie_weight + cast_weight

    return messiness, overlap

def averageCast(graph): 
    """ Calculates average size of cast across all movies in graph."""

    avg_cast = 0
    for m in graph.movies:
        avg_cast += len(m)

    avg_cast /= graph.num_movies

    return avg_cast

def findOverlap(graph):
    """Finds how many actors/nodes are in at least two movies."""

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

def datasetApproximationAccuracy(graphs):
    """ Goes over the entire dataset and approximates the label directly.
        The trained model is not involved at all but it achieves an accuracy
        of 0.71.
    """
    accuracy = 0
    for graph in graphs:
        ground_truth = graph.y

        graph = ModifyGraph(graph)
        approx = calculateApproximation(graph, label=True)

        accuracy += (ground_truth == approx).int()

    accuracy = accuracy.item()
    accuracy /= 1000

    print(f"Direct classification accuracy of approximation = {accuracy:0.2f}")

def calculateAccuracy(graphs, model, tolerance, label=False): 
    """ Returns the amount of successfully approximated predictions.
        An approximation is considered successful if the difference to the prediction
        is no greater than "tolerance".
    """

    accuracy = 0
    for i in range(0, 150):
        graph = graphs[i]

        if label:
            _, pred = model.evaluate(graph.x, graph.edge_index, batch=None) # returns the label 0 or 1
        else:
            pred = model(graph.x, graph.edge_index).item() # returns the prediction between 0 and 1

        graph = ModifyGraph(graph)
        approx = calculateApproximation(graph, label=label)

        correct_approx = 1 if abs(pred - approx) <= tolerance else 0
        accuracy += correct_approx
    
    return accuracy

def approximateModel(label=False):
    """ Calculates accuracy of the model approximation. An approximation is considered successful if
        the difference to the prediction is no greater than 0.25. The justification is:
        A prediction of 0.25 can be understood as a prediction of 0.5 with a halfway
        nudge towards 0, i.e. the model is not that sure about the prediction but thinks
        that it's probably a zero.
        Thus, we would consider an approximation between 0 and 0.5 successful.
        Similarly, if a prediction is 0.5 (the model is completely uncertain), then any
        approximation between 0.25 and 0.75 would be considered successful.

        With this definition, the approximation is able to achieve an accuracy of 0.77. If we don't
        approximate the model's prediction but, instead, the model's predicted label (which is easier and
        doesn't require setting a tolerance value), the accuracy of the approximation increases to 0.84.
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

    if label:
        print(f"Model-label approximation accuracy = {accuracy}")
    else:
        print(f"Model-prediction approximation accuracy = {accuracy}")

if __name__ == "__main__":
    approximateModel(label=False)
    approximateModel(label=True)
    datasetApproximationAccuracy(loadDataset())