"""
Single movie graphs have a fixed prediction every time. We can precompute these values for all movies of sizes
1-136 (136 is the max-degree found in the dataset) and use them in our approximation.

In this file we simply compute the predictions of all single movies (with the size of the cast ranging from 1 to 136).
The list "movie" contains the following values:

1: 0.00
2: 0.27
3: 0.00
4: 1.00
5: 1.00
6: 1.00
7: 0.98
8: 0.84
...
53: 0.00
54: 0.03
55: 0.00
56: 0.00
...
132: 0.00
133: 0.37
134: 0.01
135: 0.21
136: 0.03
"""

from symmetricGraph import SymmetricGraph
from utility import loadModel

def printFullMovie():
    """Prints all values in movie (as seen above)."""

    count = 1
    for m in movie:
        print(f"{count}: {m:.2f}")
        count += 1

movie = []
model = loadModel()
movie.append(0) # add dummy value for cast=0
for cast in range(1, 136):
    graph = SymmetricGraph(num_movies=1, cast=cast, overlap=0) # Create single movie graph of size cast
    pred = model(graph.x, graph.edge_index).item()

    movie.append(pred)