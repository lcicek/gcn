"""
Single movie graphs have a fixed prediction every time. We can precompute these values for all movies of sizes
1-136 (136 is the max-degree found in the dataset) and use them in our approximation.

In this file we simply compute the predictions of all single movies (with the size of the cast ranging from 1 to 136).
The list "movie" contains the following values:

0: 0.00
1: 0.27
2: 0.00
3: 1.00
4: 1.00
5: 1.00
6: 0.98
7: 0.84
8: 0.49
...
53: 0.03
54: 0.00
55: 0.00
56: 0.00
...
130: 0.07
131: 0.00
132: 0.37
133: 0.01
134: 0.21
135: 0.03
"""

from symmetricGraph import SymmetricGraph
from utility import loadModel

def printFullMovie():
    """Prints all values in movie (as seen above)."""

    count = 0
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