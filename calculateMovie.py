from symmetricGraph import SymmetricGraph as sg
from utility import loadModel

movie = []
model = loadModel()
for cast in range(0, 2):
    movie.append(-1) # graphs of size 0 and 1 wouldn't make sense

for cast in range(2, 31):
    graph = sg(num_movies=1, cast=cast, overlap=0)
    pred = model(graph.x, graph.edge_index).item()

    movie.append(pred)