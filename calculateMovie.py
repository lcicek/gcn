# calculateMovie.py calculates the fixed values of differently sized movies


from symmetricGraph import SymmetricGraph as sg
from utility import loadModel

movie = []
model = loadModel()
movie.append(0) # value = -1 for cast = 0
for cast in range(1, 136): # 136 is max degree, so 135 is max cast
    graph = sg(num_movies=1, cast=cast, overlap=0)
    pred = model(graph.x, graph.edge_index).item()

    movie.append(pred)