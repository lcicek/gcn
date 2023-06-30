from main import *

class CustomGraph:
    def __init__(self, graph):
        self.x = graph.x
        self.y = graph.y
        self.edge_index = graph.edge_index
        self.edges = torch.swapaxes(self.edge_index, 0, 1)
        self.num_actors = len(self.x)
        self.num_edges = graph.num_edges
        self.movies = self.get_movies()
        self.num_movies = len(self.movies)
        # self.num_movies = num_movies 


    # getmovies() returns clusters (i.e. "movies") in the graph.
    # note: it does not actually return the total amount of movies because it ignores edges across clusters 
    def get_movies(self):
        edge_index = torch.swapaxes(self.edge_index, 0, 1).tolist()

        movies = []
        for actor1 in range(len(self.x)):
            movie = set()
            movie.add(actor1)

            for actor2 in range(len(self.x)):
                if actor2 not in movie:
                    add_actor2 = True
                    for actor in movie:
                        if [actor, actor2] not in edge_index: # only add actor2 if they have mutual connection to every other actor in movie
                            add_actor2 = False
                    
                    if add_actor2:
                        movie.add(actor2)

            if movie not in movies:
                movies.append(movie)

        return movies

    # it's easier to remove nodes in self.edges, so update edge_index through edges
    def updateEdgeIndex(self):
        self.edge_index = torch.swapaxes(self.edges, 0, 1)

    def createDegreeTensor(self, degree):
        degree_tensor = torch.zeros(136)
        degree_tensor[degree] = 1

        return degree_tensor.unsqueeze(0)

    def updateVariables(self, val):
        self.num_actors += val
        self.num_edges = len(self.edges)
        self.num_movies = len(self.movies)

    def updateDegree(self, node, val):
        degree = self.x[node].nonzero().item()
        self.x[node][degree+val] = 1
        self.x[node][degree] = 0

    def addActorToMovie(self, actor): # finds the actor's movie and adds a new actor        
        new_actor = self.num_actors
        
        for i, movie in enumerate(self.movies):
            if actor in movie:
                movie.add(new_actor)

                new_edges = []
                for a in movie:
                    if new_actor != a:
                        new_edges.append([new_actor, a])
                        new_edges.append([a, new_actor])
                        self.updateDegree(a, 1)
                
                self.x = torch.cat((self.x, self.createDegreeTensor(degree=len(movie)-1)), dim=0)
                self.edges = torch.cat((self.edges, torch.tensor(new_edges)), dim=0)
                self.updateEdgeIndex()
                self.updateVariables(1)

                break

    def removeLastActor(self):
        self.removeActor(len(self.x) - 1)

    def removeActor(self, actor):
        # REMOVE FROM EDGES
        keep_edges = []
        connected = set() # all the nodes node is connected to
        for edge in self.edges:
            if actor not in edge: # test if true
                keep_edges.append(edge.unsqueeze(0))
            else:
                connected.add(edge[0].item() if edge[0] != actor else edge[1].item())

        self.edges = torch.cat(keep_edges, dim=0)

        # UPDATE INDICES GREATER THAN NODE
        self.edges[self.edges > actor] -= 1

        # UPDATE X
        for c_node in connected:
            self.updateDegree(c_node, -1)

        keep_actors = list(range(0, len(self.x)))
        keep_actors.remove(actor)
        self.x = self.x[keep_actors]

        # UPDATE EDGE INDEX
        self.updateEdgeIndex()

        # REMOVE FROM MOVIES
        for movie in self.movies:
            if actor in movie:
                movie.remove(actor)
            
        self.movies = [movie for movie in self.movies if movie != {}] # make sure that there are no empty movies 

        # UPDATE VARIABLES
        self.updateVariables(-1)


torch.set_printoptions(threshold=100_000) # to be able to see full X during debugging
#graph = CustomGraph(num_movies=5, cast=5, overlap=2)
#explain(graph)

dataset = loadDataset()
graph = dataset[0]
# explain(graph)
custom_graph = CustomGraph(graph)
custom_graph.addActorToMovie(7)
explain(custom_graph)