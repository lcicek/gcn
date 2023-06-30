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

    def addActorsToMovie(self, add, actor, see_change=False, dir=None):
        assert (see_change and dir is not None) or (not see_change and dir is None)

        explain(self, see_change=see_change, dir=dir) # if see_change=false, just regular save; if true, then save starting graph

        for _ in range(add):
            self.addActorToMovie(actor)
            if see_change:
                explain(self, see_change=True, dir=dir)

    def addActorToMovie(self, actor): # finds the actor's movie and adds a new actor        
        new_actor = self.num_actors
        
        for movie in self.movies:
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

    def removeActors(self, actors, see_change=False, dir=None):
        assert (see_change and dir is not None) or (not see_change and dir is None)

        explain(self, see_change=see_change, dir=dir) # if see_change=false, just regular save; if true, then save starting graph

        # make sure list of actors is descending, since removing actors messes with the indices
        actors.sort(reverse=True)
        for actor in actors:
            self.removeActor(actor)
            if see_change:
                explain(self, see_change=True, dir=dir)

    def removeLastActor(self):
        self.removeActor(len(self.x) - 1)

    def removeActor(self, actor):
        # REMOVE FROM EDGES
        keep_edges = []
        connected = set() # all the nodes node is connected to
        for edge in self.edges:
            if actor not in edge:
                keep_edges.append(edge.unsqueeze(0))
            else:
                connected.add(edge[0].item() if edge[0] != actor else edge[1].item())

        self.edges = torch.cat(keep_edges, dim=0)

        # UPDATE INDICES GREATER THAN ACTOR
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

        # UPDATE ACTOR'S VALUES
        for i in range(len(self.movies)):
            movie = self.movies[i]
            updated_movie = set()
            for a in movie:
                if a > actor:
                    updated_movie.add(a-1)
                else:
                    updated_movie.add(a)

            self.movies[i] = updated_movie


        self.movies = [movie for movie in self.movies if len(movie) > 1] # make sure that there are no empty movies 

        # UPDATE VARIABLES
        self.updateVariables(-1)

torch.set_printoptions(threshold=100_000) # to be able to see full X during debugging
#graph = CustomGraph(num_movies=5, cast=5, overlap=2)
#explain(graph)

dataset = loadDataset()
graph = dataset[0] #sym_graph = symmetric_graph(num_movies=1, cast=30, overlap=0) 
custom_graph = CustomGraph(graph)

see_change = False
if see_change:
    dir = createNewTestDirectory()
    #custom_graph.addActorsToMovie(add=20, actor=7, see_change=True, dir=dir)
    custom_graph.removeActors([15, 13, 7, 10, 9, 5, 4, 0])
    custom_graph.removeActors([1, 3, 6])
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)
    custom_graph.addActorToMovie(0)

    explain(custom_graph)
    #custom_graph.removeActors([])
else:
    explain(custom_graph)