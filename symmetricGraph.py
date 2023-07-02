from main import *

class SymmetricGraph:
    """ SymmetricGraph is a graph that can be divided into two equal parts.
        A SymmetricGraph's movies all have the same cast size and can have zero to cast size
        overlapping actors.
        
        Examples of symmetric graphs:
        1. Single movie graphs.
        2. Single central node graph.

        Note: When instantiating a SymmetricGraph, the amount of "overlap" actors (i.e. central
        nodes) is included in cast. E.g.:
        SymmetricGraph(num_movies=2, cast=3, overlap=1) would create a graph with one central
        node connected to two other nodes per movie.
    """


    def __init__(self, num_movies, cast, overlap): # cast includes overlap actors!
        assert num_movies * cast < 136 # don't exceed degree limit found in IMDB-BINARY
        
        self.num_movies = num_movies 
        self.cast = cast
        self.overlap = overlap
        self.num_actors = num_movies * cast - num_movies * overlap + overlap
        self.num_edges = num_movies * (cast - overlap) * (cast-1) + overlap * (self.num_actors - 1)
        self.movies = []
        self.edges = []
        self.y = None

        self.initMovies()
        self.initX()
        self.initEdges()
        self.initEdgeIndex()
        
    def initMovies(self):
        unique_per_movie = self.cast - self.overlap

        for i in range(self.num_movies):
            movies = []

            for id in range(self.overlap): # overlap actors are in all movies and have unique "id" between 0 and overlap
                movies.append(id)

            for k in range(unique_per_movie):
                id = i * unique_per_movie + k # non-overlap nodes have unique ids
                id += self.overlap # make sure to reserve the first overlap amount nodes for the overlap nodes
                movies.append(id)

            self.movies.append(movies)

        return movies

    def initX(self): # X contains degree of node (how many actors one actor is connected to)
        self.x = torch.zeros((self.num_actors, 136))

        for i in range(self.num_actors):
            if i < self.overlap:
                degree = self.num_actors - 1 # connection to all minus themselves
            else:
                degree = self.cast - 1 # connection to cast minus themselves
            
            self.x[i][degree] = 1
    
    def initEdges(self):
        for i in range(self.overlap): # add overlap edges first
            for j in range(i+1, self.num_actors):
                self.edges.append(torch.tensor([i, j]))
                self.edges.append(torch.tensor([j, i]))
        
        for i in range(self.num_movies):
            for j in range(self.overlap, self.cast):
                for k in range(j+1, self.cast):
                    s = self.movies[i][j] # start node
                    e = self.movies[i][k] # end node

                    self.edges.append(torch.tensor([s, e]))
                    self.edges.append(torch.tensor([e, s]))

        assert len(self.edges) == self.num_edges

    def initEdgeIndex(self):
        self.edge_index = torch.zeros((self.num_edges, 2), dtype=torch.int64)

        for i, edge in enumerate(self.edges):
            self.edge_index[i] = edge

        self.edge_index = torch.swapaxes(self.edge_index, 0, 1)