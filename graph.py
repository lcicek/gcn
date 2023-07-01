import torch
from utility import explain

class Graph:
    """ The Graph class can be used to remove actors or edges between actors from an IMDB-BINARY graph.
        The previously implicit movies in the graph can also be accessed directly and actors can be added to
        movies.

        Instantiation is simple and all functions and attributes can be accessed after initializing the Graph.
        E.g.:
            graph = dataset[0] # the first graph from IMDB-Binary
            new_graph = Graph(graph)
            new_graph.removeEdge(1, 2)
    """

    def __init__(self, graph):
        self.x = graph.x
        self.y = graph.y
        self.edge_index = graph.edge_index
        self.edges = torch.swapaxes(self.edge_index, 0, 1)
        self.num_actors = len(self.x)
        self.num_edges = graph.num_edges
        self.movies = self.get_movies()
        self.num_movies = len(self.movies)

    def get_movies(self):
        """ Creates and returns all movies that are implicit in the graph."""

        edges = self.edges.tolist()
        movies = []
        for actor in range(len(self.x)):
            movie = set()
            movie.add(actor) # Given an actor of a movie...

            for new_actor in range(len(self.x)):
                if new_actor not in movie: # ...and a new actor who hasn't been added to the movie yet...
                    add_new_actor = True
                    for actor in movie: # ...check if all actors in the movie...
                        if [actor, new_actor] not in edges: # ...have an edge with the new_actor...
                            add_new_actor = False # ...if not: don't add the actor to the movie
                    
                    if add_new_actor: # ...and if so: add the actor
                        movie.add(new_actor)

            if movie not in movies: # avoid duplicates
                movies.append(movie)

        return movies

    def updateEdgeIndex(self):
        self.edge_index = torch.swapaxes(self.edges, 0, 1)

    def createDegreeTensor(self, degree): # degree tensor is the equivalent to an entry in graph.x
        degree_tensor = torch.zeros(136)
        degree_tensor[degree] = 1

        return degree_tensor.unsqueeze(0)

    def updateVariables(self, add_actors):
        self.num_actors += add_actors
        self.num_edges = len(self.edges)
        self.num_movies = len(self.movies)

    def updateDegree(self, node, val):
        degree = self.x[node].nonzero().item()
        self.x[node][degree+val] = 1
        self.x[node][degree] = 0
    
    def removeEdge(self, actor1, actor2):
        """ Removes the edge from actor1 to actor2 (and vice versa) and updates the graph:
            Edge is removed from self.edges, self.edge_index is updated, self.num_edges is updated.
            Degrees of actor1 and actor2 are decremented.
            Movies are updated (for more information, see comment below).
        """

        # UPDATE EDGES
        keep_edges = []
        for edge in self.edges:
            if actor1 not in edge or actor2 not in edge:
                keep_edges.append(edge.unsqueeze(0))

        self.edges = torch.cat(keep_edges, dim=0)

        # UPDATE MOVIES:
        """ A removed edge means that a movie where actor1 and actor 2 appear, is now divided into two
            separate movies. The remaining cast has to be added to both of those new movies.
        """ 
        remove_movies = [movie for movie in self.movies if actor1 in movie and actor2 in movie]
        
        for movie in remove_movies:
            movie1 = set([a for a in movie if a != actor1])
            movie2 = set([a for a in movie if a != actor2])

            # check for potential duplicates: e.g. movies {0, 1, 3}, {0, 1, 2}
            # if edge [1, 2] is removed, movies {0, 1}, {0, 3}, {0, 2}, {0, 3} would contain a duplicate
            if movie1 not in self.movies:
                self.movies.append(movie1)
            
            if movie2 not in self.movies:
                self.movies.append(movie2)

            self.movies.remove(movie)

        # UPDATE  INFO
        self.updateEdgeIndex()
        self.updateVariables(add_actors=0)
        self.updateDegree(node=actor1, val=-1)
        self.updateDegree(node=actor2, val=-1)

    def addActorsToMovie(self, add, actor, visualize=True, see_change=False, dir=None):
        """ Calls the function addActorToMovie() (see below) "add"-many times.
            If visualize is true, graph is visualized in images/graph.png.
            If see_change is true, every change to the graph is visualized in an image in directory "dir".
        """

        assert (see_change and dir is not None) or (not see_change and dir is None)

        if visualize:
            explain(self, see_change=see_change, dir=dir) # if see_change=false, just regular save; if true, then save starting graph

        for _ in range(add):
            self.addActorToMovie(actor)
            if see_change:
                explain(self, see_change=True, dir=dir)

    def addActorToMovie(self, actor):
        """ Given an actor, this function finds the movie the actor is in and adds a new actor to it.
            If the actor is in several movies, the function uses the first movie the actor is in.
        """     
        new_actor = self.num_actors # the new actor will have index = self.num_actors
        for movie in self.movies:
            if actor in movie:
                new_edges = []
                for a in movie: # for every actor in the movie...
                    new_edges.append([a, new_actor]) # ... add an edge to the new actor 
                    new_edges.append([new_actor, a]) # ... and vice versa
                    self.updateDegree(a, 1) # ...and increment the actor's degree
                
                movie.add(new_actor)
                
                # UPDATE:
                self.x = torch.cat((self.x, self.createDegreeTensor(degree=len(movie)-1)), dim=0)
                self.edges = torch.cat((self.edges, torch.tensor(new_edges)), dim=0)
                self.updateEdgeIndex()
                self.updateVariables(add_actors=1)

                break

    def removeActors(self, actors, visualize=True, see_change=False, dir=None):
        """For each actor in actors, calls removeActor() (see below)."""
        assert (see_change and dir is not None) or (not see_change and dir is None)

        if visualize:
            explain(self, see_change=see_change, dir=dir) # if see_change=false, just regular save; if true, then save starting graph

        actors.sort(reverse=True) # make sure list of actors is descending, since removing actors messes with the indices
        for actor in actors:
            self.removeActor(actor)
            if see_change:
                explain(self, see_change=True, dir=dir)

    def removeLastActor(self):
        self.removeActor(len(self.x) - 1)

    def removeActor(self, actor):
        """Removes an actor from the graph."""

        # REMOVE FROM EDGES
        keep_edges = []
        connected = set() # save nodes connected to actor to decrement their degree later
        for edge in self.edges:
            if actor not in edge:
                keep_edges.append(edge.unsqueeze(0))
            else:
                connected.add(edge[0].item() if edge[0] != actor else edge[1].item())

        self.edges = torch.cat(keep_edges, dim=0)

        # UPDATE EDGE INDICES GREATER THAN ACTOR
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

        # UPDATE ACTOR'S INDICES/VALUES GREATER THAN REMOVED ACTOR
        for i in range(len(self.movies)):
            movie = self.movies[i]
            updated_movie = set()
            for a in movie:
                if a > actor:
                    updated_movie.add(a-1)
                else:
                    updated_movie.add(a)

            self.movies[i] = updated_movie

        self.movies = [movie for movie in self.movies if len(movie) > 1] # make sure that there are no empty or one actor movies 

        # UPDATE VARIABLES
        self.updateVariables(add_actors=-1)