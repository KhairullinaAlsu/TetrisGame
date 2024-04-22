import numpy as np
import random
from tetris import Tetris, Figure

class AntColony:
    def __init__(self, height, width, shape_seq, n_ants, n_iterations, decay, alpha=1, beta=1):
        """
        Initialize the Ant Colony Optimization algorithm.

        Parameters:
        - distances: A 2D numpy array where distances[i][j] represents the distance between cities i and j.
        - n_ants: The number of ants used in the algorithm.
        - n_iterations: The number of iterations to run the algorithm.
        - decay: The rate at which pheromone trails evaporate over time.
        - alpha: The influence of pheromone strength on the probability of selecting the next city.
        - beta: The influence of heuristic value (inverse of distance) on the probability of selecting the next city.
        """
        self.height = height
        self.width = width
        self.shape_seq = shape_seq
        self.pheromones = np.ones((len(shape_seq), height, width, 4))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        """
        Run the ACO algorithm to find the shortest path for the TSP.

        Returns:
        - shortest_path: The shortest path found by the algorithm.
        - shortest_distance: The distance of the shortest path.
        """
        lowest_path = None
        lowest_score = float('inf')
        for i in range(self.n_iterations):
            paths, scores = self._construct_solutions()
            # YOUR CODE HERE
            # Update pheromones based on paths and distances
            self._update_pheromones(paths, scores)
            best_idx = np.argmin(scores)
            if scores[best_idx] < lowest_score:
                lowest_score = scores[best_idx]
                lowest_path = paths[best_idx]
            self.pheromones *= self.decay
        return lowest_path, lowest_score


    def _construct_solutions(self):
        """
        Construct solutions for each ant in the colony.

        Returns:
        - paths: A list of paths, one for each ant.
        - scores: A list of scores, one for each path of an ant.
        """
        paths = []
        scores = []
        # Construct path for each ant
        for ant in range(self.n_ants):
            path, score = self._construct_path()
            paths.append(path)
            scores.append(score)

        return paths, scores

    def _construct_path(self):
        """
        Construct a path for an ant based on pheromone trails and distances.

        Returns:
        - path: The constructed path.
        - score: Score for the path of an ant.
        """
        path = []
        game = Tetris(self.height, self.width)
        i = 0
        
        for i, fig in enumerate(self.shape_seq):
            game.figure = fig
            ways = game.placeable()
            
            next_state = self._select_next(i, ways)
            path.append(next_state)
            game.figure = next_state
            
            game.freeze()
            if game.state == "gameover":
                break
        
        if game.state == "start":
            print("it's an actual miracle")
            
        score = self._score_function(i, game.score)



            
        """path = [0]  # Start from city 0
        remaining = set(range(1, self.distances.shape[0]))
        while remaining:
            current_node = path[-1]
            # Select the next city for the current path
            next_node = self._select_next(current_node, remaining)
            remaining.remove(next_node)
            path.append(next_node)"""

        return path, score
        
    def _select_next(self, current, choices):
        """
        Select the path to the next node based on pheromone trails and heuristic.

        Parameters:
        - current: The current node.
        - choices: The set of figures placements that can be selected.

        Returns:
        - The next path to choose, randomly choosing with respect to the probabilities (probs)
        """
        # Calculate the probability of moving to each city in choices
        probabilities = [pow(self.pheromones[current][ch.y][ch.x][ch.rotation], self.alpha) * \
                        pow(self._distance_heuristic(ch), self.beta) for ch in choices]
        prob_sum = np.sum(probabilities)
        next_node = np.random.choice(choices, p=probabilities/prob_sum)

        return next_node

    def _distance_heuristic(self, figure):
        """
        Calculate the heuristic value for a figure placement

        Parameters:
        - figure: The placement of a figure

        Returns:
        - The heuristic value for moving from city i to city j.
        """
        # Define the heuristic value based on the distance

        return figure.y/self.height

    def _score_function(self, n_pieces, n_lines):
        """Calculate the score function

        Args:
            n_pieces (_type_): number of places pieces
            n_lines (_type_): number of cleared lines
        """
        return n_pieces + n_lines*10

    def _update_pheromones(self, paths, scores):
        """
        Update the pheromones on the paths based on the scores of the paths.

        Parameters:
        - paths: The paths taken by the ants.
        - scores: The scores of these paths.
        """
        self.pheromones *= (1-self.decay)
        
        for path, score in zip(paths, scores):
            for i, node in enumerate(path):
                self.pheromones[i][node.y][node.x][node.rotation] += score
             
        
        """
        for i in range(self.pheromones.shape[0]):
            for y in range(self.pheromones.shape[1]):
                for x in range(self.pheromones.shape[2]):
                    for r in range(self.pheromones.shape[3]):
                        for path, distance in zip(paths, scores):
                            # YOUR CODE HERE
                            # Check if city j follows city i in the path and update pheromones (For both directions)
                            if path.index(i)<len(path)-1 and path[path.index(i)+1]==j:
                                pheromones_sum[i][j] += 1/distance

        for i in range(self.distances.shape[0]):
            for j in range(self.distances.shape[1]):
                new_pheromones[i][j] = self.pheromones[i][j]*pheromones_sum[i][j]*(1-self.decay)
        """
                
        
if __name__ == '__main__':
    
    width = 7
    height = 12
    decay = 0.5
    
    random.seed(42)
    shape_seq = []
    for i in range(10000):
        shape_seq.append(Figure(round(width/2)-2, 0,
                                shape=random.randint(0, random.randint(0, len(Figure.shapes) - 1))))
        
    aco = AntColony(height, width, shape_seq, 100, 1000, decay)
    path, score = aco.run()
    print(score)
    print(path)
    
    
    