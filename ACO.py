import numpy as np
import random
import copy
from tetris import Tetris, Figure

class AntColony:
    def __init__(self, height, width, shape_seq, n_ants, n_iterations, decay, alpha=2, beta=1):
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
        shape = [5 for _ in range(width-1)] 
        shape.extend([height, width, 4, 7])
        # altitude difference + height + width + rotation + shape
        self.pheromones = np.ones(shape)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def __altitude(self, game):
        """Finds the differences betweeen highest occupied spaces in neighboring cells

        Args:
            game (Tetris)
            
        Returns:
            ndarray[game.width]: altitude difference
        """
        altitudes = []
        for j in range(game.width):
            i = 0
            while i <= game.height-1 and game.field[i][j] == 0:
                i += 1
            altitudes.append(i)
        
        ans = []
        for i in range(len(altitudes)-1):
            ans.append(altitudes[i] - altitudes[i+1])
        ans = np.clip(ans, a_min=-2, a_max=2)
        # Limit the values to the range [-2; 2]
        # |0| - the altitudes are the same
        # |1| - the altitudes differ by 1
        # |2| - the altitudes differ by a value equal or greater than 2
        
        return ans
                
            
    def run(self):
        """
        Run the ACO algorithm to find the max score for the game of Tetris.

        Returns:
        - max_path: The shortest path found by the algorithm.
        - max_score: The score of the shortest path.
        """
        max_path = None
        self.max_score = 0
        for i in range(self.n_iterations):
            paths, scores = self._construct_solutions()
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.max_score:
                self.max_score = scores[best_idx]
                max_path = paths[best_idx]
                
            #self._update_pheromones(paths, scores)
            if i%5==0:
                print(f'Current iteration: {i}, Current max score: {self.max_score}')
        return max_path, self.max_score


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
            if game.intersects():
                game.state = "gameover"
                break
            
            ways = game.placeable()
            curr = list(self.__altitude(game))
            
            next_state = self._select_next(curr, ways, game)
            path.append((curr, next_state))
            game.figure = next_state
            
            game.freeze()
            if game.state == "gameover":
                break
        
        if game.state == "start":
            # Game ended with every figure placed.
            print("it's an actual miracle")
            
        score = self._score_function(i, game.score)

        return path, score
        
    def _select_next(self, current: list[int], choices: list[Figure], game:Tetris):
        """
        Select the path to the next node based on pheromone trails and heuristic.

        Parameters:
        - current: The current node.
        - choices: The set of figures placements that can be selected.

        Returns:
        - The next path to choose, randomly choosing with respect to the probabilities (probs)
        """
        # Calculate the probability of moving to each city in choices
        #print(tuple(current + [choices[0].y, choices[0].x, choices[0].rotation, choices[0].shape]))
        probabilities = [pow(self.pheromones[tuple(current + [ch.y, ch.x, ch.rotation, ch.shape])], self.alpha) * \
                        pow(self._distance_heuristic(ch), self.beta) for i, ch in enumerate(choices)]
        prob_sum = np.sum(probabilities)
        if prob_sum == 0:
            #print("zero error")
            next_node = choices[0]
        else:    
            next_node = np.random.choice(choices, p=probabilities/prob_sum)
        #next_node = choices[np.argmax(fit)]
        #next_node = sorted(zip(fit, choices), key=lambda x: x[0], reverse=True)[0][1]

        return next_node
    
    def _distance_heuristic(self, figure: Figure):
        """
        Calculate the heuristic value for a figure placement

        Parameters:
        - figure: The placement of a figure

        Returns:
        - The heuristic value for moving from current state with the new figure.
        """
        # Define the heuristic value based on the distance

        return (figure.y + max(figure.image())//4)/self.height

    def _score_function(self, n_pieces, n_lines):
        """Calculate the score function

        Args:
            n_pieces (_type_): number of places pieces
            n_lines (_type_): number of cleared lines
        """
        return n_lines

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
                curr = node[0]
                fig = node[1]
                curr.extend([fig.y, fig.x, fig.rotation, fig.shape])
                self.pheromones[tuple(curr)] += \
                    (score/self.max_score)
                
        
if __name__ == '__main__':
    
    width = 7
    height = 12
    decay = 0.5
    
    random.seed(42)
    shape_seq = []
    for i in range(10000):
        shape_seq.append(Figure(round(width/2)-2, 0,
                                shape=random.randint(0, random.randint(0, len(Figure.shapes) - 1))))
    
    best_path, best_score = [0,0]
    
    aco = AntColony(height, width, shape_seq, 100, 75, decay, alpha=1, beta=4)
    path, score = aco.run()
    
    if score > best_score:
        best_path, best_score = path, score
        
    game = Tetris(height, width)
    for node in best_path:
        figure = node[1]
        game.figure = figure
        game.freeze()
        for i in range(height):
            for j in range(width):
                if game.field[i][j] == 0:
                    print('.', end='')
                else:
                    print('O', end='')
            print()
        print('---------------')
    print(f'Eval: {best_score}, lines: {game.score}')
    
    