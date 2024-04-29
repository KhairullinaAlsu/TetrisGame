import numpy as np
import random
import copy
from tetris import Tetris, Figure

class TetrisHelper:
    def __init__(self, game: Tetris):
        self.game = game
        self.field = game.field  # reference
        self.height = game.height
        self.width = game.width
        self.weighted_field = self.height * self.width * (self.height) * (self.height-1)
    
    def _altitudes(self):
        """Finds the highest directly accessible taken tile.

        Returns:
            List[Int]: list of heights
        """
        altitude = np.zeros(self.width)
        for j in range(self.width):
            for i in range(self.height):
                if self.field[i][j] > 0:
                    altitude[j] = i
                    break
        
        return altitude
    
    def fitness(self):
        """Calculate the fitness of the current state.

        Returns:
            float: fitness of the current state (bigger == better).
        """
        self.altitude = np.zeros(self.width)
        fitness = self.weighted_field # Initial weighted field
        for j in range(self.width):
            open = True # detecting if the column is directly accesible
            for i in range(self.height):
                if self.field[i][j] > 0:
                    if open: 
                        open = False
                        self.altitude[j] = i
                    fitness -= (self.height - i)
                elif not open:
                    fitness -= i
        
        return fitness
    
    def break_lines(self):
        """Delete complete lines and return the score."""
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        return lines
    
    def fitness_next(self, figure: Figure, debug = False):
        """Places the figure on the field and calculates the fitness of the resulting grid.
        Reverts the changes afterwards.
        
        Returns:
            int: fitness of the next state (bigger == better) and the lines broken.
        """
        if debug:
            for k in range(height):
                for j in range(width):
                    if self.field[k][j] == 0:
                        print('.', end='')
                    else:
                        print('O', end='')
                print()
            print('<->')
        
        for pos in figure.image():
            i, j = divmod(pos, 4)
            self.field[i + figure.y][j + figure.x] = figure.color
            
        score = self.break_lines()
        fit = self.fitness()
        
        if debug:
            for k in range(height):
                for j in range(width):
                    if self.field[k][j] == 0:
                        print('.', end='')
                    else:
                        print('O', end='')
                print()
            print(f'----- {int(fit + self.width*score)} -----')
            input()
        
        for pos in figure.image():
            i, j = divmod(pos, 4)
            self.field[i + figure.y][j + figure.x] = 0
        return int(fit + self.width*score)

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

    def _altitude(self, state: TetrisHelper):
        """Finds the differences betweeen highest occupied spaces in neighboring cells

        Args:
            game (Tetris)
            
        Returns:
            ndarray[game.width]: altitude difference
        """
        altitudes = state._altitudes()
        
        ans = []
        for i in range(len(altitudes)-1):
            ans.append(int(altitudes[i] - altitudes[i+1]))
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
            # YOUR CODE HERE
            # Update pheromones based on paths and distances
            best_idx = np.argmax(scores)
            if scores[best_idx] >= self.max_score:
                self.max_score = scores[best_idx]
                max_path = paths[best_idx]
                
            self._update_pheromones(paths, scores)
            if i%5==0:
                print(f'Current iteration: {i}, Current max score: {self.max_score}')
                #input()
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
        helper = TetrisHelper(game)
        i = 0
        
        for i, fig in enumerate(self.shape_seq):
            game.figure = fig
            if game.intersects():
                game.state = "gameover"
                break
            
            ways = game.placeable()
            curr = list(self._altitude(helper))
            
            next_state = self._select_next(helper, curr, ways)
            path.append((curr, next_state, helper.fitness_next(next_state)))
            game.figure = next_state
            
            game.freeze()
            if game.state == "gameover":
                break
        
        if game.state == "start":
            # Game ended with every figure placed.
            print("it's an actual miracle")
            
        score = self._score_function(i, game.score)

        return path, score
        
    def _select_next(self, helper: TetrisHelper, current: list[int], choices: list[Figure]):
        """
        Select the path to the next node based on pheromone trails and heuristic.
        Additionally return the number of lines broken as a result.

        Parameters:
        - helper: Helping class to ease the calculation of values.
        - current: The current node.
        - choices: The set of figures placements that can be selected.

        Returns:
        - The next figure to choose, randomly choosing with respect to the probabilities (probs)
        """
        # Calculate the probability of moving to each city in choices
        #print(tuple(current + [choices[0].y, choices[0].x, choices[0].rotation, choices[0].shape]))
        fitnesses = [self._distance_heuristic(helper, ch) for ch in choices]
        sort = sorted(enumerate(fitnesses), key=lambda x: x[1], reverse=True)
        base = 2
        rate = 0.6
        hrst = np.zeros(len(sort))
        for i, _ in sort:
            base *= rate
            hrst[i] = base
        #print(hrst)
        probabilities = [pow(self.pheromones[tuple(current + [ch.y, ch.x, ch.rotation, ch.shape])], self.alpha) * \
                        pow(hrst[i], self.beta) for i, ch in enumerate(choices)]
        prob_sum = np.sum(probabilities)
        #print(probabilities/prob_sum)
        if prob_sum == 0:
            print("zero error")
            next_node = np.random.randint(low=0, high=len(choices))
        else:    
            next_node = np.random.choice(choices, p=probabilities/prob_sum)
        #helper.fitness_next(next_node, debug=True)

        return next_node
    
    def _distance_heuristic(self, helper: TetrisHelper, figure: Figure):
        """
        Calculate the heuristic value for a figure placement

        Parameters:
        - helper: Helping class to ease the calculation of fitness.
        - figure: The placement of a figure

        Returns:
        - The heuristic value for moving from current state with the new figure.
        """
        # Define the heuristic value based on the distance
        fit = helper.fitness_next(figure)
        hrst = (fit*2) / helper.weighted_field
        hrst = np.exp(np.exp(hrst)) # Amplify the difference between closer fitness values
        #if np.random.uniform() < 0.005: print(f'random check: {hrst}')
        return hrst

    def _score_function(self, n_pieces, n_lines):
        """Calculate the score function

        Args:
            n_pieces (_type_): number of places pieces
            n_lines (_type_): number of cleared lines
        """
        #return n_pieces + 5*n_lines
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
                hrst = node[2]
                curr.extend([fig.y, fig.x, fig.rotation, fig.shape])
                self.pheromones[tuple(curr)] += hrst
                
        
if __name__ == '__main__':
    
    width = 8
    height = 16
    decay = 0.5
    
    random.seed(42)
    shape_seq = []
    for i in range(10000):
        shape_seq.append(Figure(round(width/2)-2, 0,
                                shape=random.randint(0, random.randint(0, len(Figure.shapes) - 1))))
    
    best_path, best_score, best_alpha, best_beta = [0,0,0,0]
    
    #for alpha in [1, 2]:
     #   for beta in [4, 2, 1]:
      #      print(f'a: {alpha}, b: {beta}')
    aco = AntColony(height, width, shape_seq, 100, 75, decay, alpha=1, beta=4)
    path, score = aco.run()
            
    if score > best_score:
        best_path, best_score = path, score
        #best_alpha, best_beta = alpha, beta
    
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
    #print(f'Best alpha and beta: {best_alpha}, {best_beta}')
    
    