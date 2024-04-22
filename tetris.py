import pygame
import random
import itertools

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0 # position from 0 to width
    y = 0 # postion from 0 to height

    shapes = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]
    """ Tetriminoes description based on a 4*4 grid.\n
        |0 |1 |2 |3 |\n
        |4 |5 |6 |7 |\n
        |8 |9 |10|11|\n
        |12|13|14|15|\n
    """

    def __init__(self, x, y, rotation=None, shape=None):
        self.x = x
        self.y = y
        if shape is not None:
            self.shape = shape
        else:
            self.shape = random.randint(0, len(self.shapes) - 1)
        self.color = random.randint(1, len(colors) - 1)
        if rotation is not None:
            self.rotation = rotation%(len(self.shapes[self.shape]))
        else:
            self.rotation = 0

    def image(self):
        return self.shapes[self.shape][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shapes[self.shape])


class Tetris:
    def __init__(self, height, width):
        self.level = 2
        self.height = height
        self.width = width
        self.field = [ [0 for j in range(width)] for i in range(height) ]
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
        
        self.score = 0
        self.state = "start"

    def new_figure(self):
        self.figure = Figure(round(self.width/2)-2, 0)

    def check_intersection_figure(self, figure):
        intersection = False
        for pos in figure.image():
            i, j = divmod(pos, 4)
            if i + figure.y > self.height - 1 or \
                    j + figure.x > self.width - 1 or \
                    j + figure.x < 0 or \
                    self.field[i + figure.y][j + figure.x] > 0:
                intersection = True
        return intersection

    def intersects(self):
        """Check if the piece intersects with the field or bounds.

        Returns:
            bool: Is there an intersection
        """
        return self.check_intersection_figure(self.figure)
    
    def placeable(self):
        """        
        Returns the list of Figures that correspond to a valid figure placement.
        Valid placement here means a position where the figure is inbounds, does not intersect the grid and cannot be lowered.
        
        Returns: Figure list
        """
        rotations = len(self.figure.shapes[self.figure.shape]) # number of rotations for a current figure
        grid = [ [ [False for k in range(rotations)] for j in range(self.width+8)] for i in range(self.height) ]
                
        # grid[x,y,k] == Can the figure be placed at the x,y with rotation k
        # due to the shapes, column of the shape can be negative.
        
        dRow = [0, 1, 0, 0];
        dCol = [-1, 0, 1, 0];
        dTurn = [0, 0, 0, 1];
        
        st = []
        st.append(self.figure)
        
        valid = []
 
        # Dfs
        while len(st)>0:
            curr = st.pop(-1);
            row = curr.y;
            col = curr.x;
            turn = curr.rotation;
    
            if (self.check_intersection_figure(curr)):
                continue

            grid[row][col+4][turn] = True
            if (self.check_intersection_figure(
                Figure(y=row+1, x=col, rotation=turn, shape=self.figure.shape))):
                # The figure can't be lowered
                valid.append(curr)

            for drow, dcol, dturn in zip(dRow, dCol, dTurn):
                new_row = row + drow
                new_col = col + dcol
                new_turn = (turn + dturn)%rotations
                if not grid[new_row][new_col+4][new_turn]:
                    st.append(Figure(y=new_row, x=new_col, rotation=new_turn, shape=self.figure.shape))
                
        return valid
                
        
    def break_lines(self):
        """Delete complete lines and update the score."""
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
        self.score += lines ** 2

    def go_space(self):
        """Send the piece down until it intersects with the field."""
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        """Places the figure on the field, creates a new figure and updates the scores.
        """
        for pos in self.figure.image():
            i, j = divmod(pos, 4)
            self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def go_side(self, dx):
        """Move the figure by dx to the side"""
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


if __name__ == '__main__':
    # Initialize the game engine
    pygame.init()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

    size = (400, 500)
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Tetris")

    # Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()
    fps = 25
    game = Tetris(20, 10)
    counter = 0

    pressing_down = False

    while not done:
        if game.figure is None:
            game.new_figure()
        counter += 1
        if counter > 100000:
            counter = 0

        if counter % (fps // game.level // 2) == 0 or pressing_down:
            if game.state == "start":
                game.go_down()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.rotate()
                if event.key == pygame.K_DOWN:
                    pressing_down = True
                if event.key == pygame.K_LEFT:
                    game.go_side(-1)
                if event.key == pygame.K_RIGHT:
                    game.go_side(1)
                if event.key == pygame.K_SPACE:
                    game.go_space()
                if event.key == pygame.K_ESCAPE:
                    game.__init__(20, 10)

        if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    pressing_down = False

        screen.fill(WHITE)

        for i in range(game.height):
            for j in range(game.width):
                pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
                if game.field[i][j] > 0:
                    pygame.draw.rect(screen, colors[game.field[i][j]],
                                    [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

        if game.figure is not None:
            for pos in game.figure.image():
                i, j = divmod(pos, 4)
                pygame.draw.rect(screen, colors[game.figure.color],
                                    [game.x + game.zoom * (j + game.figure.x) + 1,
                                    game.y + game.zoom * (i + game.figure.y) + 1,
                                    game.zoom - 2, game.zoom - 2])
                        
        for fig in game.placeable()[:5]:
            for pos in fig.image():
                i, j = divmod(pos, 4)
                pygame.draw.rect(screen, (0,128,0),
                                    [game.x + game.zoom * (j + fig.x) + 1,
                                    game.y + game.zoom * (i + fig.y) + 1,
                                    game.zoom - 2, game.zoom - 2])

        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(game.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        screen.blit(text, [0, 0])
        if game.state == "gameover":
            screen.blit(text_game_over, [20, 200])
            screen.blit(text_game_over1, [25, 265])

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()