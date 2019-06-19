import pygame
import sys
import time
from random import randint
from pygame.locals import *


class SnakeGame:

    DIRS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    def __init__(self):

        # Board info
        self.ARRAY_SIZE = 50    # size of board
        self.DIRECTIONS = {
            "UP": (0, 1),
            "RIGHT": (1, 0),
            "DOWN": (0, -1),
            "LEFT": (-1, 0),
        }

        # Game info
        self.snake = [(0, 2), (0, 1), (0, 0)]
        self.score = 0
        self.food = None
        self.direction = None
        self.done = False

        # Display variables
        self.snake_game = None
        self.food_piece = None
        self.snake_body = None

        # Place food is needed last
        self.place_food()

    def place_food(self):
        # Will loop until it finds a random place where the snake isn't
        while True:
            x = randint(0, self.ARRAY_SIZE - 1)
            y = randint(0, self.ARRAY_SIZE - 1)
            if (x, y) not in self.snake:
                self.food = x, y
                return

    def move(self, new_direction):
        old_head = self.snake[0]
        movement = self.DIRECTIONS[self.DIRS[new_direction]]
        new_head = (old_head[0] + movement[0], old_head[1] + movement[1])

        # Check the head is not against a wall or turned on itself
        if (
                new_head[0] < 0 or                  # hasn't hit left side
                new_head[1] < 0 or                  # hasn't hit the bottom
                new_head[0] >= self.ARRAY_SIZE or   # hasn't hit the right side
                new_head[1] >= self.ARRAY_SIZE or   # hasn't hit the top
                new_head in self.snake
        ):
            self.done = True

        if self.food_eaten(new_head):
            self.score += 1
            self.place_food()
        else:
            del self.snake[-1]

        self.snake.insert(0, new_head)
        return self.get_game_info()

    # Specific move version for the visualised neural net
    def nn_move(self, new_direction):
        time.sleep(.050)

        self.move(new_direction)
        self.snake_game.fill((255, 255, 255))

        for bodyPart in self.snake:
            self.snake_game.blit(self.snake_body, (bodyPart[0] * 10, (self.ARRAY_SIZE - bodyPart[1] - 1) * 10))

        self.snake_game.blit(self.food_piece, (self.food[0] * 10, (self.ARRAY_SIZE - self.food[1] - 1) * 10))
        pygame.display.flip()

        if self.done:
            pygame.quit()

        return self.get_game_info()

    def food_eaten(self, snake_head):
        return snake_head == self.food

    def change_direction(self, new_direction):
        self.direction = new_direction

    def get_game_info(self):
        return self.done, self.score, self.snake, self.food

    def start(self):
        self.direction = 0

        pygame.init()
        self.snake_game = pygame.display.set_mode((self.ARRAY_SIZE * 10, self.ARRAY_SIZE * 10))
        pygame.display.set_caption('Snake Game')

        self.food_piece = pygame.Surface((10, 10))
        self.food_piece.fill((255, 0, 0))
        self.snake_body = pygame.Surface((10, 10))
        self.snake_body.fill((0, 0, 0))

        pygame.time.set_timer(1, 100)

        return self.get_game_info()

    # This is to run the game playable to the user
    def run(self):
        self.start()

        while not self.done:
            e = pygame.event.wait()
            if e.type == QUIT:
                self.done = True

            elif e.type == KEYDOWN:
                key = pygame.key.get_pressed()

                if key[pygame.K_UP]:
                    self.change_direction(0)
                elif key[pygame.K_RIGHT]:
                    self.change_direction(1)
                elif key[pygame.K_DOWN]:
                    self.change_direction(2)
                elif key[pygame.K_LEFT]:
                    self.change_direction(3)

            self.move(self.direction)

            self.snake_game.fill((255, 255, 255))
            for bodyPart in self.snake:
                self.snake_game.blit(self.snake_body, (bodyPart[0] * 10, (self.ARRAY_SIZE - bodyPart[1] - 1) * 10))
            self.snake_game.blit(self.food_piece, (self.food[0] * 10, (self.ARRAY_SIZE - self.food[1] - 1) * 10))
            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = SnakeGame()
    game.run()
