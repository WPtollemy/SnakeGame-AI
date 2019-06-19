import SnakeGame
import math
import numpy as np
import tflearn
from random import randint
from statistics import mean
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


class SnakeNN:

    def __init__(self, training_games=10000, test_games=1000, max_moves=1500, filename='SnakeNeuralNet2.tflearn'):
        self.training_games = training_games
        self.test_games = test_games
        self.max_moves = max_moves    # max moves a snake will take in case snake loops
        self.filename = filename
        self.DIRECTIONS = [
            [[0, 1], 0],    # up
            [[1, 0], 1],    # right
            [[0, -1], 2],   # down
            [[-1, 0], 3]    # left
        ]

    def train(self):
        training_data = self.create_training_data()
        neural_net = self.create_model()
        neural_net = self.train_model(training_data, neural_net)
        self.test_model(neural_net)

    def create_training_data(self):
        training_data = []
        for _ in range(self.training_games):
            game = SnakeGame.SnakeGame()
            done, prev_score, snake, food = game.get_game_info()
            prev_finding = self.create_finding(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.max_moves):
                decision, game_decision = self.create_decision(snake)
                done, score, snake, food = game.move(game_decision)
                if done:
                    training_data.append([self.add_decision_to_finding(prev_finding, decision), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_decision_to_finding(prev_finding, decision), 1])
                    else:
                        training_data.append([self.add_decision_to_finding(prev_finding, decision), 0])
                    prev_finding = self.create_finding(snake, food)
                    prev_food_distance = food_distance
        return training_data

    @staticmethod
    def create_model():
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        x = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(x, y, n_epoch=3, shuffle=True, run_id=self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        scores_arr = []
        for i in range(self.test_games):
            game = SnakeGame.SnakeGame()
            _, score, snake, food = game.start()
            prev_finding = self.create_finding(snake, food)
            for _ in range(self.max_moves):
                predictions = []
                for decision in range(-1, 2):
                    predictions.append(model.predict
                                       (self.add_decision_to_finding(prev_finding, decision).reshape(-1, 5, 1)))

                decision = np.argmax(np.array(predictions))
                game_decision = self.get_game_decision(snake, decision - 1)
                done, score, snake, food = game.nn_move(game_decision)
                if done:
                    break
                else:
                    prev_finding = self.create_finding(snake, food)

            print()
            print('Game: {}'.format(i))
            print('Score: {}'.format(score))
            scores_arr.append(score)

        print('Average score:', mean(scores_arr))

    def create_decision(self, snake):
        decision = randint(0, 2) - 1
        return decision, self.get_game_decision(snake, decision)

    def get_game_decision(self, snake, decision):
        snake_direction = self.get_snake_direction(snake)
        new_direction = snake_direction
        if decision == -1:
            new_direction = self.turn_snake_to_the_left(snake_direction)
        elif decision == 1:
            new_direction = self.turn_snake_to_the_right(snake_direction)
        for pair in self.DIRECTIONS:
            if pair[0] == new_direction.tolist():
                game_decision = pair[1]
        return game_decision

    def create_finding(self, snake, food):
        snake_direction = self.get_snake_direction(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        wall_left = self.is_direction_impassable(snake, self.turn_snake_to_the_left(snake_direction))
        wall_front = self.is_direction_impassable(snake, snake_direction)
        wall_right = self.is_direction_impassable(snake, self.turn_snake_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(wall_left), int(wall_front), int(wall_right), angle])

    @staticmethod
    def get_snake_direction(snake):
        return np.array(snake[0]) - np.array(snake[1])

    @staticmethod
    def get_food_direction_vector(snake, food):
        return np.array(food) - np.array(snake[0])

    def is_direction_impassable(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == -1 or point[1] == -1 or point[0] == 51 or point[1] == 51\
            or self.point_in_snake(point.tolist(), snake)

    @staticmethod
    def point_in_snake(point, snake):
        for i in snake:
            if np.array_equal(i, point):
                return True
        return False

    @staticmethod
    def turn_snake_to_the_left(vector):
        return np.array([-vector[1], vector[0]])

    @staticmethod
    def turn_snake_to_the_right(vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    @staticmethod
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def add_decision_to_finding(finding, decision):
        return np.append([decision], finding)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))


if __name__ == "__main__":
    SnakeNN().train()
