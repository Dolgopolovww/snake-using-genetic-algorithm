import random
import numpy as np


class brain:
    def __init__(self, layers, width, height, block, random_weights=True, random_bases=True):
        self.nextFood = None
        self.outputs = []
        self.weights = []
        self.prev_result = 1
        self.bases = []
        self.prev_food_cost = 1.0
        self.block = block
        self.width = width
        self.height = height
        if random_weights == True:
            for i in range(len(layers) - 1):
                theta = np.random.uniform(low=-0.5, high=.5, size=(layers[i], layers[i+1]))
                self.weights.append(theta)
        if random_bases == True:
            for i in range(len(layers) - 1):
                base = np.random.uniform(low=-0.1, high=0.1, size=(1, layers[i+1]))
                self.bases.append(base)

    # возвращает true, если x, y является частью змеи, иначе false
    def isBody(self, x, y, snake):
        for i in range(3, len(snake) - 1):
            if snake[i][0] == x and snake[i][1] == y:
                return True
        return False

    # следующая позиция и направление на основе переданного результата
    def next_position_direction(self, x, y, direction, result):
        l = self.block
        if direction == 'north':
            if result == 1:
                return (x, y - l), 'north'
            elif result == 2:
                return (x - l, y), 'west'
            else:
                return (x + l, y), 'east'
        elif direction == 'east':
            if result == 1:
                return (x + l, y), 'east'
            elif result == 2:
                return (x, y - l), 'north'
            else:
                return (x, y + l), 'south'
        elif direction == 'south':
            if result == 1:
                return (x, y + l), 'south'
            elif result == 2:
                return (x + l, y), 'east'
            else:
                return (x - l, y), 'west'
        else:
            if result == 1:
                return (x - l, y), 'west'
            elif result == 2:
                return (x, y + l), 'south'
            else:
                return (x, y - l), 'north'

    # возвращает список с тремя элементами, указывающими еду, часть тела и границу в зависимости от пройденного направления
    def look_in_direction(self, x, y, dirx, diry, fx, fy, snake):
        distance = 1
        input = [0, 0, 0]
        food_found = False
        body_found = False
        while((x != 0) and (x != self.width-self.block) and (y != 0) and (y != self.height-self.block)):
            x, y = x + dirx, y + diry
            distance += 1
            if(not food_found and fx == x and fy == y): # нашел хавку
                input[0] = 1
                food_found = True
            if(not body_found and self.isBody(x, y, snake)): # нашел тело
                input[1] = 1 / distance
                body_found = True
        input[2] = 1 / distance
        return input

    # делает ввод для нейронной сети, передавая все 8 направлений в look_in_direction
    def make_input(self, x, y, fx, fy, snake, direction):
        input = []
        # смотреть в сторону, куда движется змея
        (new_x, new_y), _ = self.next_position_direction(x, y, direction, 1)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 90 градусов влево от направления движения змеи
        (new_x, new_y), _ = self.next_position_direction(x, y, direction, 2)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 90 градусов вправо от направления движения змеи
        (new_x, new_y), _ = self.next_position_direction(x, y, direction, 3)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 45 градусов влево от направления движения змеи
        (tempx, tempy), new_dir = self.next_position_direction(x, y, direction, 1)
        (new_x, new_y), _ = self.next_position_direction(tempx, tempy, new_dir, 2)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 45 градусов вправо от направления движения змеи
        (tempx, tempy), new_dir = self.next_position_direction(x, y, direction, 1)
        (new_x, new_y), _ = self.next_position_direction(tempx, tempy, new_dir, 3)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # смотреть в противоположном направлении, в котором движется змея
        (tempx, tempy), new_dir = self.next_position_direction(x, y, direction, 2)
        (new_x, new_y), new_dir = self.next_position_direction(tempx, tempy, new_dir, 2)
        (new_x, new_y), _ = self.next_position_direction(new_x, new_y, new_dir, 2)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 135 градусов вправо от направления движения змеи
        (tempx, tempy), new_dir = self.next_position_direction(x, y, direction, 3)
        (new_x, new_y), _ = self.next_position_direction(tempx, tempy, new_dir, 3)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        # посмотрите на 135 градусов влево, куда движется змея
        (tempx, tempy), new_dir = self.next_position_direction(x, y, direction, 2)
        (new_x, new_y), _ = self.next_position_direction(tempx, tempy, new_dir, 2)
        dir_x, dir_y = new_x - x, new_y - y
        input.extend(self.look_in_direction(x, y, dir_x, dir_y, fx, fy, snake))
        return input

    # прямая связь с использованием нейронной сети
    def decision_from_nn(self, x, y, snake, direction):
        closer_to_food = True
        fx, fy = self.nextFood
        input = self.make_input(x, y, fx, fy, snake, direction)
        input = np.array(input)

        output = input
        for i in range(len(self.weights) - 1):
            output = self.relu(np.dot(output, self.weights[i]) + self.bases[i])
            self.outputs.append(output)
        output = self.softmax(np.dot(output, self.weights[i+1]) + self.bases[i+1])
        self.outputs.append(output)
        result = np.argmax(self.outputs[-1]) + 1
        return result

    # установить следующую переменную еды
    def setNextFood(self, food):
        self.nextFood = food

    def sigmoid(self, mat):
        return 1.0 / (1.0 + np.exp(-mat))

    def relu(self, mat):
        return mat * (mat > 0)

    def softmax(self, mat):
        mat = mat - np.max(mat)
        return np.exp(mat) / np.sum(np.exp(mat), axis=1)
