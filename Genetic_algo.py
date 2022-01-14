import snake
import random
import numpy as np
import pickle
from Arena import Arena
import argparse
from input import *
import time


# индикатор выполнения
def progress_bar(curr, total, length):
    frac = curr/total
    filled_bar = round(frac*length)
    print('\r', '#'*filled_bar + '-'*(length - filled_bar), '[{:>7.2%}]'.format(frac), end='')


def run(snakes, arena):
    i = 1
    count = [0 for _ in range(300)]
    snakes_killed = 0
    env_seed = random.random()
    for s in snakes:
        start_time = time.time()
        checkloop = False
        progress_bar(i, population_size, 30)
        random.seed(env_seed)
        s.Brain.setNextFood(arena.newFood(s.list))
        while s.isAlive():
            result = s.Brain.decision_from_nn(s.head_x, s.head_y, s.list, s.direction)
            # чтобы проверить, образована ли непрерывная петля змеей, а затем убить эту змею
            if s.steps_taken > 250:
                if not checkloop:
                    checkloop = True
                    any_point_of_loop = (s.head_x, s.head_y)
                    times = 0
                elif (s.head_x, s.head_y) == any_point_of_loop:
                    times += 1
                if times > 2:
                    s.crash_wall = True
                    s.crash_body = True
                    snakes_killed += 1
            else:
                checkloop = False
            # принудительное уничтожение, если петля не поймана
            if time.time() - start_time > 0.5:
                s.crash_wall = True
                s.crash_body = True
                snakes_killed += 1
            # если съела
            if (s.head_x, s.head_y) == arena.food:
                s.steps_taken = 0
                result = s.Brain.decision_from_nn(s.head_x, s.head_y, s.list, s.direction)
                if not s.increaseSize(result):
                    s.crash_wall = True
                start_time = time.time()
                s.Brain.setNextFood(arena.newFood(s.list))
            if s.move(result) == False:
                break
        random.seed()
        count[len(s.list) - 1] += 1
        i += 1
    print('\nраспределение змей с индексом в качестве оценки:',
          count[0:15], 'змеи убиты', snakes_killed)


# для вывода информации о пяти лучших змеях
def print_top_5(five_snakes):
    i = 0
    for snake in five_snakes:
        i += 1
        print('Змея : ', i, ', баллы : ', len(snake.list) -
              1, ', шаги : ', snake.steps_taken, end='\t')
        if snake.crash_body and snake.crash_wall:
            print('повторение')
        elif snake.crash_wall and not snake.crash_body:
            print('разбился об стену')
        else:
            print('съел тело')


# сохранение
def save_top_snakes(snakes,  filename):
    f = open(filename, 'wb')
    pickle.dump(snakes, f)
    f.close()


# Создание новой популяции для след поколения
def create_new_population(snakes):
    """
    выбор лучшей x% популяции и разведение их для создания новой популяции
    верхний x% и нижний y% также включаются в новую популяцию
    :param snakes:
    :return:
    """

    parents = []
    top_old_parents = int(population_size * per_of_best_old_pop / 100)
    bottom_old_parents = int(population_size * per_of_worst_old_pop / 100)
    for i in range(top_old_parents):
        parent = snake.snake(width, height, brainLayer, block_length,
                             random_weights=False, random_bases=False)
        parent.Brain.weights = snakes[i].Brain.weights
        parent.Brain.bases = snakes[i].Brain.bases
        parents.append(parent)
    for i in range(population_size - 1, population_size - bottom_old_parents - 1, -1):
        parent = snake.snake(width, height, brainLayer, block_length,
                             random_weights=False, random_bases=False)
        parent.Brain.weights = snakes[i].Brain.weights
        parent.Brain.bases = snakes[i].Brain.bases
        parents.append(parent)
    children = generate_children(parents, population_size - (top_old_parents + bottom_old_parents))
    children = mutate_children(children)
    parents.extend(children)
    return parents


def mutate_children(children):
    for child in children:
        for weight in child.Brain.weights:
            for ele in range(int(weight.shape[0]*weight.shape[1]*mutation_percent/100)):
                row = random.randint(0, weight.shape[0]-1)
                col = random.randint(0, weight.shape[1]-1)
                weight[row, col] += random.uniform(-mutation_intensity, mutation_intensity)
    return children


# генерация детей на основе переданных родителей
def generate_children(parents, no_of_children):
    all_children = []
    l = len(parents)
    for count in range(no_of_children):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = snake.snake(width, height, brainLayer, block_length)
        for i in range(len(parent1.Brain.weights)):
            for j in range(parent1.Brain.weights[i].shape[0]):
                for k in range(parent1.Brain.weights[i].shape[1]):
                    child.Brain.weights[i][j, k] = random.choice(
                        [parent1.Brain.weights[i][j, k], parent2.Brain.weights[i][j, k]])
            for j in range(parent1.Brain.bases[i].shape[1]):
                child.Brain.bases[i][0, j] = random.choice(
                    [parent1.Brain.bases[i][0, j], parent2.Brain.bases[i][0, j]])
        all_children.append(child)
    return all_children


def main():
    # получаем аргументы из командной строки
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True, help='путь сохраненной змейки')
    args = vars(ap.parse_args())
    snakes = [snake.snake(width, height, brainLayer, block_length) for _ in range(population_size)]
    arena = Arena(width, height, block_length)
    top_snakes = []
    for i in range(no_of_generations):
        print('Поколение : ', i+1, ',', end='\n')
        run(snakes, arena)
        # сортировка популяции по длине змеи и пройденным шагам
        snakes.sort(key=lambda x: (len(x.list), -x.steps_taken), reverse=True)
        print_top_5(snakes[0:5])
        # обобщение всей популяции
        print('Сохранение змейки')
        top_snakes.append(snakes[0])
        # сохранение улчшей змейки
        save_top_snakes(top_snakes, args['output'])
        snakes = create_new_population(snakes)


if __name__ == "__main__":
    main()
