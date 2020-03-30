import numpy as np

def score_game(game_core):
    '''Запускаем игру 1000 раз, чтоб узнать как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1, 101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    # score = int(np.mean(count_ls))
    score = np.mean(count_ls)
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)

def game_core_v3(number):
    '''Деление пополам'''
    def dimidiate(min, max):
        return int(min + (max-min) / 2)
    count = 1
    min = 1
    max = 100
    predict = dimidiate(min, max)
    while number != predict:
        count+=1
        if number > predict: 
            min = predict + 1
        elif number < predict: 
            max = predict - 1
        predict = dimidiate(min, max)
    return(count) # выход из цикла, если угадали

# Проверяем
score_game(game_core_v3)