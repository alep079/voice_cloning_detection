import os
import numpy as np
import fleep
from pydub import AudioSegment

from keras.models import load_model
from c_features import *
from voice_auth import *
from metrics import *

C_SIZE = 100
COEFF = c_bfcc

# функция оценки синтезированного голоса
def mark(model, target):
    target = np.reshape(target, (1, C_SIZE, 1))
    array = model.predict(target)
    if array[0] < 0.5:
        print('Это настоящий голос! Подсчитанный коэффициент:' + str(1- array[0]))
    else:
        print('Это синтезированный голос! Подсчитанная коэффициент:' + str(array[0]))

# вывод всех синтезированных записей
def print_all_synt(predict):
    array = []
    for audio in predict:
        if predict[audio] == 1:
            array.append(audio)
    return (array)

# функция конвертации файла
def init_proces(src):
    with open(src, "rb") as f:
        info = fleep.get(f.read(128))

    # преобразовываем mp3 wav
    if info.extension[0] == 'mp3':
        sound = AudioSegment.from_mp3(src)
        sound.export(src[:-3] + 'wav', format="wav")
        src = src[:-3] + 'wav'
#    elif info.extension[0] != ('wav' and 'flac' and 'mp3'):
#        print ('Неправильный формат файла')
    return(src)

def get_audio():
    while True:
        ans = input("Введите полный путь до аудиофайла: ")
        if os.path.isfile(ans):
            ans = init_proces(ans)
        else:
            print("Такого файла нет\n")
        return(ans)

# Функция приложения, принимает 1 входной файл и выносит вердикт
def API(model):
    #os.system("clear")
    while True:
        print("Вы хотите использовать существующий файл или записать свой голос?")
        print('1 - Использовать существующий')
        print('2 - Записать свой голос')
        ans = input()
        while ans not in ('1', '2'):
            print("Ошибка. Выберите 1 или 2")
            print('1 - Использовать существующий')
            print('2 - Записать свой голос')
            ans = input()

        if ans == '1':
            audio = get_audio()
            target = COEFF(audio, C_SIZE)
            mark(model, target)

        if ans == '2':
            try:
                sec = int(input("Введите целое количество секунд для записи: \n"))
            except ValueError:
                print("Попробуйте целое число\n")
            filename = voice_record(sec)
            target = c_mfcc(filename, C_SIZE)
            os.remove(filename)
            mark(model, target)

        ans = input("Вы хотите проверить еще один голос? [д/н]: ")
        if ans == 'д' or ans == 'l':
            #os.system("clear")
            continue
        else:
            break

def check_list(model, file):
    predict = {}
    with open(file) as f:
        for audio in f.readlines():
            predict[audio.strip()] = mark_file(model, audio.strip())
    print(print_all_synt(predict))
    return predict


# функция аутентификации по голосу, записывает в течении какого-то времени голос и выносит по нему вердикт
def voice_auth():
    while True:
        print("Вы хотите добавить говорящего для аутентификации или проверить голос?")
        print('1 - Добавить говорящего')
        print('2 - Распознать говорящего')
        ans = input()
        while ans not in ('1', '2'):
            print("Ошибка. Выберите 1 или 2")
            print('1 - Добавить говорящего')
            print('2 - Распознать говорящего')
            ans = input()

        if ans == '1':
            audio = get_audio()
            name = input('Введите свое имя')
            remember(name, audio)
        if ans == '2':
            audio = get_audio()
            recognize(audio)

        ans = input("Вы хотите продолжить? [д/н]: ")
        if ans == 'д' or ans == 'l':
            #os.system("clear")
            continue
        else:
            break

# функция вывода контекстного меню
def menu():
    print("------------------------------")
    print("1. Проверка голоса")
    print("2. Проверка списка аудиофайлов, поиск поддельных")
    print("3. Аутентификация по голосу")
    print("4. Выход")
    print("------------------------------")

if __name__ == '__main__':

#    ans = input("Введите полный путь до модели: ")
#    try:
#        model = load_model(ans)
#    except OSError:
#        print("Такой модели нет\n")
#    ans = input("Введите полный путь до модели: ")

    model = load_model('model_bfccs', custom_objects={"F1": F1, "DCF": DCF})
    while True:
        menu()
        try:
            ans = int(input("Выберите режим работы: \n"))
        except ValueError:
            print("Это не номер\n")
        if ans == 1:
            API(model)
        elif ans == 2:
            file = input("Введите полный путь до файла: ")
            print(check_list(model, file))
        elif ans == 3:
            voice_auth()
        elif ans == 4:
            exit()
        else:
            print("Некорректный номер\n")
            continue
