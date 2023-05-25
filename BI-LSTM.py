import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
from keras.layers import Dense, LSTM, Bidirectional
from c_features import *
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from metrics import *

# переменные
C_SIZE = 100                                                    # количесвто выделяемых коэффициентов
TSV_TRAIN_PATH = '/home/alep/Dataset/Audio/dataset_new.tsv'     # путь до tsv файла
TSV_TEST_PATH = '/home/alep/Dataset/Audio/dataset_new.tsv'      # путь до tsv файла
DATASET_PATH = '/home/alep/Dataset/Audio/'                      # путь до датасета с аудио

# загрузка извлеченных ранее признаков
def load_features(file, coeff = 'mfccs'):
    
    # необходимо при проблемах в совместимости версий numpy
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    ex_2 = np.load(file)
    np.load = np_load_old

    extracted_features = ex_2[coeff]
    features_df=pd.DataFrame(extracted_features,columns=['feature','fake'])
    x=np.array(list(features_df['feature'].tolist()))
    x_train = np.reshape(x,len(extract_features), C_SIZE, 1)
    y_train=np.array(features_df['fake'].tolist())
    return(x_train, y_train)

# извлечение признаков
def extract_features(data, func = c_mfcc):
    extracted_features=[]
    for i in range(0, len(data)):
        fake=data['fake'].loc[i]
        path = DATASET_PATH + str(data['path'].loc[i])
        extracted_features.append([func(path,C_SIZE),fake])
    features_df=pd.DataFrame(extracted_features,columns=['feature','fake'])

    # разбиваем массивы на нужные части и задаем размерности
    x=np.array(features_df['feature'].tolist())
    x_train = np.reshape(x,(len(data), C_SIZE, 1))
    y_train=np.array(features_df['fake'].tolist())
    return(x_train, y_train)

# построение графика
def fit_show(history, metric):
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.plot(history.history[metric])

# создание модели
def model_create():
    model = keras.Sequential(name="my_sequential")
# первый слой Bi-LSTM сети со 100 нейронами
    model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(C_SIZE, 1)),merge_mode='concat'))

# второй слой Bi-LSTM сети со 100 нейронами
    model.add(Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat'))

# третий слой Bi-LSTM сети со 100 нейронами
    model.add(Bidirectional(LSTM(150, return_sequences=False),merge_mode='concat'))

# последний сигмоид слой с 1 нейроном, который возвращает значение 0 или 1 в зависимости от того, поддельный голос или нет
    model.add(Dense(1, activation = 'sigmoid'))

    #компиляция модели с заданами метриками, оптимизатором и потерями
    model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
               metrics=['accuracy', 
                        'AUC',
                        DCF,
                        F1],
                        run_eagerly=True)
    return(model)

# обучение и сохранение модели
def save_model(model, x_train, y_train, save = True, filename = 'model'):
    #обучение
    history = model.fit(x_train, y_train, epochs = 10, batch_size = 50)

    if save == True:
    #сохраняем обученную модель
        try:
            model.save(filename)
            print(f'Модель сохранена в {filename}')
        except:
            print('Не получилось сохранить модель')
    return(history)

if __name__ == "__main__":

# загружаем данные из датасета
    try:
        data_train = pd.read_csv(TSV_TRAIN_PATH, sep='\t')
        data_test = pd.read_csv(TSV_TEST_PATH, sep='\t')
    except:
        print('Неправильные пути файлов')
        exit()
        
# извлечение признаков
    try:
        x_train, y_train = extract_features(data_train)
        x_test, y_test = extract_features(data_test)
    except:
        print('Не получилось извлечь признаки')
        exit()

    model = model_create()
    history = save_model(model, x_train, y_train, save=False)

#загружаем обученную модель
    #model = load_model('model_all_20')

#тестирование
    model.evaluate(x_test, y_test, batch_size = 32)