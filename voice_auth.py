import os
import numpy as np
from scipy.spatial.distance import euclidean 
from keras.models import load_model
import librosa
from python_speech_features import sigproc
import wave
import pyaudio
from c_features import *

# Параметры сигнала
SR = 16000
ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
FFT = 512

# параметры для записи с микрофона
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Параметры модели
MODEL_AUTH_FILE = "/home/alep/Proba/nir/Voice/voice_auth_model_cnn"
MODEL_SYNT_FILE = "/home/alep/Proba/nir/model"
EMBED_FILE = "/home/alep/Proba/nir/Voice/data/embed"
THRESHOLD = 0.2
C_SIZE = 100
COEFF = c_lfcc

# оценка файла в бинарном виде
def mark_file(model, file):
    target = COEFF(file, C_SIZE)
    target = np.reshape(target,(1, C_SIZE, 1))
    array = model.predict(target)
    return (round(int(array[0])))

def voice_record(record_seconds):
    filename = "recorded.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SR,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)
    frames = []
    #os.system("clear")
    print("Запись...")
    for _ in range(int(SR/ CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Запись остановлена")
    # остановить и закрыть поток
    stream.stop_stream()
    stream.close()
    p.terminate()
    # сохранить аудиофайл
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SR)
    wf.writeframes(b"".join(frames))
    wf.close()
    return (filename)

# функция запоминания модели говорящего
def remember(name,file):
    try:
        model = load_model(MODEL_AUTH_FILE)
    except:
        print("Укажите в параметрах другой путь к модели")
        exit()
    
    speaker_repr = get_embedding(model, file)
    speaker_weight = np.array(speaker_repr.tolist())
    speaker = name
    np.save(os.path.join(EMBED_FILE, speaker + ".npy"), speaker_weight)
    print("Пользователь запомнен")

# распознование человека
def recognize(file):
    try:
        users = os.listdir(EMBED_FILE)
    except:
        print('Проверьте путь к файлу с представлениями')
    if len(users) is 0:
        print("Еще нет ни одного представления пользователя")
        exit()
    try:
        model_auth = load_model(MODEL_AUTH_FILE)
    except:
        print("Не удалось загрузить модель аутентификации")
        exit()
    try:
        model_synt = load_model(MODEL_SYNT_FILE)
    except:
        print("Укажите в параметрах другой путь к модели детектирования синтезированного голоса")
        exit()
    
        
    probabilities = {}
    speaker_repr = get_embedding(model_auth, file)
    speaker_weight = np.array(speaker_repr.tolist())
    for user in users:
        user_weight = np.load(os.path.join(EMBED_FILE, user))
        probability = euclidean(speaker_weight, user_weight)
        probabilities.update({user.replace(".npy", ""):probability})
    if mark_file(model_synt, file) == 1:
        print("Похоже, этот голос поддельный")
    elif min(list(probabilities.values())) < THRESHOLD:
        print(f"Этот голос принадлежит: {min(probabilities, key=probabilities.get)}")
        print(f"Оценка: {(1 - min(list(probabilities.values())))*100}%")
    else:
        print("К сожалению, данный голос не распознан")
        print(f"Оценка: {(1 - min(list(probabilities.values())))*100}%")
        exit()

# функция нормализации фреймов
def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])

# функция извлечения признаков
def get_fft_spectrum(filename):
    signal, sr= librosa.load(filename, sr=SR)
    signal = (signal.flatten())*(2**15)
    signal = sigproc.preemphasis(signal, coeff=ALPHA)
    frames = sigproc.framesig(signal, frame_len=FRAME_LEN*SR, frame_step=FRAME_STEP*SR, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=FFT))
    fft_norm = normalize_frames(fft.T)
    return (fft_norm)

# функция создания представления пользователя
def get_embedding(model, wav_file):
    signal = get_fft_spectrum(wav_file)
    embedding = np.squeeze(model.predict(signal.reshape(1, *signal.shape, 1)))
    return (embedding)