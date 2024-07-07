import numpy as np
import librosa
from spafe.features import lfcc, bfcc, gfcc, ngcc, cqcc

def c_mfcc(audio, sr, c_size=20):
    'Извлечение MFCC'
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=c_size)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def c_lfcc(audio, sr, c_size=20):
    'Извлечение LFCC'
    lfccs = lfcc.lfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    lfccs = np.mean(lfccs.T, axis=1)
    return(lfccs)

def c_bfcc(audio, sr, c_size=20):
    'Извлечение BFCC'
    bfccs = bfcc.bfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    bfccs = np.mean(bfccs.T, axis = 1)
    return(bfccs)

def c_gfcc(audio, sr, c_size=20):
    'Извлечение GFCC'
    gfccs = gfcc.gfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    gfccs = np.mean(gfccs.T, axis=1)
    return(gfccs)

def c_cqcc(audio, sr, c_size=20):
    'Извлечение CQCC'
    cqccs = cqcc.cqcc(audio, sr, num_ceps = c_size)
    cqccs = np.mean(cqccs.T, axis=1)
    return(cqccs)

# извлечение NGCC
def c_ngcc(audio, sr, c_size=20):
    'Извлечение NGCC'
    ngccs = ngcc.ngcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    ngccs = np.mean(ngccs.T, axis=1)
    return(ngccs)

def features_extraction(audio, sr):
    'Извлечение всех признаков (20 MFCC + 20 LFCC + 20 CQCC + 20 BFCC)'
    vector = c_mfcc(audio, sr)
    vector = np.append(vector, c_lfcc(audio, sr))
    vector = np.append(vector, c_cqcc(audio, sr))
    vector = np.append(vector, c_bfcc(audio, sr))
    return vector