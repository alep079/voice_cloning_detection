import numpy as np
import librosa
from spafe.features import lfcc, bfcc, gfcc, ngcc, cqcc
      
# извлечение MFCC
def c_mfcc(file, c_size):
    audio, sr = librosa.load(file, res_type = 'kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=c_size)
    mfccs = np.mean(mfccs.T,axis=0)
    return mfccs

# извлечение LFCC
def c_lfcc(file, c_size):
    audio, sr = librosa.load(file,res_type = 'kaiser_fast' )
    lfccs = lfcc.lfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    lfccs = np.mean(lfccs.T,axis=1)
    return(lfccs)

# извлечение BFCC
def c_bfcc(file, c_size):
    audio, sr = librosa.load(file,res_type = 'kaiser_fast' )
    bfccs = bfcc.bfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    bfccs = np.mean(bfccs.T, axis = 1)
    return(bfccs)

# извлечение GFCC
def c_gfcc(file, c_size):
    audio, sr = librosa.load(file,res_type = 'kaiser_fast' )
    gfccs = gfcc.gfcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    gfccs = np.mean(gfccs.T,axis=1)
    return(gfccs)

# извлечение CQCC
def c_cqcc(file, c_size):
    audio, sr = librosa.load(file,res_type = 'kaiser_fast' )
    cqccs = cqcc.cqcc(audio, sr, num_ceps = c_size)
    cqccs = np.mean(cqccs.T,axis=1)
    return(cqccs)

# извлечение NGCC
def c_ngcc(file, c_size):
    audio, sr = librosa.load(file,res_type = 'kaiser_fast' )
    ngccs = ngcc.ngcc(audio, fs = sr, num_ceps=c_size, nfilts = c_size*2)
    ngccs = np.mean(ngccs.T,axis=1)
    return(ngccs)


