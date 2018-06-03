import argparse
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import fft, ifft

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p) * np.sign(fft)

def overwrap(fft, length, dif, side):
    X = np.zeros((128,254))
    for i in range(128):
        for j in range(254):
            if(j < 128):
                X[i,j] = fft[0,i,j]
            else :
                X[i,j] = fft[0,i,254-j]
    dst = np.zeros(dif * (side-1)+length, dtype=float)
    for i, f in enumerate(X):
        dst[i*dif:i*dif+length] += np.fft.ifft(f).real
    return dst

def FFT(wav,stride,window,pos):
    wave_len = stride*2 - 2
    data_fft = np.array([wav[p:p+wave_len,0]*window for p in range(pos, pos+(wave_len-stride)+stride*stride, stride)])
    fft_phase = np.array([np.arctan(data_fft[i].imag/data_fft[i].real) for i in range(len(data_fft))])
    data_fft = np.fft.fft(data_fft, axis=1)
    data_fft = data_fft[:,:stride]
    data_fft = np.abs([data_fft], dtype=np.float32)

    im = data_fft
    #plt.imshow(np.squeeze(im*256))
    #plt.waitforbuttonpress(0)

    return data_fft,fft_phase

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path',          type=str,   default='./test.wav')
parser.add_argument('--out_path',      type=str,   default='./out_test.wav')
parser.add_argument('--length',        type=int,   default=254)
parser.add_argument('--stride',        type=int,   default=128)
args = parser.parse_args()

bps = 141000
stride = args.stride
wave_len = stride*2-2
window = np.hanning(wave_len)
test_len = 10
data_size = 10000

wav_bps, data = wav.read(args.path)
print(wav_bps)

datas = np.zeros((100,16510))
for i in range(100):
    size, phase = FFT(data,128,window,i*((args.length-stride)+stride*stride))
    data_ifft = overwrap(size,254,128,128)
    data_ifft = np.reshape(data_ifft, -1)
    datas[i] = data_ifft

datas = np.reshape(datas, -1)
print(np.shape(datas))
wav.write(args.out_path, (int)(wav_bps/16), datas)
