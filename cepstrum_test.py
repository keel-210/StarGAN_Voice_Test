import argparse
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import fft, ifft
import pysptk

np.seterr(divide='ignore', invalid='ignore')


def _pow_scale(fft, p):
    return np.power(fft, p)


def overwrap(fft, length, size, stride, phase, scale):
    X = np.zeros((size, length))
    for i in range(size):
        for j in range(length):
            if(j <= (int)(length/2)):
                X[i, j] = fft[i, j]
            else:
                X[i, j] = fft[i, length - j]
    X /= scale
    X = _pow_scale(X, 5)
    dst = np.zeros(((length - stride) + size * stride), dtype=float)
    for i, (f, p) in enumerate(zip(X, phase)):
        comp = np.zeros(wave_len, dtype='complex128')
        comp = (f.real * np.cos(p)) + (1j * f.real * np.sin(p))
        dst[i * stride:i * stride + length] += np.fft.ifft(comp).real
    return dst


def FFT(wav, length, stride, window, pos, size):
    data_fft = np.array(wav[pos:pos+(wave_len-size)+size*stride]*window)
    data_fft = np.fft.fft(data_fft)
    fft_phase = np.arctan2(data_fft.imag, data_fft.real)
    data_fft = np.abs([data_fft], dtype='f4')

    return data_fft, fft_phase

stride = 128
wave_len = 254
size = 128
all_size = ((wave_len - stride) + size * stride)
window = np.hanning(all_size)
power = 0.2
scale = 1/18
path ='./test.wav'
out_path = './cepstrum_test.wav'

wav_bps, data = wav.read(path)

datas = np.zeros((100, ((wave_len - stride) + size * stride)))

for i in range(1):
    pos = 4 * all_size
    data_fft = np.array(data[pos:pos+(wave_len-size)+size*stride]*window,dtype='float32')
    ceps = pysptk.sptk.mcep(data_fft,order= 127,etype=2,eps=-2.71828)
    pitch = pysptk.sptk.rapt(data_fft,wav_bps,(int)(wav_bps/100),otype='pitch')
    #fft_abs, phase = FFT(data, wave_len, stride, window,10 * all_size, size)
    # log_fft = np.log(fft_abs[0])
    # ceps = np.fft.fft(log_fft)
    plt.plot(pitch)
    plt.show()
    #data_ifft = overwrap(fft_abs[0], wave_len, size, stride, phase, scale)
    #data_ifft = np.reshape(data_ifft, -1)
    #datas[i] = data_ifft

#datas = np.reshape(datas, -1)
#datas = np.array(datas, np.int16)

#wav.write(out_path, wav_bps, datas)
