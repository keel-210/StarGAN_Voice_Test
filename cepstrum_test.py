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
power = 0.2
scale = 1/18
path = './test.wav'
out_path = './cepstrum_test.wav'

wav_bps, data = wav.read(path)
window = np.hanning(wav_bps*1)

datas = np.zeros((100, ((wave_len - stride) + size * stride)))

pos = 4 * all_size
data_raw = np.array(data[pos:pos+wav_bps*1], dtype='float32')
data_fft = np.array(data_raw*window, dtype='float32')
ceps = pysptk.sptk.mcep(data_fft, miniter=2, order=128, etype=2, eps=-1)
ceps = np.array(ceps, dtype='float64')
pitch = pysptk.sptk.rapt(data_raw, wav_bps, 50, otype='pitch')
pitch = np.array(pitch, dtype='float64')
pitch = pysptk.sptk.excite(pitch, 100)
pitch *= 500
#pitch = [p if p<10 else p*500 for p in pitch]
delay = pysptk.sptk.mlsadf_delay(128,4)
mlsa = np.array([pysptk.sptk.mlsadf(p, ceps, 0.42, 4, delay) for p in pitch],dtype='int16')
plt.subplot(3,1,1)
plt.plot(data_raw)
plt.subplot(3,1,2)
plt.plot(pitch)
plt.subplot(3,1,3)
plt.plot(mlsa)
plt.show()
wav.write(out_path,wav_bps,mlsa)
