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
path = './test.wav'
out_path = './cepstrum_test.wav'

wav_bps, data = wav.read(path)

datas = np.zeros((100, ((wave_len - stride) + size * stride)))

for i in range(1):
    pos = 4 * all_size
    data_fft = np.array(data[pos:pos+(wave_len-size)+size*stride, 0]*window, dtype='float32')
    ceps = pysptk.sptk.mcep(data_fft, miniter=2, order=127, etype=2, eps=-2.71828)
    ceps = np.array(ceps, dtype='float64')
    delay = np.zeros(all_size*5)
    mlsa = pysptk.sptk.mlsadf(127, ceps, 0.35, 4, delay)
    pitch = pysptk.sptk.rapt(data_fft, wav_bps, (int)(wav_bps / 100), otype='pitch')
    pitch = np.array(pitch, dtype='float64')
    pitch = pysptk.sptk.excite(pitch, wav_bps)
    pitch *= 1000
    plt.plot(pitch)
    plt.show()
