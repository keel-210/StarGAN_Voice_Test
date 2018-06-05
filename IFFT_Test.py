import argparse
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import fft, ifft

np.seterr(divide='ignore', invalid='ignore')


def _pow_scale(fft, p):
    return np.power(np.abs(fft), p) * np.sign(fft)


def overwrap(fft, length, size, stride, phase):
    X = np.zeros((size, length))
    for i in range(size):
        for j in range(length):
            if(j <= (int)(length/2)):
                X[i, j] = fft[i, j]
            else:
                X[i, j] = fft[i, length-j]
    dst = np.zeros(((length - stride) + size * stride), dtype=float)
    for i, (f, p) in enumerate(zip(X, phase)):
        comp_size = np.sqrt(np.power(f.real, 2)+np.power(f.imag, 2))
        comp = (comp_size * np.cos(p)) + (1j * comp_size * np.sin(p))
        dst[i * stride:i * stride + length] += np.fft.ifft(comp).real
        #dst[i * stride:i * stride + length] += np.fft.ifft(f).real
    return dst


def FFT(wav, length, stride, window, pos, size):
    wave_len = length
    data_fft = np.array([wav[p:p+wave_len, 0]*window for p in range(pos,
                                                                    pos + (wave_len - stride) + size * stride, stride)])
    data_fft = np.fft.fft(data_fft, axis=1)
    fft_phase = [np.arctan(f.imag/f.real) for f in data_fft]
    data_fft = data_fft[:, :size]
    data_fft = np.abs([data_fft], dtype='f4')

    # im = data_fft
    # plt.imshow(np.squeeze(im*256))
    # plt.waitforbuttonpress(0)

    return data_fft, fft_phase


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str,   default='./test.wav')
parser.add_argument('--out_path', type=str, default='./out_test_32.wav')
parser.add_argument('--length', type=int, default=254)  # 254固定
parser.add_argument('--size', type=int, default=128)  # 128固定
parser.add_argument('--stride', type=int,   default=32)
args = parser.parse_args()

bps = 141000
stride = args.stride
wave_len = args.length
size = args.size
window = np.hanning(wave_len)
all_size = ((wave_len - stride) + size * stride)

wav_bps, data = wav.read(args.path)
print(wav_bps)

datas = np.zeros((100, ((args.length - stride) + size * stride)))

for i in range(100):
    fft_abs, phase = FFT(data, wave_len, stride, window, i * all_size, size)
    data_ifft = overwrap(fft_abs[0], wave_len, size, stride, phase)
    data_ifft = np.reshape(data_ifft, -1)
    datas[i] = data_ifft

datas = np.reshape(datas, -1)
datas = np.array(datas, np.int16)

wav.write(args.out_path, wav_bps, datas)
