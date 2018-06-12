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
    return dst

def FFT(wav, length, stride, window, pos, size):
    wave_len = length
    data_fft = np.array([wav[p:p+wave_len, 0]*window for p in range(pos,
                                                                    pos + (wave_len - stride) + size * stride, stride)])
    data_fft = np.fft.fft(data_fft, axis=1)
    fft_phase = [np.arctan(f.imag/f.real) for f in data_fft]
    data_fft = data_fft[:, :size]
    data_fft = np.abs([data_fft], dtype='f4')
    return data_fft, fft_phase

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str,   default='./test.wav')
parser.add_argument('--out_path', type=str, default='./out_test.wav')
parser.add_argument('--length', type=int, default=254)  # 254固定
parser.add_argument('--size', type=int, default=128)  # 128固定
parser.add_argument('--stride', type=int,   default=64)
args = parser.parse_args()

stride = args.stride
wave_len = args.length
size = args.size
window = np.hanning(wave_len)
all_size = ((wave_len - stride) + size * stride)

wav_bps, data = wav.read(args.path)
print(wav_bps)

datas = np.zeros((100, ((args.length - stride) + size * stride)))

print(all_size)

pos = 5000
wav_data = data[pos:pos+wave_len,0] * window
fft_data = np.fft.fft(wav_data)
fft_abs = np.abs(fft_data)
fft_phase = np.arctan2(fft_data.imag,fft_data.real)
print(fft_phase)
fft_comp = np.zeros(wave_len,dtype = 'complex128')
for i in range(wave_len):
	fft_comp[i] = fft_abs[i].real * np.cos(fft_phase[i]) + fft_abs[i].real *np.sin(fft_phase[i]) * 1j

fig, plots  = plt.subplots(nrows=3,ncols=2, figsize=(10,5))
plots[0,0].plot(fft_data.real,color = 'r',linewidth = 0.5)
plots[0,1].plot(fft_data.imag,color = 'b',linewidth = 0.5)

plots[1,0].plot(fft_abs.real,color = 'g',linewidth = 0.5)
plots[1,1].plot(fft_abs.imag,color = 'y',linewidth = 0.5)

plots[2,0].plot(fft_comp.real,color = 'g',linewidth = 0.5)
plots[2,1].plot(fft_comp.imag,color = 'y',linewidth = 0.5)

fig.show()
plt.waitforbuttonpress()