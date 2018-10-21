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

stride = 128
wave_len = 254
size = 128
all_size = ((wave_len - stride) + size * stride)
path = './test.wav'
out_path = './cepstrum_test.wav'
ideal_path = './cepstrum_test_ideal.wav'
# read .wav
wav_bps, data = wav.read(path)

# const.
length = 2 ** 18
window = np.hanning(length)
hopSize = 10
pos = 1 * wav_bps
alpha = 0.43
order = 128

data_raw = np.array(data[pos:pos + length], dtype='float64')
wav.write(ideal_path, wav_bps, data_raw.astype('int16'))
data_fft = np.array(data_raw*window, dtype='float64')
ceps = pysptk.sptk.mcep(data_fft, order=order, alpha=alpha)
ceps = np.array(ceps, dtype='float64')

pitch = pysptk.sptk.swipe(data_raw, wav_bps, hopSize, max=5000, threshold=0.3, otype='pitch').astype('float64')
pitch = pysptk.sptk.excite(pitch, hopsize=hopSize)
pitch = [p if p > 2 else p/100 for p in pitch]

b_ceps = np.array([pysptk.mc2b(ceps, alpha=alpha)])
synth = pysptk.synthesis.Synthesizer(pysptk.synthesis.MLSADF(order=order, alpha=alpha), length-200)
vocode = synth.synthesis(pitch, b_ceps)
vocode /= 100

plt.subplot(2, 2, 1)
plt.plot(data_raw)
plt.subplot(2, 2, 2)
plt.plot(data_raw)
plt.subplot(2, 2, 3)
plt.plot(vocode)
plt.subplot(2, 2, 4)
plt.plot(pitch)
plt.show()

wav.write(out_path, wav_bps, vocode.astype('int16'))
