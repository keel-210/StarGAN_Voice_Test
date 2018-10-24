import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pysptk
import pyworld as pw

path = './test.wav'
out_path = './cepstrum_test_pyworld.wav'
ideal_path = './cepstrum_test_pyworld_ideal.wav'
# read .wav
wav_bps, data = wav.read(path)

alpha = 0.43
order = 63
length = 14000
pos = 2 * wav_bps

data_raw = np.array(data[pos:pos + wav_bps*5], dtype='float64')
f0, sp, pitch = pw.wav2world(data_raw, wav_bps)
ceps = pysptk.sp2mc(sp, order, alpha)
print(np.shape(ceps))
sp = pysptk.mc2sp(ceps, alpha, 2048)
y = pw.synthesize(f0, sp, pitch, wav_bps)

plt.subplot(2, 2, 1)
plt.plot(data_raw)
plt.subplot(2, 2, 2)
plt.plot(ceps[:, 1:63])
plt.subplot(2, 2, 4)
plt.plot(ceps[:, 0])
plt.subplot(2, 2, 3)
plt.plot(y)
plt.show()

wav.write(ideal_path, wav_bps, data_raw.astype('int16'))
wav.write(out_path, wav_bps, y.astype('int16'))
