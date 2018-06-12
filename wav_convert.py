import numpy as np
import os
import random
import scipy.io.wavfile as wav

def load(path, time=-1): #pathのwav読み込み、bpsとnp.arrayを返す
    bps, data = wav.read(path)
    if len(data.shape) != 1:
        data = data[:,0] + data[:,1]
    if time > 0:
        length = int(bps * time)
        if length <= len(data):
            dst = data[0:length]
        else:
            dst = np.zeros(length)
            dst[0:len(data)] = data
        data = dst
    return bps, data

def save(path, bps, data): #np.arrayからpathの位置にwavを書く
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    data = np.reshape(data, -1)
    wav.write(path, bps, data)

def find_wav(path): #path直下のデータパスの取得
    name = os.listdir(path)
    dst = [path + "/" + n for n in name]
    return dst, name

def image_single_split_pad(src, side, pos, power, scale, window): #スペクトログラムのデータ処理
    wave_len = side*2 - 2
    spl = np.array([src[p:p+wave_len]*window for p in range(pos, pos+side*side, side)])
    spl = np.fft.fft(spl, axis=1)
    spl = spl[:,:side]
    spl = np.abs([spl], dtype=np.float32)
    spl = _pow_scale(spl, power)
    spl *= scale
    return spl

def image_single_pad(src, side, power, scale, window):
    wave_len = side*2-2
    src = np.array(src)
    src *= scale
    src = _pow_scale(src, power)
    src = src.reshape((side, side))
    mil = np.array(src[:,1:side-1][:,::-1])
    src = np.concatenate([src, mil], 1)
    mil = None

    src = FGLA(src, wave_len, side, side, window)
    return src

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p)

def overwrap(fft, length, dif, side):
    dst = np.zeros(dif * (side-1)+length, dtype=float)
    for i, f in enumerate(fft):
        dst[i*dif:i*dif+length] += np.fft.ifft(f).real
    return dst

def split(w, length, dif, side, window):
    dst = np.array([np.fft.fft(w[i:i+length]*window) for i in range(0, side*dif, dif)])
    return dst
