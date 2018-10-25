import numpy as np
import os
import scipy.io.wavfile as wav
import sys
import pyworld as pw
import pysptk

stride = 128
wave_len = 254
size = 128
window = np.hanning(wave_len)
all_size = ((wave_len - stride) + size * stride)
power = 0.2
scale = 1/18

def load_FFT_attr(data_dir):
    attr_list = np.zeros((len(os.listdir(data_dir+'\\train\\')),7))
    for i in range(len(os.listdir(data_dir))):
        data = np.load(data_dir+'\\train\\'+'{0:06d}'.format(i)+'.npz')
        attr_list[i] = data['attr']
    return  attr_list

def overwrap(fft, length, size, stride, _phase, scale):
    X = np.zeros((size, length))
    fft = np.array(fft)
    for i in range(size):
        for j in range(length):
            if(j <= (int)(length/2)):
                X[i,j] = fft[i,j,0]
            else:
                X[i,j] = fft[i, length - j,0]
    X /= scale
    X = np.power(X, 5)
    dst = np.zeros(((length - stride) + size * stride), dtype=float)
    for i, (f, p) in enumerate(zip(X, _phase)):
        comp = np.zeros(wave_len, dtype='complex128')
        comp = (f.real * np.cos(p)) + (1j * f.real * np.sin(p))
        dst[i * stride:i * stride + length] += np.fft.ifft(comp).real
    return dst

def save_wav(realA, realB, fake_B, image_size, sample_file, phase, num=10):
    #datas = np.zeros((len(realA), ((wave_len - stride) + size * stride)))
    #for i,(w,p) in enumerate(zip(realA,phase)):
        #data_ifft = overwrap(w, wave_len, size, stride, p, scale)
        #data_ifft = np.reshape(data_ifft, -1)
        #datas[i] = data_ifft
    #datas = np.reshape(datas, -1)
    #datas = np.array(datas, np.int16)
    #wav.write(sample_file+'_real.wav', 44100, datas)

    datas = np.zeros((len(fake_B), ((wave_len - stride) + size * stride)))
    for i,(w,p) in enumerate(zip(fake_B,phase)):
        data_ifft = overwrap(w, wave_len, size, stride, p, scale)
        data_ifft = np.reshape(data_ifft, -1)
        datas[i] = data_ifft
    datas = np.reshape(datas, -1)
    datas = np.array(datas, np.int16)
    wav.write(sample_file+'_fake.wav', 44100, datas)
    
def save_wav_ceps(fake_B,input_path, sample_path):
    length = 14000
    bps,wav_data = wav.read(input_path)
    datas = [wav_data[i:i+length,0] for i in range(0,len(wav_data),length)]
    wave = np.zeros([len(fake_B),length])
    for (b,d) in zip(fake_B,datas):
        f0,_,pitch = pw.wav2world(d, bps)
        for cep in b:
            for i,Scep in enumerate(cep):
                if(i==0):
                    Scep = (Scep*28)-20
                else:
                    Scep = (Scep*7)-3
                cep[i] = Scep
        sp = pysptk.mc2sp(b,0.48,2048)
        w = pw.synthesize(f0, sp, pitch, bps)
        np.append(wave,w)
    wave = np.reshape(wave,-1).astype('int16')
    wav.write(sample_path+'_fake.wav', bps, wave)