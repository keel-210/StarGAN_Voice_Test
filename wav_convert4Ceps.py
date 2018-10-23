import numpy as np
import os
import sys
import argparse
import scipy.io.wavfile as wav
import tqdm
from glob import glob

def saveZ_datas(datas,path,attribute,phase):
	j = len(os.listdir(path))
	for i in range(len(datas)):
		np.savez_compressed(path+str('{0:03d}'.format(i + j))+'.npz',MCep=datas[i],attr=attribute,phase = phase[i])

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p)

def mel_ceps(wav,length,alpha,order):

def FFT(wav, length, stride, window, pos, size, power, scale):
	wave_len = length
	data_fft = np.array([wav[p:p+wave_len, 0]*window for p in range(pos,pos + size * stride, stride)])
	data_fft = np.fft.fft(data_fft, axis=1)
	fft_phase = np.arctan2(data_fft.imag, data_fft.real, dtype='f4')
	data_fft = data_fft[:, :size]
	data_fft = np.abs([data_fft], dtype='f4')
	data_fft = _pow_scale(data_fft, power)
	data_fft *= scale

	return data_fft, fft_phase

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
		comp = np.zeros(length, dtype='complex128')
		comp = (f.real * np.cos(p) + (1j * f.real * np.sin(p)))
		dst[i * stride:i * stride + length] += np.fft.ifft(comp).real
	return dst

def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--path', type=str,   default='./wav/Nekomasu2.wav')
	parser.add_argument('--out_path', type=str, default='./data_mcep/')
	parser.add_argument('--length', type=int, default=14000)  # 254固定
	args = parser.parse_args()

	stride = args.stride
	wave_len = args.length
	size = args.size
	window = np.hanning(wave_len)
	all_size = ((wave_len - stride) + size * stride)
	power = 0.2
	scale = 1/18

	_, data = wav.read(args.path)

	datas = np.zeros((50, size,size))
	phases = np.zeros((50, size,wave_len))
	for i in range(50):
		fft_abs, phase = FFT(data, wave_len, stride, window,i * all_size, size, power, scale)
		datas[i] = fft_abs
		phases[i] = phase
	saveZ_datas(datas,args.out_path,args.attr,phases)

	test_files = glob(os.path.join(args.out_path, '*'))
	print(test_files)
	testA_list = test_files[:50]
	print(testA_list)
	datas = np.zeros((50, ((args.length - stride) + size * stride)))
	FFTs = [np.load(val)['FFT'] for val in testA_list]
	phases = [np.load(val)['phase'] for val in testA_list]
	print(np.shape(FFTs))
	for i,(w,p) in enumerate(zip(FFTs,phases)):
		data_ifft = overwrap(w, wave_len, size, stride, p, scale)
		data_ifft = np.reshape(data_ifft, -1)
		datas[i] = data_ifft
	datas = np.reshape(datas, -1)
	datas = np.array(datas, np.int16)
	wav.write(args.out_path+'_real.wav', 44100, datas)
   
if __name__ == '__main__':
	main()
