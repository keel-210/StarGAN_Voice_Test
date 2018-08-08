import numpy as np
import os
import sys
import argparse
import scipy.io.wavfile as wav
import tqdm

def saveZ_datas(datas,path,attribute):
	j = len(os.listdir(path))
	for i in range(len(datas)):
		np.savez_compressed(path+str('{0:06d}'.format(i + j))+'.npz',FFT=datas[i],attr=attribute)

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p)

def FFT(wav, length, stride, window, pos, size, power, scale):
	wave_len = length
	data_fft = np.array([wav[p:p+wave_len, 0]*window for p in range(pos,pos + size * stride, stride)])
	data_fft = np.fft.fft(data_fft, axis=1)
	fft_phase = np.arctan2(data_fft.imag, data_fft.real)
	data_fft = data_fft[:, :size]
	data_fft = np.abs([data_fft], dtype='f4')
	data_fft = _pow_scale(data_fft, power)
	data_fft *= scale

	return data_fft, fft_phase

def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--path', type=str,   default='./wav/Akari.wav')
	parser.add_argument('--out_path', type=str, default='./all_datas/')
	parser.add_argument('--length', type=int, default=254)  # 254固定
	parser.add_argument('--size', type=int, default=128)  # 128固定
	parser.add_argument('--stride', type=int,   default=128)  # 128がよさげ
	parser.add_argument('--attr',metavar='N', type=int, nargs='+',   default=[0,1,0,0,1,0,0])
	#self.attr_keys = ['Male', 'Female', 'KizunaAI', 'Nekomasu', 'Mirai', 'Shiro', 'Kaguya']
	args = parser.parse_args()

	stride = args.stride
	wave_len = args.length
	size = args.size
	window = np.hanning(wave_len)
	all_size = ((wave_len - stride) + size * stride)
	power = 0.2
	scale = 1/18

	_, data = wav.read(args.path)

	datas = np.zeros(((int)(len(data)/all_size), size,size))

	for i in range((int)(len(data)/all_size)):
		fft_abs, _ = FFT(data, wave_len, stride, window,i * all_size, size, power, scale)
		datas[i] = fft_abs
	if(datas != np.zeros(((int)(len(data)/all_size), size,size))):
		saveZ_datas(datas,args.out_path,args.attr)
if __name__ == '__main__':
	main()