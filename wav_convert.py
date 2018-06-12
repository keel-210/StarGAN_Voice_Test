import numpy as np
import os
import random
import scipy.io.wavfile as wav
import argparse
import cv2

def save(path, bps, data): #np.arrayからpathの位置にwavを書く
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    data = np.reshape(data, -1)
    wav.write(path, bps, data)

def image_single_split_pad(src, side, pos, power, scale, window): #スペクトログラムのデータ処理
    wave_len = side*2 - 2
    spl = np.array([src[p:p+wave_len]*window for p in range(pos, pos+side*side, side)])
    spl = np.fft.fft(spl, axis=1)
    spl = spl[:,:side]
    spl = np.abs([spl], dtype=np.float32)
    spl = _pow_scale(spl, power)
    spl *= scale
    return spl

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p)

def overwrap(fft, length, dif, side):
    dst = np.zeros(dif * (side-1)+length, dtype=float)
    for i, f in enumerate(fft):
        dst[i*dif:i*dif+length] += np.fft.ifft(f).real
    return dst

def FFT(wav, length, stride, window, pos, size, power, scale):
	wave_len = length
	data_fft = np.array([wav[p:p+wave_len, 0]*window for p in range(pos,pos + (wave_len - stride) + size * stride, stride)])
	print(np.shape(data_fft))
	data_fft = np.fft.fft(data_fft, axis=1)
	fft_phase = np.arctan2(data_fft.imag, data_fft.real)
	data_fft = data_fft[:, :size]
	data_fft = np.abs([data_fft], dtype='f4')
	data_fft = _pow_scale(data_fft, power)
	data_fft *= scale

	return data_fft, fft_phase

def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--path', type=str,   default='./test.wav')
	parser.add_argument('--out_path', type=str, default='./FFT/')
	parser.add_argument('--length', type=int, default=254)  # 254固定
	parser.add_argument('--size', type=int, default=128)  # 128固定
	parser.add_argument('--stride', type=int,   default=128)  # 128がよさげ
	args = parser.parse_args()

	stride = args.stride
	wave_len = args.length
	size = args.size
	window = np.hanning(wave_len)
	all_size = ((wave_len - stride) + size * stride)
	power = 0.2
	scale = 1/18

	wav_bps, data = wav.read(args.path)

	datas = np.zeros((100, ((args.length - stride) + size * stride)))

	for i in range(10):
		fft_abs, phase = FFT(data, wave_len, stride, window,i * all_size, size, power, scale)
		cv2.imwrite(args.out_path+str(i)+'.png',fft_abs[0]*256)

if __name__ == '__main__':
	main()