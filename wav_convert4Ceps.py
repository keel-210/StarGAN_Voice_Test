import numpy as np
import os
import sys
import argparse
import scipy.io.wavfile as wav
import tqdm
import glob
import re
import pyworld
import pysptk


def saveZ_datas(datas, path, attribute):
    j = len(os.listdir(path))
    for i in range(len(datas)):
        np.savez_compressed(path+str('{0:06d}'.format(i + j))+'.npz', MCep=datas[i], attr=attribute)


def melCeps(wav, length, alpha, order):
    sys.exit(0)


def convert(wav, attr, alpha, order):
    melCeps(wav, length, alpha, order)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str,   default='./wav/*')
    parser.add_argument('--out_path', type=str, default='./data_mcep/')
    parser.add_argument('--length', type=int, default=14000)
    args = parser.parse_args()

    wave_len = args.length
    window = np.hanning(wave_len)

    childs = glob.glob(args.path)
    print(childs)
    nums = [re.findall(r'\d{7}', c) for c in childs]
    attrs = [[int(c) for c in str(n[0])] for n in nums]
    sys.exit(0)


if __name__ == '__main__':
    main()
