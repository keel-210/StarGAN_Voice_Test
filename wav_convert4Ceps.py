import numpy as np
import os
import sys
import argparse
import scipy.io.wavfile as wav
from tqdm import tqdm
import glob
import re
import pyworld as pw
import pysptk
import matplotlib.pyplot as plt


def saveZ_datas(datas, path, attribute):
    j = len(os.listdir(path))
    for i in tqdm(range(len(datas))):
        np.savez_compressed(path+str('{0:06d}'.format(i + j))+'.npz', MCep=datas[i], attr=attribute)


def melCeps(wav,bps, length, alpha, order):
    _, sp, _ = pw.wav2world(wav.copy(order='C').astype('float64'), bps)
    ceps = pysptk.sp2mc(sp, order, alpha)
    return ceps


def convert(wav,bps,length ,attr, alpha, order,out_path):
    datas = [wav[i:i+length,0] for i in range(0,len(wav),length)]
    print(np.shape(datas))
    Cepss = np.array([len(datas),64,64])
    for d in tqdm(datas):
        c = melCeps(d,bps, length, alpha, order)
        for cep in c:
            for i,Scep in enumerate(cep):
                if(i==0):
                    Scep = (Scep+20)/28
                else:
                    Scep = (Scep+3)/7
                cep[i] = Scep
        np.append(Cepss,c)
    saveZ_datas(datas,out_path,attr)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str,   default='./wav/*')
    parser.add_argument('--out_path', type=str, default='./data_mcep/')
    parser.add_argument('--length', type=int, default=14000)
    parser.add_argument('--alpha', type=float, default=0.48)
    parser.add_argument('--order', type=float, default=63)
    args = parser.parse_args()

    childs = glob.glob(args.path)
    #childs = [args.path]
    print(childs)
    nums = [re.findall(r'[0~1]{7}', c) for c in childs]
    print(nums)
    attrs = [[int(c) for c in str(n[0])] for n in nums]
    print(attrs)
    for (c,attr) in zip(childs,attrs):
        bps, data = wav.read(c)
        convert(data,bps,args.length,attr,args.alpha,args.order,args.out_path)


if __name__ == '__main__':
    main()
