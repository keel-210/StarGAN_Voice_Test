import numpy as np
import os

def load_FFT_attr(data_dir):
    attr_list = np.zeros((len(os.listdir(data_dir+'\\train\\')),7))
    for i in range(len(os.listdir(data_dir))):
        data = np.load(data_dir+'\\train\\'+str(i)+'.npz')
        attr_list[i] = data['attr']
    return  attr_list