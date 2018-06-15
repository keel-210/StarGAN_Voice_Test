import numpy as np

def load_FFT_attr(data_dir):
    attr_list = np.zeros((len(data_dir)))
    for i in range(len(data_dir)):
        data = np.load(data_dir+str(i)+'.npz')
        attr_list[i] = data['wav_attr']
    return  attr_list