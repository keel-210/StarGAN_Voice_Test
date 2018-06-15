import os
from glob import glob
from collections import namedtuple
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt


def load_data_list(data_dir):
    path = os.path.join(data_dir, 'train', '*')
    file_list = glob(path)
    return file_list


def attr_extract(data_dir):
    attr_list = {}

    path = os.path.join(data_dir, 'list_attr_celeba.txt')
    file = open(path, 'r')

    n = file.readline()
    n = int(n.split('\n')[0])  # of celebA img: 202599

    attr_line = file.readline()
    attr_names = attr_line.split('\n')[0].split()  # attribute name

    for line in file:
        row = line.split('\n')[0].split()
        img_name = row.pop(0)
        row = [int(val) for val in row]
#    img = img[..., ::-1] # bgr to rgb
        attr_list[img_name] = row

    file.close()
    return attr_names, attr_list
