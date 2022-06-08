import glob
import os.path
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import soundfile as sf
import random
import math

random.seed(1)

def total_data(persons, total_wav_dir, data_name):
    total_path = ['' for _ in range(1)]
    for person in persons:
        single_train_path = r"F:\OESense\wave_dir\data_{}_{}_1".format(person, data_name)
        with open(single_train_path) as f:
            lines = f.readlines()
        files_scp = [line.strip() for line in lines]
        for file in files_scp:
            file += '\n'
            total_path[0] += file
    if not os.path.exists(total_wav_dir):
        os.mkdir(total_wav_dir)
    with open(os.path.join(total_wav_dir, 'total_data_{}'.format(data_name)), 'w') as f:
        f.write(total_path[0])


if __name__ == '__main__':
    data_root = r'F:\OESense\data\Gesture Recognition'
    persons = [3, 4, 5, 7, 10]
    total_data(persons, 'total_dir', 'train')
    total_data(persons, 'total_dir', 'val')
