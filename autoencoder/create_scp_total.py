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

def total_data(persons, total_wav_dir):
    total_path = ['' for _ in range(1)]
    for person in persons:
        single_train_path = r"F:\OESense\autoencoder\wave_dir\data_{}_train_1".format(person)
        single_val_path = r"F:\OESense\autoencoder\wave_dir\data_{}_val_1".format(person)
        with open(single_train_path) as f:
            lines_train = f.readlines()
        with open(single_val_path) as f:
            lines_val = f.readlines()
        files_train = [line.strip() for line in lines_train]
        files_val = [line.strip() for line in lines_val]
        for file in files_train:
            file += '\n'
            total_path[0] += file
        for file in files_val:
            file += '\n'
            total_path[0] += file
    if not os.path.exists(total_wav_dir):
        os.mkdir(total_wav_dir)
    with open(os.path.join(total_wav_dir, 'total_data'), 'w') as f:
        f.write(total_path[0])


def dataset_cut(scp_path, wave_dir, channel, val_rate):
    with open(scp_path) as f:
        lines = f.readlines()
    files_scp = [line.strip() for line in lines]
    # select 400 person to test the code
    nums_vl = int(len(files_scp) * val_rate)
    nums_tr = len(files_scp) - nums_vl
    a = np.arange(0, len(files_scp), 1)
    vl = []
    for _ in range(nums_vl):
        vl.append(random.choice(a))

    path_tr = ['' for _ in range(1)]
    path_vl = ['' for _ in range(1)]
    for i in range(len(files_scp)):
        if i in vl:
            vl_name = files_scp[i]
            vl_name += '\n'
            path_vl[0] += vl_name
        else:
            tr_name = files_scp[i]
            tr_name += '\n'
            path_tr[0] += tr_name

    if not os.path.exists(wave_dir):
        os.mkdir(wave_dir)
    with open(os.path.join(wave_dir, 'data_train_{}'.format(channel)), 'w') as f:
        f.write(path_tr[0])
    with open(os.path.join(wave_dir, 'data_val_{}'.format(channel)), 'w') as f:
        f.write(path_vl[0])
    return None

if __name__ == '__main__':
    # data_root = r'F:\OESense\data\Gesture Recognition'
    persons = [i for i in range(3, 32) if i != 6]
    # total_data(persons, 'total_dir')
    total_data([1], 'total_dir')
    scp_path = r'F:\OESense\autoencoder\total_dir\total_data'
    dataset_cut(scp_path, 'total_dir', 2, 0.1)

