import os.path
import torch
from feature_extract import featureExtract
# from feature_extract import time_featureExtract as featureExtract
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import numpy as np
import soundfile as sf
import math
import random
random.seed(1)


class myDataset(Dataset):
    def __init__(self, data_path, channel, feature):
        self.data_path = data_path
        self.channel = channel
        self.feature = feature
        with open(self.data_path) as f:
            lines = f.readlines()
        self.files_data = [line.strip() for line in lines]

        self.FeaExt = featureExtract(self.feature)

    def __len__(self):
        return len(self.files_data)

    def __getitem__(self, idx):
        cur_idx = idx

        audio_path = self.files_data[cur_idx]
        wave, fs = sf.read(audio_path)
        wave_spectrum = self.FeaExt.forward(wave)
        wave_spectrum = np.array(wave_spectrum, dtype='float32')

        c0 = [1, 4, 6, 8]  # [1.left forehead, 6. left cheek, 8. left jaw]
        c1 = [3, 5, 7, 9]  # [3. right forehead, 7.right cheek, 9. right jaw]


        path_name = os.path.split(audio_path)[0]
        wave_name = os.path.basename(audio_path)
        idx = wave_name.split('-')[-1]
        base = wave_name.split('-')[0]
        base_sp = base.split('_')
        pre_label = int(base_sp[-2])
        if pre_label in c0:
            pre_label = 4
        if pre_label in c1:
            pre_label = 5
        base_sp[-2] = str(pre_label)
        base = '_'.join(base_sp)
        Ch = wave_name.split('-')[1]
        lab = base +'-' + Ch + '-' + idx
        lab = str(lab)
        label_path = os.path.join(path_name, lab)
        label, fs = sf.read(label_path)
        label_spectrum = self.FeaExt.forward(label)
        label_spectrum = np.array(label_spectrum, dtype='float32')

        return wave_spectrum, label_spectrum


def time_CollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    data_feature = [x[0] for x in sample_batch]
    data_label = torch.tensor([x[1] for x in sample_batch]).unsqueeze(-1)
    return data_feature, data_label.numpy()


def stft_CollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    feature_seq = [torch.from_numpy(x[0]) for x in sample_batch]
    label_seq = [torch.from_numpy(x[1]) for x in sample_batch]
    data_feature = torch.stack(feature_seq, dim=0)
    data_label = torch.stack(label_seq, dim=0)
    return data_feature, data_label


class myDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataloader, self).__init__(*args, **kwargs)
        # self.collate_fn = time_CollateFn
        self.collate_fn = stft_CollateFn


if __name__ == '__main__':
    scp_path = r"F:\OESense\wave_dir\data_train_0"
    channel = 2
    dtst = myDataset(scp_path, channel, 'mel')