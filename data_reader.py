import os.path
import torch
from feature_extract import  featureExtract
# from feature_extract import time_featureExtract as featureExtract
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import numpy as np
import soundfile as sf


def peakDetection(audio, fs):
    # 50Hz filter
    order1 = 2
    bn1, an1 = signal.butter(order1, 50, btype='lowpass', fs=fs)
    filtered_audio = signal.lfilter(bn1, an1, audio)

    # up_evlp extract
    high_envelop = signal.hilbert(filtered_audio)
    high_envelop_audio = np.abs(high_envelop)

    # 5Hz filter
    order2 = 2
    bn2, an2 = signal.butter(order2, 5, btype='lowpass', fs=fs)
    filtered_envelop = signal.lfilter(bn2, an2, high_envelop_audio)

    # PeakDetection
    peak_posotion = []
    peak_value = []
    audio_pitch = []
    period = fs
    pitch = int(len(audio) / period)
    for i in range(pitch):
        idx = period * i + np.argmax(filtered_envelop[i * period: (i + 1) * period])
        begin = idx - 0.25 * fs
        end = idx + 0.15 * fs
        if begin > 0 and end < len(filtered_audio):
            peak_posotion.append(idx)
            peak_value.append(filtered_envelop[idx])
            audio_pitch.append(filtered_audio[int(idx - 0.25 * fs): int(idx + 0.15 * fs)])

    return audio_pitch


class myDataset(Dataset):
    def __init__(self, scp_path, channel, feature):
        self.scp_path = scp_path
        self.channel = channel
        self.feature = feature
        with open(self.scp_path) as f:
            lines = f.readlines()
        self.files_scp = [line.strip() for line in lines]
        self.FeaExt = featureExtract(peakDetection, self.feature)

    def __len__(self):
        return len(self.files_scp)

    def __getitem__(self, idx):
        cur_idx = idx

        audio_path = self.files_scp[cur_idx]
        audio_name = os.path.basename(audio_path)
        audio_pitch_idx = (audio_name.split('.')[0]).split('-')[-1]
        # audio_pitch_channel = (audio_name.split('.')[0]).split('-')[-2]
        total_audio_path = audio_path.split('-')[0] + '.wav'
        wave, fs = sf.read(total_audio_path)
        wave = self.norm(wave)


        # if self.channel == audio_pitch_channel == 0:
        if self.channel == 0:
            wave_spectrum = self.FeaExt.forward(wave[:, 0])
        elif self.channel == 1:
            wave_spectrum = self.FeaExt.forward(wave[:, 1])
        elif self.channel == 2:
            wave_spectrum_l = self.FeaExt.forward(wave[:, 0])
            wave_spectrum_r = self.FeaExt.forward(wave[:, 1])
            wave_spectrum = np.vstack((wave_spectrum_l, wave_spectrum_r))

        wave_name = total_audio_path
        wave_name = wave_name.split('.')[0]
        label = wave_name.split('_')[-1]
        label = float((int(label) - 1))
        label = np.float32(label)
        wave_spectrum = np.array(wave_spectrum, dtype='float32')

        if self.channel == 0 or self.channel == 1:
            selected_wave = np.squeeze(wave_spectrum[int(audio_pitch_idx)])
        elif self.channel == 2:
            selected_wave_l = np.squeeze(wave_spectrum[int(audio_pitch_idx)])
            selected_wave_r = np.squeeze(wave_spectrum[int(audio_pitch_idx) + len(audio_pitch_idx[0])])
            selected_wave = np.concatenate((selected_wave_l, selected_wave_r), axis=0)


        return selected_wave, label

    def norm(self, data):
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)

        norm_data = (data - min_val) / (max_val - min_val)
        return norm_data

def time_CollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    # data_feature = [torch.from_numpy(x[0]) for x in sample_batch]
    # data_label = torch.tensor([x[1] for x in sample_batch]).unsqueeze(-1)
    data_feature = [x[0] for x in sample_batch]
    data_label = torch.tensor([x[1] for x in sample_batch]).unsqueeze(-1)
    return data_feature, data_label.numpy()


def stft_CollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    feature_seq = [torch.from_numpy(x[0]) for x in sample_batch]
    data_label = torch.tensor([x[1] for x in sample_batch]).long().unsqueeze(-1)
    data_feature = torch.stack(feature_seq, dim=0)
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
