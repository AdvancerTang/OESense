import glob
import os.path
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import random

random.seed(1)

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


def norm(data):
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)

    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data


def find_wave(data_path, data_root, scp_dir, scp_name='wave_total'):
    all_wave_path = []
    for i in data_path:
        all_wave_path += glob.glob(i)
    all_wave_path = sorted(all_wave_path)
    lines = ['' for _ in range(1)]
    waves_0 = ['' for _ in range(1)]
    waves_1 = ['' for _ in range(1)]
    waves_2 = ['' for _ in range(1)]
    # for wav_idx in range(len(all_wave_path)):
    #     line = all_wave_path[wav_idx]
    #     wave, fs = sf.read(line)
    #     for j in range(2):
    #         audio_pitch = peakDetection(wave[:, j], fs)
    #         for i in range(len(audio_pitch)):
    #             wave_name = os.path.basename(line).replace('.wav', '-{}-{}.wav'.format(j, i))
    #             wave_name = os.path.join(data_root, wave_name)
    #             wave_name += '\n'
    #             waves[0] += wave_name
    #     line += '\n'
    #     lines[0] += line

    # don't select channel
    for wav_idx in range(len(all_wave_path)):
        line = all_wave_path[wav_idx]
        wave, fs = sf.read(line)
        wave = norm(wave)
        audio_pitch_0 = peakDetection(wave[:, 0], fs)
        audio_pitch_1 = peakDetection(wave[:, 1], fs)
        audio_pitch_2 = peakDetection(wave, fs)
        for i in range(len(audio_pitch_0)):
            # wave_name = os.path.basename(line).replace('.wav', '-{}-{}.wav'.format(j, i))
            wave_name = os.path.basename(line).replace('.wav', '-{}.wav'.format(i))
            wave_name = os.path.join(data_root, wave_name)
            wave_name += '\n'
            waves_0[0] += wave_name
        for i in range(len(audio_pitch_1)):
            # wave_name = os.path.basename(line).replace('.wav', '-{}-{}.wav'.format(j, i))
            wave_name = os.path.basename(line).replace('.wav', '-{}.wav'.format(i))
            wave_name = os.path.join(data_root, wave_name)
            wave_name += '\n'
            waves_1[0] += wave_name
        for i in range(len(audio_pitch_2)):
            # wave_name = os.path.basename(line).replace('.wav', '-{}-{}.wav'.format(j, i))
            wave_name = os.path.basename(line).replace('.wav', '-{}.wav'.format(i))
            wave_name = os.path.join(data_root, wave_name)
            wave_name += '\n'
            waves_2[0] += wave_name
        line += '\n'
        lines[0] += line

    # total wave
    if not os.path.exists(scp_dir):
        os.mkdir(scp_dir)
    with open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as f:
        f.write(lines[0])

    # cut wave
    with open(os.path.join(scp_dir, 'wave_cut_0.scp'), 'w') as f:
        f.write(waves_0[0])
    with open(os.path.join(scp_dir, 'wave_cut_1.scp'), 'w') as f:
        f.write(waves_1[0])
    with open(os.path.join(scp_dir, 'wave_cut_2.scp'), 'w') as f:
        f.write(waves_2[0])

    return None


def dataset_cut(scp_path, wave_dir, channel,  val_rate):
    with open(scp_path) as f:
        lines = f.readlines()
    files_scp = [line.strip() for line in lines]
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
    data_root = r'F:\OESense\data\Gesture Recognition'
    data_path = [os.path.join(data_root, 'S1_Ges_*.wav')]
    find_wave(data_path, data_root, 'scp_dir', 'total')
    scp_path_0 = r"F:\OESense\scp_dir\wave_cut_0.scp"
    dataset_cut(scp_path_0, 'wave_dir', 0, val_rate=0.1)
    scp_path_1 = r"F:\OESense\scp_dir\wave_cut_1.scp"
    dataset_cut(scp_path_1, 'wave_dir', 1, val_rate=0.1)
    scp_path_2 = r"F:\OESense\scp_dir\wave_cut_2.scp"
    dataset_cut(scp_path_2, 'wave_dir', 2, val_rate=0.1)
