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

    plt.figure(1)
    ax1 = plt.subplot(411)
    plt.plot(audio[:fs])
    plt.title('original signal ')
    ax2 = plt.subplot(412)
    plt.plot(filtered_audio[:fs])
    plt.title('50Hz filter ')
    ax3 = plt.subplot(413)
    plt.plot(high_envelop_audio[:fs])
    plt.title('high envelop ')
    ax4 = plt.subplot(414)
    plt.plot(filtered_envelop[:fs])
    plt.title('5Hz filter ')
    plt.xlabel('time')
    plt.ylabel('mag')
    plt.tight_layout()

    return audio_pitch


def add_noise(data, snr):
    P_clean = np.sum(abs(data) ** 2) / len(data)
    noise = np.random.randn(len(data))
    P_noise = np.sum(abs(noise) ** 2) / len(data)
    noise_variance = P_clean / (math.pow(10., (snr / 10)))
    add_noise = np.sqrt(noise_variance / P_noise) * noise
    noise_ = np.transpose(np.tile(add_noise, (2, 1)))
    noisy = data + noise_
    return noisy


def norm(data):
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)

    data = add_noise(data, 5)
    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data


def add_bg(wave, bg, weight):
    n = len(wave)
    start = np.random.randint(len(bg) - n)
    bg_slice = bg[start: start + n]
    wav_with_bg = wave * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, weight)
    return wav_with_bg


def find_wave(data_path, data_root, scp_dir, noise_path, person,  scp_name='wave_total'):
    all_wave_path = []
    for i in data_path:
        all_wave_path += glob.glob(i)
    all_wave_path = sorted(all_wave_path)
    lines = ['' for _ in range(1)]
    waves_0 = ['' for _ in range(1)]
    waves_1 = ['' for _ in range(1)]
    waves_2 = ['' for _ in range(1)]

    # add chew noise
    chew_path = r"F:\OESense\data\Activity Recognition\S{}_Act_Chew.wav".format(person)
    chew, fs = sf.read(chew_path)
    chew_0 = ['' for _ in range(1)]
    chew_1 = ['' for _ in range(1)]


    # add Drink noise

    drink_path = r"F:\OESense\data\Activity Recognition\S{}_Act_Drink.wav".format(person)
    drink, fs = sf.read(drink_path)
    drink_0 = ['' for _ in range(1)]
    drink_1 = ['' for _ in range(1)]

    if not os.path.exists(noise_path):
        os.mkdir(noise_path)

    # don't select channel
    for wav_idx in range(len(all_wave_path)):
        line = all_wave_path[wav_idx]
        wave, fs = sf.read(line)
        wave = norm(wave)
        audio_pitch_0 = peakDetection(wave[:, 0], fs)
        audio_pitch_1 = peakDetection(wave[:, 1], fs)
        audio_pitch_2 = peakDetection(wave, fs)

        # label sum
        c0 = [1, 4, 6, 8]
        c1 = [3, 5, 7, 9]
        c2 = [10, 11, 12]
        c3 = [2]
        base_name = os.path.basename(line)
        wave_name = base_name.split('.')[0]
        label = wave_name.split('_')[-1]
        label = int(label)
        # person_rawLabel_newLabel
        if label in c0:
            base_name = 'S{}_Ges_{}_1.wav'.format(person, label)
        elif label in c1:
            base_name = 'S{}_Ges_{}_2.wav'.format(person, label)
        elif label in c2:
            base_name = 'S{}_Ges_{}_3.wav'.format(person, label)
        elif label in c3:
            base_name = 'S{}_Ges_{}_4.wav'.format(person, label)
        for i in range(len(audio_pitch_0)):
            # person_rawLabel_newLabel-Channel-pitchNum
            wave_name = base_name.replace('.wav', '-0-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_0[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            waves_0[0] += wave_name
        for i in range(len(audio_pitch_1)):
            wave_name = base_name.replace('.wav', '-1-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_1[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            waves_1[0] += wave_name
        for i in range(len(audio_pitch_2)):
            # wave_name = os.path.basename(line).replace('.wav', '-{}-{}.wav'.format(j, i))
            wave_name = base_name.replace('.wav', '-{}.wav'.format(i))
            wave_name = os.path.join(data_root, wave_name)
            wave_name += '\n'
            waves_2[0] += wave_name

        # save chew wave
        wave_chew_0 = add_bg(wave[:, 0], chew[:, 0], 0.1)
        wave_chew_1 = add_bg(wave[:, 1], chew[:, 1], 0.1)
        chew_pitch_0 = peakDetection(wave_chew_0, fs)
        chew_pitch_1 = peakDetection(wave_chew_1, fs)
        for i in range(len(chew_pitch_0)):
            # person_rawLabel_newLabel - Channel - Noise - pitchNum
            wave_name = base_name.replace('.wav', '-0-c-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_0[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            chew_0[0] += wave_name
        for i in range(len(chew_pitch_1)):
            wave_name = base_name.replace('.wav', '-1-c-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_1[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            chew_1[0] += wave_name

        # save drink wave
        wave_drink_0 = add_bg(wave[:, 0], drink[:, 0], 0.1)
        wave_drink_1 = add_bg(wave[:, 1], drink[:, 1], 0.1)
        drink_pitch_0 = peakDetection(wave_drink_0, fs)
        drink_pitch_1 = peakDetection(wave_drink_1, fs)
        for i in range(len(drink_pitch_0)):
            wave_name = base_name.replace('.wav', '-0-d-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_0[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            drink_0[0] += wave_name
        for i in range(len(drink_pitch_1)):
            wave_name = base_name.replace('.wav', '-1-d-{}.wav'.format(i))
            soundfile.write(os.path.join(noise_path, wave_name), audio_pitch_1[i], fs)
            wave_name = os.path.join(noise_path, wave_name)
            wave_name += '\n'
            drink_1[0] += wave_name

        line += '\n'
        lines[0] += line

    # total wave
    if not os.path.exists(scp_dir):
        os.mkdir(scp_dir)
    with open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as f:
        f.write(lines[0])

    # cut wave
    with open(os.path.join(scp_dir, 'wave_cut_1.scp'), 'w') as f:
        f.write(waves_0[0])
        f.write(waves_1[0])
        f.write(chew_0[0])
        f.write(chew_1[0])
        f.write(drink_0[0])
        f.write(drink_1[0])
    with open(os.path.join(scp_dir, 'wave_cut_2.scp'), 'w') as f:
        f.write(waves_2[0])

    return None


def dataset_cut(scp_path, wave_dir, channel, val_rate, person):
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
    with open(os.path.join(wave_dir, 'data_{}_train_{}'.format(person, channel)), 'w') as f:
        f.write(path_tr[0])
    with open(os.path.join(wave_dir, 'data_{}_val_{}'.format(person, channel)), 'w') as f:
        f.write(path_vl[0])
    return None


if __name__ == '__main__':
    data_root = r'F:\OESense\data\Gesture Recognition'
    person = 10
    data_path = [os.path.join(data_root, 'S{}_Ges_*.wav'.format(person))]

    noise_path = r"F:\OESense\data\Person{}".format(person)
    find_wave(data_path, data_root, 'scp_dir', noise_path, person, 'total')
    # scp_path_0 = r"F:\OESense\scp_dir\wave_cut_0.scp"
    # dataset_cut(scp_path_0, 'wave_dir', 0, val_rate=0.1)
    scp_path_1 = r"F:\OESense\scp_dir\wave_cut_1.scp"
    dataset_cut(scp_path_1, 'wave_dir', 1, val_rate=0.2, person=person)
    scp_path_2 = r"F:\OESense\scp_dir\wave_cut_2.scp"
    dataset_cut(scp_path_2, 'wave_dir', 2, val_rate=0.2, person=person)
