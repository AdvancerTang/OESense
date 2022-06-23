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
    # plt.show()

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
    m = len(bg)
    if m < n :
        start = np.random.randint(n - m)
        wave = wave[start: start + m]
    elif m > n:
        start = np.random.randint(m - n)
        bg = bg[start: start + n]
    wav_with_bg = wave * np.random.uniform(0.8, 1.2) + bg * np.random.uniform(0, weight)
    return wav_with_bg


def find_wave(data_path, data_root, scp_dir, noise_path, person, scp_name='wave_total'):
    all_wave_path = []
    for i in data_path:
        all_wave_path += glob.glob(i)
    all_wave_path = sorted(all_wave_path)
    lines = ['' for _ in range(1)]
    waves_0 = ['' for _ in range(1)]
    waves_1 = ['' for _ in range(1)]

    # add left distribute
    l_0 = ['' for _ in range(1)]
    l_1 = ['' for _ in range(1)]

    # add right distribute
    r_0 = ['' for _ in range(1)]
    r_1 = ['' for _ in range(1)]

    # add down distribute
    d_0 = ['' for _ in range(1)]
    d_1 = ['' for _ in range(1)]

    if not os.path.exists(noise_path):
        os.mkdir(noise_path)

    # don't select channel
    for wav_idx in range(len(all_wave_path)):
        line = all_wave_path[wav_idx]
        wave, fs = sf.read(line)
        wave = norm(wave)
        audio_pitch_0 = peakDetection(wave[:, 0], fs)
        audio_pitch_1 = peakDetection(wave[:, 1], fs)

        # label: 3class [2.middle forehead, 4.left temple, 5.right temple, 12.chin]
        c0 = [1, 6, 8]  # [1.left forehead, 6. left cheek, 8. left jaw]
        c1 = [3, 7, 9]  # [3. right forehead, 7.right cheek, 9. right jaw]
        c2 = [10, 11]   # [10.nose, 11. philtrum]
        base_name = os.path.basename(line)
        wave_name = base_name.split('.')[0]
        label = wave_name.split('_')[-1]
        label = int(label)
        # person_rawLabel_newLabel
        if label == 4:
            base_name = 'S{}_Ges_{}_1.wav'.format(person, label)
        elif label == 5:
            base_name = 'S{}_Ges_{}_2.wav'.format(person, label)
        elif label == 12:
            base_name = 'S{}_Ges_{}_3.wav'.format(person, label)
        elif label == 2:
            base_name = 'S{}_Ges_{}_4.wav'.format(person, label)
        # left distribute
        elif label in c0:
            label_name = 'S{}_Ges_4.wav'.format(person)
            way = os.path.join(data_root, label_name)
            wave_tar, fs = sf.read(way)
            wave_tar = norm(wave_tar)
            base_name = 'S{}_Ges_{}_1.wav'.format(person, label)
            # distribute, fs = sf.read(line)
            distribute = wave
            distri_0 = add_bg(wave_tar[:, 0], distribute[:, 0], 0.1)
            distri_1 = add_bg(wave_tar[:, 1], distribute[:, 1], 0.1)
            distri_l_0 = peakDetection(distri_0, fs)
            distri_l_1 = peakDetection(distri_1, fs)
            for i in range(len(distri_l_0)):
                # person_rawLabel_newLabel - Channel - Noise - pitchNum
                wave_name = base_name.replace('.wav', '-0-l-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_l_0[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                l_0[0] += wave_name
            for i in range(len(distri_l_1)):
                # person_rawLabel_newLabel - Channel - Noise - pitchNum
                wave_name = base_name.replace('.wav', '-1-l-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_l_1[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                l_1[0] += wave_name
        # right distribute
        elif label in c1:
            label_name = 'S{}_Ges_5.wav'.format(person)
            way = os.path.join(data_root, label_name)
            wave_tar, fs = sf.read(way)
            wave_tar = norm(wave_tar)
            # distribute, fs = sf.read(line)
            distribute = wave
            distri_0 = add_bg(wave_tar[:, 0], distribute[:, 0], 0.1)
            distri_1 = add_bg(wave_tar[:, 1], distribute[:, 1], 0.1)
            distri_r_0 = peakDetection(distri_0, fs)
            distri_r_1 = peakDetection(distri_1, fs)
            base_name = 'S{}_Ges_{}_2.wav'.format(person, label)
            for i in range(len(distri_r_0)):
                wave_name = base_name.replace('.wav', '-0-r-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_r_0[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                r_0[0] += wave_name
            for i in range(len(distri_r_1)):
                wave_name = base_name.replace('.wav', '-1-r-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_r_1[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                r_1[0] += wave_name
        elif label in c2:
            label_name = 'S{}_Ges_12.wav'.format(person)
            way = os.path.join(data_root, label_name)
            wave_tar, fs = sf.read(way)
            wave_tar = norm(wave_tar)
            # distribute, fs = sf.read(line)
            distribute = wave
            distri_0 = add_bg(wave_tar[:, 0], distribute[:, 0], 0.5)
            distri_1 = add_bg(wave_tar[:, 1], distribute[:, 1], 0.5)
            distri_d_0 = peakDetection(distri_0, fs)
            distri_d_1 = peakDetection(distri_1, fs)
            base_name = 'S{}_Ges_{}_3.wav'.format(person, label)
            for i in range(len(distri_d_0)):
                wave_name = base_name.replace('.wav', '-0-d-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_d_0[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                d_0[0] += wave_name
            for i in range(len(distri_d_1)):
                wave_name = base_name.replace('.wav', '-1-d-{}.wav'.format(i))
                soundfile.write(os.path.join(noise_path, wave_name), distri_d_1[i], fs)
                wave_name = os.path.join(noise_path, wave_name)
                wave_name += '\n'
                d_1[0] += wave_name

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
        f.write(l_0[0])
        f.write(l_1[0])
        f.write(r_0[0])
        f.write(r_1[0])
        f.write(d_0[0])
        f.write(d_1[0])

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
    person = 1
    data_path = [os.path.join(data_root, 'S{}_Ges_*.wav'.format(person))]
    noise_path = r"F:\OESense\data\Person{}".format(person)
    find_wave(data_path, data_root, 'scp_dir', noise_path, person, 'total')
    # scp_path_0 = r"F:\OESense\scp_dir\wave_cut_0.scp"
    # dataset_cut(scp_path_0, 'wave_dir', 0, val_rate=0.1)
    scp_path_1 = r"F:\OESense\scp_dir\wave_cut_1.scp"
    dataset_cut(scp_path_1, 'wave_dir', 1, val_rate=0.2, person=person)

    # data_root = r'F:\OESense\data\Activity Recognition'
    # data_name = 'S1_Act_Walk.wav'
    # sound, fs = sf.read(os.path.join(data_root, data_name))
    # pitch = peakDetection(sound, fs)
    # te_path = r'F:\OESense\data\Activity Recognition\sth'
    # wave = 'walk.wav'
    # soundfile.write(os.path.join(te_path, wave), pitch[0], fs)


