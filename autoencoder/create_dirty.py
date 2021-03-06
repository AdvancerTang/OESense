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
    # pitch = 20
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
    if len(bg) - n == 0:
        start = 0
    else:
        start = np.random.randint(len(bg) - n)
    bg_slice = bg[start: start + n]
    wav_with_bg = wave * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, weight)
    return wav_with_bg


def find_wave(data_root, scp_dir, persons, scp_name='wave_total', mod='train'):
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

    # add down distribute
    up_0 = ['' for _ in range(1)]
    up_1 = ['' for _ in range(1)]

    for person in persons:
        data_path = [os.path.join(data_root, 'S{}_Ges_*.wav'.format(person))]
        noise_path = r"F:\OESense\autoencoder\dirty_data\Person{}".format(person)
        all_wave_path = []
        for i in data_path:
            all_wave_path += glob.glob(i)
        all_wave_path = sorted(all_wave_path)

        if not os.path.exists(noise_path):
            os.mkdir(noise_path)

        # don't select channel
        for wav_idx in range(len(all_wave_path)):
            line = all_wave_path[wav_idx]
            wave, fs = sf.read(line)
            wave = norm(wave)

            # label: 3class [2.middle forehead, 4.left temple, 5.right temple, 12.chin]
            c0 = [1, 4, 6, 8]  # [1.left forehead, 6. left cheek, 8. left jaw]
            c1 = [3, 5, 7, 9]  # [3. right forehead, 7.right cheek, 9. right jaw]
            c2 = [2]  # [2 .forehead]
            c3 = [12] # [12. chin]
            star = [2, 4, 5, 12]
            base_name = os.path.basename(line)
            wave_name = base_name.split('.')[0]
            label = wave_name.split('_')[-1]
            label = int(label)
            if label in star:
                if label == 4:
                    base_name = 'S{}_Ges_{}_1.wav'.format(person, label)
                elif label == 5:
                    base_name = 'S{}_Ges_{}_2.wav'.format(person, label)
                elif label == 12:
                    base_name = 'S{}_Ges_{}_4.wav'.format(person, label)
                elif label == 2:
                    base_name = 'S{}_Ges_{}_3.wav'.format(person, label)
                distribute, fs = sf.read(line)
                distri_0 = add_bg(wave[:, 0], distribute[:, 0], 0.1)
                distri_1 = add_bg(wave[:, 1], distribute[:, 1], 0.1)
                distri_0_ = peakDetection(distri_0, fs)
                distri_1_ = peakDetection(distri_1, fs)
                distri_0_ = distri_0_[:21]
                distri_1_ = distri_1_[:21]
                for i in range(len(distri_0_)):
                    wave_name = base_name.replace('.wav', '-0-{}.wav'.format(i))
                    soundfile.write(os.path.join(noise_path, wave_name), distri_0_[i], fs)
                    wave_name = os.path.join(noise_path, wave_name)
                    wave_name += '\n'
                    waves_0[0] += wave_name
                for i in range(len(distri_1_)):
                    wave_name = base_name.replace('.wav', '-1-{}.wav'.format(i))
                    soundfile.write(os.path.join(noise_path, wave_name), distri_1_[i], fs)
                    wave_name = os.path.join(noise_path, wave_name)
                    wave_name += '\n'
                    waves_1[0] += wave_name
            # person_rawLabel_newLabel
            if label in c2:
                base_name = 'S{}_Ges_{}_3.wav'.format(person, label, label)
                distribute, fs = sf.read(line)
                distri_0 = add_bg(wave[:, 0], distribute[:, 0], 0.1)
                distri_1 = add_bg(wave[:, 1], distribute[:, 1], 0.1)
                distri_u_0 = peakDetection(distri_0, fs)
                distri_u_1 = peakDetection(distri_1, fs)
                distri_u_0 = distri_u_0[:21]
                distri_u_1 = distri_u_1[:21]
                for i in range(len(distri_u_0)):
                    # person_rawLabel_newLabel - Channel - Noise - pitchNum
                    wave_name = base_name.replace('.wav', '-0-u-{}.wav'.format(i))
                    soundfile.write(os.path.join(noise_path, wave_name), distri_u_0[i], fs)
                    wave_name = os.path.join(noise_path, wave_name)
                    wave_name += '\n'
                    up_0[0] += wave_name
                for i in range(len(distri_u_1)):
                    # person_rawLabel_newLabel - Channel - Noise - pitchNum
                    wave_name = base_name.replace('.wav', '-1-u-{}.wav'.format(i))
                    soundfile.write(os.path.join(noise_path, wave_name), distri_u_1[i], fs)
                    wave_name = os.path.join(noise_path, wave_name)
                    wave_name += '\n'
                    up_1[0] += wave_name
            elif label in c3:
                base_name = 'S{}_Ges_{}_4.wav'.format(person, label)
                distribute, fs = sf.read(line)
                distri_0 = add_bg(wave[:, 0], distribute[:, 0], 0.1)
                distri_1 = add_bg(wave[:, 1], distribute[:, 1], 0.1)
                distri_d_0 = peakDetection(distri_0, fs)
                distri_d_1 = peakDetection(distri_1, fs)
                distri_d_0 = distri_d_0[:21]
                distri_d_1 = distri_d_1[:21]
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
            # left distribute
            elif label in c0:
                base_name = 'S{}_Ges_{}_1.wav'.format(person, label)
                distribute, fs = sf.read(line)
                distri_0 = add_bg(wave[:, 0], distribute[:, 0], 0.1)
                distri_1 = add_bg(wave[:, 1], distribute[:, 1], 0.1)
                distri_l_0 = peakDetection(distri_0, fs)
                distri_l_1 = peakDetection(distri_1, fs)
                distri_l_0 = distri_l_0[:21]
                distri_l_1 = distri_l_1[:21]
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
                base_name = 'S{}_Ges_{}_2.wav'.format(person, label)
                distribute, fs = sf.read(line)
                distri_0 = add_bg(wave[:, 0], distribute[:, 0], 0.1)
                distri_1 = add_bg(wave[:, 1], distribute[:, 1], 0.1)
                distri_r_0 = peakDetection(distri_0, fs)
                distri_r_1 = peakDetection(distri_1, fs)
                distri_r_0 = distri_r_0[:21]
                distri_r_1 = distri_r_1[:21]
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

            line += '\n'
            lines[0] += line

    # total wave
    if not os.path.exists(scp_dir):
        os.mkdir(scp_dir)
    with open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as f:
        f.write(lines[0])

    # cut wave
    with open(os.path.join(scp_dir, 'label_{}.scp'.format(mod)), 'w') as f:
        f.write(waves_0[0])
        f.write(waves_1[0])
    with open(os.path.join(scp_dir, 'data_{}.scp'.format(mod)), 'w') as f:
        f.write(l_0[0])
        f.write(l_1[0])
        f.write(r_0[0])
        f.write(r_1[0])
        f.write(d_0[0])
        f.write(d_1[0])
        f.write(up_0[0])
        f.write(up_1[0])

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
    # persons = [i for i in range(3, 30) if i != 6]
    persons = [3, 4]
    find_wave(data_root, 'scp1_dir', persons, 'total', 'train')
    evl_p = [1]
    find_wave(data_root, 'scp1_dir', evl_p, 'total', 'evl')
    # scp_path_0 = r"F:\OESense\scp_dir\wave_cut_0.scp"
    # dataset_cut(scp_path_0, 'wave_dir', 0, val_rate=0.1)
    # scp_path_1 = r"F:\OESense\autoencoder\scp_dir\wave_cut_1.scp"
    # dataset_cut(scp_path_1, 'wave_dir', 1, val_rate=0.2, person=person)
