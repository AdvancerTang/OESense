import audiomentations
import IPython.display as ipd
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os

def add_bg(wave, bg, weight):
    n = len(wave)
    start = np.random.randint(len(bg) - n)
    bg_slice = bg[start: start + n]
    wav_with_bg = wave * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, weight)
    return wav_with_bg


def add_noise(data, snr):
    P_clean = np.sum(abs(data) ** 2) / len(data)
    noise = np.random.randn(len(data))
    P_noise = np.sum(abs(noise) ** 2) / len(data)
    noise_variance = P_clean / (math.pow(10., (snr / 10)))
    add_noise = np.sqrt(noise_variance / P_noise) * noise
    noise_ = np.transpose(np.tile(add_noise, (2, 1)))
    noisy = data + noise_
    return noisy


if __name__ == '__main__':
    data_path = r'F:\OESense\data\Gesture Recognition\S1_Ges_1.wav'
    wave, fs = sf.read(data_path)
    clean_stft = librosa.stft(wave[:, 0], n_fft=256, hop_length=40, window='hamming')
    clean = librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max, amin=0.001, top_db=120)
    plt.figure(1)
    librosa.display.specshow(clean, x_axis='time', y_axis='hz', sr=fs, hop_length=40, cmap='gnuplot2')
    plt.colorbar()

    # add noise
    noisy = add_noise(wave, 5)
    sf.write(os.path.join(r"F:\OESense\data", 'noise_wave.wav'), noisy, fs)
    noise_stft = librosa.stft(noisy[:, 0], n_fft=256, hop_length=40, window='hamming')
    noisy_ = librosa.amplitude_to_db(np.abs(noise_stft), ref=np.max, amin=0.001, top_db=120)
    plt.figure(2)
    librosa.display.specshow(noisy_, x_axis='time', y_axis='hz', sr=fs, hop_length=40, cmap='gnuplot2')
    plt.colorbar()

    # add chew bg
    chew_path = r"F:\OESense\data\Activity Recognition\S1_Act_Drink.wav"
    chew, fs = sf.read(chew_path)
    wave_addChew = add_bg(noisy[:, 0], chew[:, 0], 0.01)
    sf.write(os.path.join(r"F:\OESense\data", 'drink_wave.wav'), wave_addChew, fs)
    plt.figure(3)
    chew_stft = librosa.stft(wave_addChew, n_fft=256, hop_length=40, window='hamming')
    chew_ = librosa.amplitude_to_db(np.abs(chew_stft), ref=np.max, amin=0.001, top_db=120)
    librosa.display.specshow(chew_, x_axis='time', y_axis='hz', sr=fs)
    plt.colorbar()
    plt.show()
