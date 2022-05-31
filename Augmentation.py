import audiomentations
import IPython.display as ipd
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math


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

    noisy = add_noise(wave, 5)
    noise_stft = librosa.stft(noisy, n_fft=256, hop_length=40, window='hamming')
    noisy = librosa.amplitude_to_db(np.abs(noise_stft), ref=np.max, amin=0.001, top_db=120)
    plt.figure(2)
    librosa.display.specshow(noisy, x_axis='time', y_axis='hz', sr=fs, hop_length=40, cmap='gnuplot2')
    plt.colorbar()
    plt.show()
