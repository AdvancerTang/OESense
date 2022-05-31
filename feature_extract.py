import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
import os
import numpy as np

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def peakDetection(audio, fs):
    # 50Hz filter
    order1 = 2
    bn1, an1 = signal.butter(order1, 50, btype='lowpass', fs=fs)
    filtered_audio = signal.lfilter(bn1, an1, audio)
    # sf.write('clean_3_l.wav', filtered_audio[:, 0], fs)
    # sf.write('clean_3_r.wav', filtered_audio[:, 1], fs)

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

    # return peak_posotion, peak_value
    return audio_pitch

class featureExtract(object):
    def __init__(self, feature=None, n_fft=256, hop_length=40, fs=4000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fs = fs
        self.feature = feature

    def forward(self, x):
        # STFT
        wave_spectrum = 0
        if self.feature == 'stft':
            wave_spectrum = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window='hamming')

        # Mel spectrum
        elif self.feature == 'mel':
            # mel_spect = librosa.feature.melspectrogram(wave, sr=self.fs, n_fft=self.n_fft,
            # hop_length=self.hop_length, n_mels=40)
            wave_spectrum = librosa.feature.melspectrogram(x, sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length)
        wave_spectrum = np.array(wave_spectrum)
        return wave_spectrum

class time_featureExtract(object):
    def __init__(self, peakDetection=None, channel=None, fs=4000):
        self.fs = fs
        self.peakDetection = peakDetection
        self.channel = channel

    def forward(self, x):
        peakWave = self.peakDetection(x, self.fs)
        waves = []
        if self.channel == 2:
            for wave in peakWave:
                audio = np.concatenate((wave[:, 0], wave[:, 1]), axis=0)
                waves.append(np.abs(audio))
        else:
            for wave in peakWave:
                waves.append(wave)
        # wave_spectrum = np.array(wave_spectrum)
        return waves

if __name__ == '__main__':
    data_path = r"F:\OESense\data\Person1"
    data_name = 'S1_Ges_1-0-0.wav'
    audio, fs = sf.read(os.path.join(data_path, data_name))
    plt.figure(1)
    plt.plot(audio)
    plt.title('original signal ')
    # channel_l = audio[:, 0]
    # channel_r = audio[:, 1]
    # audio_pitch = peakDetection(channel_l, fs)
    # audio_pitch = np.array(audio_pitch)
    # print(audio_pitch.shape)
    # feature = time_featureExtract(peakDetection)
    feature = featureExtract('mel')
    spectrum = feature.forward(audio)
    mel_spectrum = np.array(spectrum)
    print(mel_spectrum.shape)

    audio_stft = librosa.stft(audio, n_fft=512, hop_length=40, window='hamming')
    XdB = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max, amin=0.001, top_db=120)
    plt.figure(2)
    librosa.display.specshow(XdB, x_axis='time', y_axis='hz', sr=fs, hop_length=40, cmap='gnuplot2')
    plt.colorbar()
    plt.title('stft')
    plt.figure(3)
    mel_spect = librosa.amplitude_to_db(np.abs(mel_spectrum), ref=np.max)
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    # plt.ylim(0, 20)
    plt.colorbar()
    plt.title('mel')
    plt.show()



