import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import imageio
# from pygifsicle import optimize
import os


# UPLOAD_FOLDER = "data"
# image_path = os.path.join(UPLOAD_FOLDER, 'freqWaves{}.png'.format(i=i))
# plt.savefig(image_path)

# gif_path = "test.gif"
# frames_path = "{i}.jpg"
# plt.savefig("{i}.jpg".format(i=i))

class WaveletFreq:
    UPLOAD_FOLDER = "app/static/files"
    GIF_PATH = "waves.gif"

    def __init__(self, **kwargs):
        self.params = {}

        for k, v in kwargs.items():
            self.params[k] = v

        self.freqs = np.zeros(self.params['nfrex'], dtype=float)
        self.fwhms = np.zeros(self.params['nfrex'], dtype=float)

        self.__f0 = 0.9549

    @property
    def freq0(self):
        return self.__f0

    @freq0.setter
    def freq0(self, f0):
        self.__f0 = f0

    @property
    def fourier_factor(self):
        omega0 = 2 * np.pi * self.freq0
        return (4 * np.pi) / (omega0 + np.sqrt(2 + omega0 ** 2))

    @staticmethod
    def frex_fwhm_pnts(*args):

        lofreq = args[0]
        hifreq = args[1]
        nfrex = args[2]

        frequencies = np.logspace(np.log10(lofreq), np.log10(hifreq), nfrex)
        fullHalfMax = np.logspace(np.log10(1), np.log10(10), nfrex)
        return {'freq': frequencies, 'fwhm': fullHalfMax}

    @staticmethod
    def pnts_padding(npnts_times, signal):
        base2 = np.fix(np.log2(npnts_times) / np.log2(2) + 0.4999)
        num = 2 ** base2
        if npnts_times < num:
            nz = 2 ** base2 - npnts_times
        elif num == npnts_times:
            nz = num
        else:
            nz = 2 ** (base2 + 1) - npnts_times

        zeros = np.zeros(int(nz))
        signal_padded = np.concatenate((signal, zeros), axis=0)
        return signal_padded, nz

    @staticmethod
    def signal_spectrum(npnts_times, signal):
        signal_padded, nz = WaveletFreq.pnts_padding(npnts_times, signal)
        signal_fft = fft.fft(a=signal_padded)
        return signal_fft, nz

    @staticmethod
    def base_line_index(time_vector: np.ndarray, pnt1: float = -300.0, pnt2: float = 0.0) -> np.ndarray:
        array = np.array([pnt1, pnt2])
        index = np.array([np.argmin(np.abs(np.fix(time_vector - pnt))) for pnt in array])
        return index

    @staticmethod
    def base_line(signal_power: np.ndarray, base_line: np.ndarray, index0: np.ndarray) -> np.ndarray:
        base_line = np.mean(signal_power[:, index0[0]:index0[1] + 1], axis=1)
        base_line_matrix = np.transpose(np.ones((signal_power.shape[0], signal_power.shape[1])).T * base_line)
        return base_line_matrix

    @staticmethod
    def dB_power(signal_power: np.ndarray, base_line_matrix: np.ndarray) -> np.ndarray:
        return 10*np.log10(signal_power/base_line_matrix)

    def coi(self, npnts_times):
        dt = 1.0 / self.params['srate']
        e_folding = 1.0 / np.sqrt(2)
        pnt0 = np.arange(0.5, np.int(npnts_times/2))
        pnt1 = pnt0[::-1]
        pnt = np.concatenate([pnt0, pnt1])
        coi = self.fourier_factor * e_folding * dt * pnt
        return coi

    def wavelet_windows(self, signal, erp_bool=False):

        freq_fwhm = WaveletFreq.frex_fwhm_pnts(self.params['lofreq'], self.params['hifreq'], self.params['nfrex'])
        frequencies = freq_fwhm['freq']
        fwhms = freq_fwhm['fwhm']

        self.freqs = frequencies
        self.fwhms = fwhms
        npnts_times = signal.shape[0]

        if not erp_bool:
            signal_spectrum, nz = WaveletFreq.signal_spectrum(npnts_times, signal)
        else:
            trials = np.size(signal, 1)
            n_data = int(np.floor(npnts_times * trials))
            # signal.T = np.transpose(signal)
            signal_reshape = signal.T.reshape(n_data)
            signal_spectrum, nz = WaveletFreq.signal_spectrum(n_data, signal_reshape)

        npnts_signal = np.size(signal_spectrum, 0)
        limit_padded = int(np.floor(npnts_signal / 2))

        hz1 = np.linspace(0, int(np.floor(self.params['srate'] / 2)), limit_padded)
        hz0 = np.flip(-1 * hz1)  # -1*hz[::-1]

        hz = np.concatenate((hz0, hz1), axis=0)

        signal_power = np.zeros((len(frequencies), int(npnts_times)))

        if self.params['graphs']:
            fig_graphs1 = plt.figure(figsize=(7, 7))
            ax = plt.axes()

        for fi in range(len(frequencies)):
            s = (fwhms[fi] * (2 * np.pi - 1)) / (4 * np.pi)
            fhz = hz - frequencies[fi]
            fx = np.exp(-0.5 * ((fhz / s) ** 2))
            fx = fx / max(fx)
            fx = np.roll(fx, limit_padded)

            if self.params['graphs']:
                ax.cla()

            if not erp_bool:
                frequency_product = fx * signal_spectrum
                coef = fft.ifft(a=frequency_product)[0: int(npnts_times)]
                signal_power[fi, :] = np.abs(coef) ** 2

                if self.params['graphs']:
                    wavelet = np.abs(fx[0:limit_padded])
                    filtered = np.abs(frequency_product[0:limit_padded])
                    signal = np.abs(signal_spectrum[0:limit_padded])

                    # wavelet = (wavelet - min(wavelet)) / (
                    #             max(wavelet) - min(wavelet))

                    signal = (signal - min(wavelet)) / (
                            max(wavelet) - min(wavelet))

                    filtered = (filtered - min(wavelet)) / (
                            max(wavelet) - min(wavelet))

                    wavelet = 200 * wavelet
                    ax.plot(hz1, signal, 'g', label='Signal')
                    ax.plot(hz1, wavelet, 'b--', label='Wavelet')
                    ax.plot(hz1, filtered, 'r', label='Filtered signal')

            else:
                frequency_product = fx * signal_spectrum
                coef = np.fft.ifft(a=frequency_product)[0: int(npnts_signal-nz)]
                matrix_reshape = coef.reshape(trials, npnts_times)
                coef_matrix = np.abs(matrix_reshape) ** 2
                signal_power[fi, :] = np.mean(coef_matrix, axis=0)

                if self.params['graphs']:
                    wavelet = np.abs(fx[0:limit_padded])
                    filtered = np.abs(frequency_product[0:limit_padded])
                    signal = np.abs(signal_spectrum[0:limit_padded])

                    # wavelet = (wavelet - min(wavelet)) / (
                    #             max(wavelet) - min(wavelet))

                    signal = (signal - min(wavelet)) / (
                            max(wavelet) - min(wavelet))

                    filtered = (filtered - min(wavelet)) / (
                            max(wavelet) - min(wavelet))


                    wavelet = 10000 * wavelet
                    ax.plot(hz1, signal, 'g', label='Signal')
                    ax.plot(hz1, wavelet, 'b--', label='Wavelet')
                    ax.plot(hz1, filtered, 'r', label='Filtered signal')

            if self.params['graphs']:
                ax.set_title('Spectrum', fontsize='14')
                ax.set_ylabel('Amplitude', fontsize='12')
                ax.set_xlabel('Frequency (Hz)', fontsize='12')
                ax.legend()
                ax.set_xlim([0, 60])
                # plt.draw()
                # plt.pause(0.05)
                image_path = os.path.join(WaveletFreq.UPLOAD_FOLDER, 'freqWaves{i}.png'.format(i=fi))
                plt.savefig(image_path)

        path = os.path.join(WaveletFreq.UPLOAD_FOLDER, WaveletFreq.GIF_PATH)
        with imageio.get_writer(path, mode='I') as writer:
            for i in range(frequencies.shape[0]):
                image_path = os.path.join(WaveletFreq.UPLOAD_FOLDER, 'freqWaves{i}.png'.format(i=i))
                writer.append_data(imageio.imread(image_path))
        # optimize(path)
        return signal_power

