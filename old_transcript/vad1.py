from django.conf import settings

import numpy as np

SPEECH_ENERGY_THRESHOLD = getattr(settings, 'SPEECH_ENERGY_THRESHOLD', 0.6)
SPEECH_START_BAND = getattr(settings, 'SPEECH_START_BAND', 300)
SPEECH_END_BAND = getattr(settings, 'SPEECH_END_BAND', 3000)


class Vad():
    """
    Use signal energy to detect voice activity in audio data
    """

    def __init__(self, fs, lang):
        self.rate = fs
        self.lang = lang
        self.speech_energy_threshold = SPEECH_ENERGY_THRESHOLD
        self.speech_start_band = SPEECH_START_BAND
        self.speech_end_band = SPEECH_END_BAND


    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data), 1.0 / self.rate)
        data_freq = data_freq[1:]
        return data_freq


    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl


    def _calculate_energy(self, audio_data):
        data_amplitude = self._calculate_amplitude(audio_data)
        data_energy = data_amplitude ** 2
        return data_energy


    def _znormalize_energy(self, data_energy):
        energy_mean = np.mean(data_energy)
        energy_std = np.std(data_energy)
        energy_znorm = (data_energy - energy_mean) / energy_std
        return energy_znorm


    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq


    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        # znormalize brings worse results
        # data_energy = self._znormalize_energy(data_energy)
        energy_freq = self._connect_energy_with_frequencies(
            data_freq, data_energy)
        return energy_freq


    def _sum_energy_in_band(self, energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band < f < end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy


    def _median_filter(self, x, k):
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i+1)] = x[j:]
            y[-j:, -(i+1)] = x[-1]
        return np.median(y, axis=1)


    def _smooth_speech_detection(self, detected_windows):
        median_window = int(self.speech_window / self.sample_window)
        if median_window % 2 == 0:
            median_window = median_window - 1
        median_energy = self._median_filter(
            detected_windows[:, 1], median_window)
        return median_energy


    def is_speech(self, data):
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        energy_freq = self._calculate_normalized_energy(data)
        sum_voice_energy = self._sum_energy_in_band(
            energy_freq, start_band, end_band)
        sum_full_energy = sum(energy_freq.values())
        speech_ratio = sum_voice_energy / sum_full_energy
        speech_ratio = speech_ratio > self.speech_energy_threshold
        return speech_ratio
