from django.conf import settings

import math
import billiard
import numpy as np
import scipy.io.wavfile as wf

from vad1 import Vad


SAMPLE_WINDOW = getattr(settings, 'SAMPLE_WINDOW', 0.05)
SAMPLE_OVERLAP = getattr(settings, 'SAMPLE_OVERLAP', 0.01)
THRESHOLD_WINDOW = getattr(settings, 'THRESHOLD_WINDOW', 0.05)
WINDOW_DIV = getattr(settings, 'WINDOW_DIV', 3)

NB_WORKERS_POOL = max(
    getattr(settings, 'NB_WORKERS_POOL', 4), 1)


class VoiceActivityDetector():

    def __init__(self, wave_input_filename, lang):
        self.rate, self.data = wf.read(wave_input_filename)
        self.sample_window = SAMPLE_WINDOW
        self.sample_overlap = SAMPLE_OVERLAP # compute_overlap(len(self.data), int(SAMPLE_WINDOW*16000))[-1]
        self.lang = lang


    def convert_windows_to_readible_labels(self, detected_windows):
        speech_time = []
        is_speech = 0
        for window in detected_windows:
            if (window[2] == 1.0 and is_speech == 0): 
                is_speech = 1
                speech_label = {}
                speech_time_start = window[0] # / self.rate #En secondes
                speech_label['speech_begin'] = speech_time_start
            if (window[2] == 0.0 and is_speech == 1):
                is_speech = 0
                speech_time_end = window[0] # / self.rate #En secondes
                speech_label['speech_end'] = speech_time_end
                speech_time.append(speech_label)
        if(is_speech==1):
            speech_time_end = window[1] # / self.rate #En secondes
            speech_label['speech_end'] = speech_time_end
            speech_time.append(speech_label)
        return speech_time


    def initfunc(self):
        global vad
        vad = Vad(self.rate, self.lang)


    def detect_speech(self):
        list_res = []
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        #sample_overlap = self.sample_overlap
        data = self.data
        p = billiard.Pool(processes=NB_WORKERS_POOL,
                          initializer=self.initfunc)
        size_div = math.ceil(len(data) / NB_WORKERS_POOL)
        for i in range(NB_WORKERS_POOL):
            sample_start = i * size_div
            sample_end = (i + 1) * size_div
            if sample_end >= len(data):
                sample_end = len(data) - 1
            p.apply_async(self.detect_speech_aux,
                          args=(sample_window, sample_overlap,
                                sample_start, sample_end),
                          callback=list_res.extend,
                          error_callback=print)
        p.close()
        p.join()
        list_res.sort()
        detected_windows = np.array(list_res)
        seg_list = self.convert_windows_to_readible_labels(detected_windows)
        return seg_list


    def detect_speech_aux(self, window, overlap, start, end):
        data = self.data
        all_windows = []
        detected_windows = []
        sample_start = start
        sample_window = window
        while (sample_start < (end - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end >= end:
                sample_end = end - 1
            data_window = data[sample_start:sample_end]
            is_speech = vad.is_speech(data_window)
            sample_window = int(sample_window / WINDOW_DIV)
            all_windows.append([sample_start, sample_end])
            if is_speech or sample_window < THRESHOLD_WINDOW * self.rate or sample_window < 1:
                sample_window = window
                detected_windows.append([sample_start, sample_end, is_speech])
                sample_start += overlap
        return detected_windows






def decomp_premiers(n):
    res = []
    i, puis = 2, 0
    while n > 1:
        while n % i == 0: 
            n = n / i
            puis += 1
        if puis:
            res.append([i, puis])
        puis = 0
        i += 1
    return res

def compute_overlap(len_data, len_window):
    tmp = decomp_premiers(len_data - len_window)
    res = []
    for e, p in tmp:
        if not res:
            res = [pow(e, i) for i in range(p+1) if pow(e,i) <= len_window]
        else:
            res = [x * pow(e, i) for x in res for i in range(p+1) if (x * pow(e,i)) <= len_window ]
    res.sort()
    return res