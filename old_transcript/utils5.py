from django.conf import settings

from django.core.files import File
from pod.video.models import Video
from pod.completion.models import Track

from scipy.io import wavfile
from webvtt import WebVTT, Caption
from tempfile import NamedTemporaryFile

from voiceActivityDetector import VoiceActivityDetector

import math
import numpy
import billiard
import os
import subprocess
import deepspeech
import time
import shutil

TRANSCRIPT_REDUCE_NOISE = getattr(
    settings, 'TRANSCRIPT_REDUCE_NOISE', False)
TRANSCRIPT_SPLIT_CHANNELS = getattr(
    settings, 'TRANSCRIPT_SPLIT_CHANNELS', False)
TRANSCRIPT_TRIM_SILENCE = getattr(
    settings, 'TRANSCRIPT_TRIM_SILENCE', False)

if getattr(settings, 'USE_PODFILE', False):
    from pod.podfile.models import CustomFileModel
    from pod.podfile.models import UserFolder
    FILEPICKER = True
else:
    FILEPICKER = False
    from pod.main.models import CustomFileModel

FFMPEG = getattr(settings, 'FFMPEG', 'ffmpeg')
FFMPEG_NB_THREADS = getattr(settings, 'FFMPEG_NB_THREADS', 0)
SEG_BASE_OPTIONS = "-format wav \
                    -acodec pcm_s16le -ar 16000 \
                    -threads %(nb_threads)s " % {
                        'nb_threads': FFMPEG_NB_THREADS
                    }

MODELS_DIR = getattr(settings, 'MODELS_DIR', '')

BEAM_WIDTH = getattr(settings, 'BEAM_WIDTH', 500)
LM_ALPHA = getattr(settings, 'LM_ALPHA', 0.75)
LM_BETA = getattr(settings, 'LM_BETA', 1.85)
N_FEATURES = getattr(settings, 'N_FEATURES', 26)
N_CONTEXT = getattr(settings, 'N_CONTEXT', 9)

SAMPLE_WINDOW = getattr(settings, 'SAMPLE_WINDOW', 0.02)
SAMPLE_OVERLAP_ONE = getattr(settings, 'SAMPLE_OVERLAP_ONE', 0.01)
SAMPLE_OVERLAP_TWO = getattr(settings, 'SAMPLE_OVERLAP_TWO', 0.01)
SAMPLE_TRIME_OVERLAP = getattr(settings, 'SAMPLE_TRIME_OVERLAP', 0.01)

NB_WORKERS_POOL = max(
    getattr(settings, 'NB_WORKERS_POOL', 4), 1)

DEBUG = getattr(settings, 'DEBUG', True)


def initfunc(lang):
    global ds_model
    ds_model = deepspeech.Model(
        MODELS_DIR[lang]['model'], N_FEATURES, N_CONTEXT,
        MODELS_DIR[lang]['alphabet'], BEAM_WIDTH
    )
    if 'lm' in MODELS_DIR[lang] and 'trie' in MODELS_DIR[lang]:
        ds_model.enableDecoderWithLM(
            MODELS_DIR[lang]['alphabet'], MODELS_DIR[lang]['lm'],
            MODELS_DIR[lang]['trie'], LM_ALPHA, LM_BETA
        )


def princ(entry_path):
#def princ(video):
    res = []
    lang = 'en'
    #lang = video.main_lang
    rate, data = wavfile.read(entry_path)
    #rate, data = wavfile.read(video.video.path)
    data = numpy.ndarray.astype(data, dtype=numpy.int16)
    vad = VoiceActivityDetector(entry_path, lang)
    window_list = vad.detect_speech()
    readible_list = vad.convert_windows_to_readible_labels(window_list)
    if TRANSCRIPT_TRIM_SILENCE:
        readible_list = trim_silence_process(readible_list, data, rate, lang)
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    #print(readible_list)
    for elem in readible_list:
        sample_start = elem["speech_begin"]
        sample_end = elem["speech_end"]
        p.apply_async(aux,
                      args=(data, rate,
                            sample_start, sample_end),
                      callback=lambda x: res.append(x) if x[2] else None,
                      error_callback=print)
    p.close()
    p.join()
    res.sort()
    print(res)


def aux(data, rate, start, end):
    data_window = data[start:end]
    res = ds_model.stt(data_window, rate)
    return (start, end, res)


# #############################################
# TRIM THE SILENCE BEFORE AND AFTER THE SAMPLE
# OF DATA BETWEEN START AND END
# #############################################
def trim_silence_process(data, window_list, rate, lang):
    trimed_list = []
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    for elem in window_list:
        sample_start = elem["speech_begin"]
        sample_end = elem["speech_end"]
        p.apply_async(trim_silence,
                      args=(data,
                            sample_start, sample_end,
                            int(SAMPLE_TRIME_OVERLAP * rate),
                            rate, lang),
                      callback=lambda x: trimed_list.append(x) if x else None,
                      error_callback=print)
    p.close()
    p.join()
    trimed_list.sort()
    return trimed_list


def trim_silence(data, start, end, overlap, rate, lang):
    res = trim_left_silence(data, start, end, overlap, rate, lang)
    if res:
        trim_right_silence(data, res[0], res[1], overlap, rate, lang)
    return res

  
def trim_left_silence(data, start, end, overlap, rate, lang):
    sample_start, sample_end = start, end
    old, new = None, None
    while ((new or new is None) and (old is None or old == new) and sample_start < sample_end - 1):
        if new:
            old = new
        sample_start = sample_start + overlap
        if sample_start >= sample_end - 1:
            sample_start = sample_end - 1
        data_window = data[sample_start:sample_end]
        new = ds_model.stt(data_window, rate)
    if not new or new is None or sample_start >= sample_end - 1:
        return dict()
    else:
        return {'speech_begin': sample_start - overlap, 'speech_end': sample_end}


def trim_right_silence(data, start, end, overlap, rate, lang):
    sample_start, sample_end = start, end
    old, new = None, None
    while ((new or new is None) and (old is None or old == new) and sample_start + 1 < sample_end):
        if new:
            old = new
        sample_start = sample_start + overlap
        if sample_start + 1 >= sample_end:
            sample_end = sample_start + 1
        data_window = data[sample_start:sample_end]
        new = ds_model.stt(data_window, rate)
    if not new or new is None or sample_start >= sample_end - 1:
        return dict()
    else:
        return {'speech_begin': sample_start, 'speech_end':  sample_end + overlap}


def main():
    return princ('pod/media/videos/1b2385219d50b162c9451b5cd47d337ca794d719dc159bc61c1b1c797134445d/0001/osr_us_000_0060_8k_ds.wav')