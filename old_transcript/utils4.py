from django.conf import settings

from django.core.files import File
from pod.video.models import Video
from pod.completion.models import Track

from scipy.io import wavfile
from webvtt import WebVTT, Caption
from tempfile import NamedTemporaryFile

import math
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


def f(entry_path):
    rate, data = wavfile.read(entry_path)
    window, overlap, trim_overlap = 2*rate, int(1*rate), int(0.25*rate)
    lang = 'en'
    window_list = []
    overlap_list = []
    trimed_overlap_list = []
    list_transc = []
    sample_start = 0
    sample_end = -1
    end = len(data)
    while (sample_end < end - 1):
        sample_end = sample_start + window
        if sample_end >= end - 1:
            sample_end = end - 1
        window_list.append((sample_start, sample_start + overlap))
        sample_start += overlap
        overlap_list.append((sample_start, sample_end))
    window_list.extend(overlap_list)
    window_list.sort()
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    for i, elem in enumerate(window_list):
        sample_start = elem[0]
        sample_end = elem[1]
        p.apply_async(trim_silence,
                      args=(i, data,
                            sample_start, sample_end,
                            trim_overlap,
                            rate, lang),
                      callback=lambda x: trimed_overlap_list.append(x) if x else None, #trimed_overlap_list.append,
                      error_callback=print)
    p.close()
    p.join()
    # window_list.extend(trimed_overlap_list)
    # window_list.sort()
    # window_list = merge_seg(window_list)
    trimed_overlap_list.sort()
    trimed_overlap_list = merge_seg(trimed_overlap_list)
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    for start, end in trimed_overlap_list:
        p.apply_async(aux,
                      args=(data, rate, 
                            start, end),
                      callback=lambda x: list_transc.append(x) if x else None,
                      error_callback=print)
    p.close()
    p.join()
    list_transc.sort()
    print(list_transc)
    

def aux(data, rate, start, end):
    data_window = data[start:end]
    res = ds_model.stt(data_window, rate)
    return (start, end, res)


def trim_silence(idx, data, start, end, overlap, rate, lang):
    stop_start, stop_end = False, False
    final_start, final_end = None, None
    sample_start, sample_end = start, end
    old, new = None, None
    while (new is None or new) and (not stop_start or not stop_end) and sample_start < sample_end - 1:
        if new:
            old = new
        data_window = data[sample_start:sample_end]
        new = ds_model.stt(data_window, rate)
        sample_start += overlap
        if old and not stop_end and old.split(' ')[-1] != new.split(' ')[-1]:
            stop_end = True
            final_end = sample_end - overlap
        if old and not stop_end and old.split(' ')[-1] == new.split(' ')[-1]:
            sample_end -= overlap
        if old and not stop_start and old.split(' ')[0] != new.split(' ')[0]:
            stop_start = True
            final_start = sample_start - overlap
        elif old and not stop_start and old.split(' ')[0] == new.split(' ')[0]:
            sample_start += overlap
    if not new or new is None or sample_start >= sample_end - 1:
        return ()
    else:
        return (final_start, final_end)
    

def merge(window_list, overlap_list, trimed_overlap_list):
    res = []
    start, end = None, None
    for idx, o_start, o_end in trimed_overlap_list:
        if not (start is None or end is None):
            res.append((start, end))
            start, end = None, None
        if start is None and overlap_list[idx][0] == o_start:
            start = window_list[idx][0]
        if overlap_list[idx][1] != o_end:
            end = o_end
        elif overlap_list[idx][0] != o_start and overlap_list[idx][1] == o_end:
            start = o_start
        elif overlap_list[idx][0] != o_start and overlap_list[idx][1] != o_end:
            res.append(trimed_overlap_list[idx])
    return res


def merge_seg(seg_list):
    res = []
    start, to_merge = False, False
    speech_begin, speech_end = 0, 0
    for i in range(len(seg_list) - 1):
        start_cur, end_cur = seg_list[i][0], seg_list[i][1]
        start_next, end_next = seg_list[i+1][0], seg_list[i+1][1]
        if start_next - end_cur <= 1 and end_next > end_cur:
            speech_end = end_cur
        if start_next - end_cur <= 1 and end_next <= end_cur:
            speech_end = end_next  
        if start_next - end_cur <= 1 and not start:
            start = True
            speech_begin = start_cur             
        elif start_next - end_cur > 1 and start:
            to_merge = True
        if start and to_merge:
            start, to_merge = False, False
            res.append((speech_begin, speech_end))
        elif not start:
            res.append(seg_list[i])
    if start:
        res.append((speech_begin, speech_end))
    if seg_list and not start:
        res.append(seg_list[len(seg_list) - 1])
    return res

def main():
    return f('pod/media/videos/1b2385219d50b162c9451b5cd47d337ca794d719dc159bc61c1b1c797134445d/0001/osr_us_000_0060_8k_ds.wav')