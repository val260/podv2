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


# ##########################################################
# TRANSCRIPT VIDEO : TEMPORARY FILES AND FOLDERS MANAGEMENT
# ##########################################################
def create_named_dir(output_dir, name):
    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def delete_tmp(video, output_dir):
    name = os.path.splitext(os.path.basename(video.video.path))[0]
    path = os.path.join(output_dir, "channels")
    if os.path.exists(path):
        if DEBUG:
            print(" --> START DELETING CHANNELS FILES")
        shutil.rmtree(path, ignore_errors=True)
        if DEBUG:
            print("   --> DELETE CHANNELS : OK")
    path = os.path.join(output_dir, "vtt")
    if os.path.exists(path):
        if DEBUG:
            print(" --> START DELETING VTT FILES")
        shutil.rmtree(path, ignore_errors=True)
        if DEBUG:
            print("   --> DELETE VTT : OK")
    path = os.path.join(output_dir, name + "_ds.wav")
    if os.path.exists(path):
        if DEBUG:
            print(" --> START DELETING WAVE 16BIT 16KHZ FILE")
        os.unlink(path)
        if DEBUG:
            print("   --> DELETE WAVE : OK")
    return "\nremoving temp files"


# #########################################
# TRANSCRIPT VIDEO : TRANSCRIPTION PROCESS
# #########################################
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


def run_transcription(video, output_dir):
    msg = ""
    list_res = []
    res = []
    lang = video.main_lang
    ch_dir = create_named_dir(output_dir, 'channels')
    ch_rep = os.listdir(ch_dir)
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    # Iterate on the files in channels folder
    # One file per channel
    rate = 16000
    for entry in ch_rep:
        entry_path = os.path.join(ch_dir, entry)
        data = wavfile.read(entry_path)[1]
        sample_window = int(rate * SAMPLE_WINDOW)
        sample_overlap_one = int(rate * SAMPLE_OVERLAP_ONE)
        sample_overlap_two = int(rate * SAMPLE_OVERLAP_TWO)
        size_div = math.ceil(len(data) / NB_WORKERS_POOL)
        for i in range(NB_WORKERS_POOL):
            sample_start = i * size_div
            sample_end = sample_start + size_div
            if sample_end >= len(data):
                sample_end = len(data) - 1
            p.apply_async(run_aux,
                          args=(data, rate, sample_window,
                                sample_overlap_one, sample_overlap_two,
                                sample_start, sample_end),
                          callback=list_res.extend,
                          error_callback=print)
    p.close()
    p.join()
    list_res.sort()
    list_res = merge_seg(list_res, data)
    nb_elem = math.ceil(len(list_res) / NB_WORKERS_POOL)
    p = billiard.Pool(processes=NB_WORKERS_POOL,
                      initializer=initfunc,
                      initargs=(lang,))
    for i in range(NB_WORKERS_POOL):
        start = i * nb_elem
        end = start + size_div
        if end >= len(list_res):
            end = len(list_res) - 1
        p.apply_async(transc,
                      args=(list_res, data, start, end, rate),
                      callback=res.extend,
                      error_callback=print)
    p.close()
    p.join()
    res.sort()
    msg += mergeVTT(video, res, rate)
    return msg #list_res


def run_aux(data, rate, window, overlap_o, overlap_t, start, end):
    res = []
    sample_start = start
    sample_end = -1
    while (sample_end < end - 1):
        stop_start, stop_end = False, False
        final_start, final_end = None, None
        old, new = None, None
        while (not stop_start or not stop_end) and sample_end < end - 1:
            if new:
                old = new
            sample_end = sample_start + window
            if sample_end >= end - 1:
                sample_end = end - 1
            data_window = data[sample_start:sample_end]
            new = ds_model.stt(data_window, rate)
            sample_start += overlap_t
            if old and not stop_end and old.split(' ')[-1] == new.split(' ')[-1]:
                stop_end = True
                final_end = sample_end - overlap_o
            if old and not stop_start and old.split(' ')[0] != new.split(' ')[0]:
                stop_start = True
                final_start = sample_start - overlap_o
        if final_start is None:
            final_start = start
        if final_end is None:
            final_end = sample_end
        data_window = data[final_start:final_end]
        #final_detect = ds_model.stt(data_window, rate)
        res.append((final_start, final_end))#, final_detect))
        sample_start = final_end
    print('RES ', res, flush=True)
    return res


def merge_seg(seg_list, data):
    res = []
    start, to_merge = False, False
    speech_begin, speech_end = 0, 0
    for i in range(len(seg_list) - 1):
        start_cur, end_cur = seg_list[i][0], seg_list[i][1]
        start_next, end_next = seg_list[i+1][0], seg_list[i+1][1]
        if start_next - end_cur <= 1:
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


def transc(content, data, start, end, rate):
    res = []
    for i in range(start, end):
        speech_begin, speech_end = content[i]
        data_window = data[speech_begin:speech_end]
        t = ds_model.stt(data_window, rate)
        res.append((speech_begin, speech_end, t))
    return res


def convert_to_readible(content, rate):
    res = []
    for b, e, s in content:
        res.append((b/rate, e/rate, s))
    return res


# ###########################################
# TRANSCRIPT VIDEO : WEBVTT FILES MANAGEMENT
# ###########################################
def mergeVTT(video, content_l, rate):
    msg = "\ncreate subtitles vtt file"
    lang = video.main_lang
    webvtt = WebVTT()
    for start, end, text in content_l:
        start = format(start/rate, '.3f')
        end = format(end/rate, '.3f')
        start_time = time.strftime(
            '%H:%M:%S',
            time.gmtime(int(str(start).split('.')[0]))
        )
        start_time += ".%s" % (str(start).split('.')[1])
        end_time = time.strftime(
            '%H:%M:%S',
            time.gmtime(int(str(end).split('.')[0])))
        end_time += ".%s" % (str(end).split('.')[1])
        caption = Caption(
            '%s' % start_time,
            '%s' % end_time,
            '%s' % text
        )
        webvtt.captions.append(caption)
    temp_vtt_file = NamedTemporaryFile(suffix='.vtt')
    with open(temp_vtt_file.name, 'w') as f:
        if webvtt.captions:
            webvtt.write(f)
    if webvtt.captions:
        msg += "\nstore vtt file in bdd with CustomFileModel model file field"
        if FILEPICKER:
            videodir, created = UserFolder.objects.get_or_create(
                name='%s' % video.slug,
                owner=video.owner)
            previousSubtitleFile = CustomFileModel.objects.filter(
                name__startswith="subtitle_%s" % lang,
                folder=videodir,
                created_by=video.owner
            )
            for subt in previousSubtitleFile:
                subt.delete()
            subtitleFile, created = CustomFileModel.objects.get_or_create(
                name="subtitle_%s" % lang,
                folder=videodir,
                created_by=video.owner)
            if subtitleFile.file and os.path.isfile(subtitleFile.file.path):
                os.remove(subtitleFile.file.path)
        else:
            subtitleFile, created = CustomFileModel.objects.get_or_create()
        subtitleFile.file.save("subtitle_%s.vtt" % lang, File(temp_vtt_file))
        msg += "\nstore vtt file in bdd with Track model src field"
        subtitleVtt, created = Track.objects.get_or_create(video=video)
        subtitleVtt.src = subtitleFile
        subtitleVtt.lang = lang
        subtitleVtt.save()
    else:
        msg += "\nERROR SUBTITLES Output size is 0"
    return msg


# ####################################
# TRANSCRIPT VIDEO : AUDIO CONVERSION
# ####################################
def av2wav16b16k(path_in, output_dir):
    """
    Convert video or audio file into Wave format 16bit 16kHz
    """
    name = os.path.splitext(os.path.basename(path_in))[0]
    path_out = os.path.join(output_dir, name + "_ds.wav")
    options = "-format wav -acodec pcm_s16le -ar 16000 \
               -threads %(nb_threads)s" % {
                   'nb_threads': FFMPEG_NB_THREADS
                }
    cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
        'ffmpeg': FFMPEG,
        'input': path_in,
        'options': options,
        'output': path_out
    }
    msg = "\nffmpegconv16bit16kHzCommand :\n%s" % cmd
    msg += "\n- Converting WAVE 16bit 16kHz : %s" % time.ctime()
    ffmpegconv = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    msg += "\n- End Converting WAVE 16bit 16kHz : %s" % time.ctime()
    with open(output_dir + "/encoding.log", "ab") as f:
        f.write(b'\n\nffmpegconv16bit16kHz:\n\n')
        f.write(ffmpegconv.stdout)
    return msg


def wave16b2mono(path_in, output_dir):
    """
    Convert video or audio file into Wave format mono
    """
    msg = ""
    name = os.path.splitext(os.path.basename(path_in))[0]
    path_in = os.path.join(output_dir, name + "_ds.wav")
    ch_dir = create_named_dir(output_dir, 'channels')
    if TRANSCRIPT_SPLIT_CHANNELS:
        msg += "\n- Encoding WAVE Mono : %s" % time.ctime()
        fs, data = wavfile.read(path_in)
        nch = len(data[0])
        content_l = []
        content_s = set()
        for c in range(0, nch):
            ch_data = data[:, c]
            if not (ch_data.tostring() in content_s):
                content_s.add(ch_data.tostring())
                content_l.append(ch_data)
        for c, ch_data in enumerate(content_l):
            fn = os.path.join(ch_dir, format(c+1, '03d') + ".wav")
            wavfile.write(fn, fs, ch_data)
        msg += "\n- End Encoding WAVE Mono : %s" % time.ctime()
    else:
        path_out = os.path.join(ch_dir, format(1, '03d') + ".wav")
        options = "-ac 1 \
                   -threads %(nb_threads)s" % {
                       'nb_threads': FFMPEG_NB_THREADS
                   }
        cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
            'ffmpeg': FFMPEG,
            'input': path_in,
            'options': options,
            'output': path_out
        }
        msg = "\nffmpegMonoCommand :\n%s" % cmd
        msg += "\n- Encoding WAVE Mono : %s" % time.ctime()
        ffmpegmono = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        msg += "\n- End Encoding WAVE Mono : %s" % time.ctime()
        with open(output_dir + "/encoding.log", "ab") as f:
            f.write(b'\n\nffmpegmono:\n\n')
            f.write(ffmpegmono.stdout)
    return msg


# #################################
# TRANSCRIPT VIDEO : MAIN FUNCTION
# #################################
def main(video, output_dir):
    msg = ""
    lang = video.main_lang
    if lang in MODELS_DIR:
        msg += "\n- transcript lang :\n%s" % lang
        if DEBUG:
            print(" --> START CONVERTING TO WAVE 16BIT 16KHZ")
        msg += av2wav16b16k(video.video.path, output_dir)
        if DEBUG:
            print("   --> CONVERT : OK")
        if DEBUG and TRANSCRIPT_REDUCE_NOISE:
            print(" --> START REDUCING NOISES")
        if TRANSCRIPT_REDUCE_NOISE:
            reduce_noise(video.id)
        if DEBUG and TRANSCRIPT_REDUCE_NOISE:
            print("   --> RECUDING NOISES : OK")
        if DEBUG:
            print(" --> START EXTRACTING CHANNELS")
        msg += wave16b2mono(video.video.path, output_dir)
        if DEBUG:
            print("   --> EXTRACTING : OK")
        if DEBUG:
            print(" --> START TRANSCRIPTION PROCESS")
        msg += run_transcription(video, output_dir)
        if DEBUG:
            print("   --> DEEPSPEECH: OK")
        msg += delete_tmp(video, output_dir)
    else:
        msg += "\n- transcript lang not available :\n%s" % lang
    return msg
