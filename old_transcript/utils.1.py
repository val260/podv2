from django.conf import settings

from django.core.files import File
from pod.video.models import Video
from pod.completion.models import Track

from scipy.io import wavfile
from webvtt import WebVTT, Caption
from tempfile import NamedTemporaryFile

import multiprocessing as mp
import os
import subprocess
import deepspeech
import wave
import numpy
import bisect
import shutil
import time

if getattr(settings, 'USE_VAD', False):
    from vad2 import VoiceActivityDetector
    VAD_AGRESS = getattr(settings, 'VAD_AGRESS', 0)
    VAD = True
else:
    FFMPEG_SEG_TIME = getattr(settings, 'FFMPEG_SEG_TIME', 5)
    VAD = False

if getattr(settings, 'USE_PODFILE', False):
    from pod.podfile.models import CustomImageModel
    from pod.podfile.models import CustomFileModel
    from pod.podfile.models import UserFolder
    FILEPICKER = True
else:
    FILEPICKER = False
    from pod.main.models import CustomImageModel
    from pod.main.models import CustomFileModel

FILES_DIR = getattr(settings, 'FILES_DIR', 'files')
FFMPEG = getattr(settings, 'FFMPEG', 'ffmpeg')
FFPROBE = getattr(settings, 'FFPROBE', 'ffprobe')
FFMPEG_NB_THREADS = getattr(settings, 'FFMPEG_NB_THREADS', 0)
NB_WORKERS_POOL = getattr(settings, 'NB_WORKERS_POOL', 2)
SEG_BASE_OPTIONS = "-format wav \
                -acodec pcm_s16le -ar 16000 \
                -threads %(nb_threads)s " % {'nb_threads': FFMPEG_NB_THREADS}

MODEL_DIR = '/home/pod/models'
CST = {
    'en': {
        'alphabet': '%s/en/alphabet.txt' % MODEL_DIR,
        'model': '%s/en/output_graph.pbmm' % MODEL_DIR,
        'lm': '%s/en/lm.binary' % MODEL_DIR,
        'trie': '%s/en/trie' % MODEL_DIR
    },
    'fr': {
        'alphabet': '%s/fr/alphabet.txt' % MODEL_DIR,
        'model': '%s/fr/output_graph.pbmm' % MODEL_DIR,
        'lm': '%s/fr/lm.binary' % MODEL_DIR,
        'trie': '%s/fr/trie' % MODEL_DIR
    }
}


BEAM_WIDTH = 500  # Beam width used in the CTC decoder when building candidate transcriptions
LM_ALPHA = 0.75  # The alpha hyperparameter of the CTC decoder. Language Model weight
LM_BETA = 1.85  # The beta hyperparameter of the CTC decoder. Word insertion bonus.
N_FEATURES = 26  # Number of MFCC features to use
N_CONTEXT = 9  # Size of the context window used for producing timesteps in the input vector


"""
Create the directory dirname(path)/id/name
path the path of the video file of id id
id formatted as %04d
"""
def create_dir(video_id, name):
    video = Video.objects.get(id=video_id)
    dirname = os.path.dirname(video.video.path)
    output_dir = os.path.join(dirname, "%04d" % video_id, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def delete_tmp(video_id):
    try:
        video = Video.objects.get(id=video_id)
        dirname = os.path.dirname(video.video.path)
        path = os.path.join(dirname, "%04d" % video_id, "split")
        if os.path.exists(path):
            print(" --> START DELETING SPLIT SEGMENTS")
            shutil.rmtree(path, ignore_errors=True)
            print("   --> DELETE SEGMENTS : OK")
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)



# Passer le Model en arg au thread fils : marche pas -> pas serialisable ?
# Pickle sur le model puis le passer en arg : marche pas -> pas serialisable
# Passer le model au travers d'une Queue de Multiprocess : marche pas -> pas serialisable
# Initializer avec global : marche

def initfunc(lang):
    global ds_model
    ds_model = deepspeech.Model(CST[lang]['model'], N_FEATURES, N_CONTEXT, CST[lang]['alphabet'], BEAM_WIDTH)
    if 'lm' in CST[lang] and 'trie' in CST[lang]:
        ds_model.enableDecoderWithLM(CST[lang]['alphabet'], CST[lang]['lm'], CST[lang]['trie'], LM_ALPHA, LM_BETA)

"""
Run the deepspeech process to the all segment of the video of id video_id
"""
def deepspeech_run(video_id):
    try:
        video = Video.objects.get(id=video_id)
        lang = video.main_lang
        if lang in CST:
            list_res = []
            list_params = []
            f_insert = lambda x : bisect.insort(list_res, x)
            split_dir = create_dir(video_id, "split")
            rep = os.listdir(split_dir)
            p = mp.Pool(processes=NB_WORKERS_POOL,
                        initializer=initfunc,
                        initargs=(lang,))
            for entry in rep:
                entry_path = os.path.join(split_dir, entry)
                if os.path.isfile(entry_path):
                    p.apply_async(deepspeech_part,  args=(entry_path, lang), callback=f_insert)
            p.close()
            p.join()
            deepspeech2vtt(video_id, list_res)
        else:
            print("%s is not available" % lang)
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)

"""
Apply deepspeech to the file pointed by path
"""
def deepspeech_part(path, lang):
    fs, audio = wavfile.read(path)
    if fs != 16000:
        print("Wav is not 16 kHz but %d kHz" % fs)
    else:
        audio_length = len(audio) / fs
        name = os.path.splitext(os.path.basename(path))[0]
        res =  ds_model.stt(audio, fs)
        return (name, res, audio_length)


def main(video_id):
    print(" --> START CONVERTING TO WAVE AND SPLITTING")
    av2segwave3(video_id)
    print("   --> CONVERT AND SPLIT : OK")
    print(" --> START DEEPSPEECH PROCESS")
    deepspeech_run(video_id)
    print("   --> DEEPSPEECH: OK")
    delete_tmp(video_id)


def deepspeech2vtt(video_id, content):
    try:
        video = Video.objects.get(id=video_id)
        dirname = os.path.dirname(video.video.path)
        if not VAD:
            tps = 0
        webvtt = WebVTT()
        for name, string, duration in content:
            if string:
                if VAD:
                    start = format(float(name), '.3f')
                    end = format(float(name) + duration, '.3f')
                else:
                    start = format(float(tps), '.3f')
                    tps = tps + duration
                    end = format(float(tps), '.3f')
                start_time = time.strftime(
                    '%H:%M:%S',
                    time.gmtime(int(str(start).split('.')[0]))
                )
                start_time += ".%s" % (str(start).split('.')[1])
                end_time = time.strftime('%H:%M:%S', time.gmtime(
                    int(str(end).split('.')[0]))) + ".%s" % (str(end).split('.')[1])
                caption = Caption(
                    '%s' % start_time,
                    '%s' % end_time,
                    '%s' % string
                )
                webvtt.captions.append(caption)
        temp_vtt_file = NamedTemporaryFile(suffix='.vtt')
        with open(temp_vtt_file.name, 'w') as f:
            webvtt.write(f)
        if FILEPICKER:
            videodir, created = UserFolder.objects.get_or_create(
                name='%s' % video.slug,
                owner=video.owner)
            previousSubtitleFile = CustomFileModel.objects.filter(
                name__startswith="subtitle",
                folder=videodir,
                created_by=video.owner
            )
            for subt in previousSubtitleFile:
                subt.delete()
            subtitleFile, created = CustomFileModel.objects.get_or_create(
                name='subtitle',
                folder=videodir,
                created_by=video.owner)
            if subtitleFile.file and os.path.isfile(subtitleFile.file.path):
                os.remove(subtitleFile.file.path)
        else:
            subtitleFile, created = CustomFileModel.objects.get_or_create()
        subtitleFile.file.save("subtitle.vtt", File(temp_vtt_file))
        subtitleVtt, created = Track.objects.get_or_create(video=video)
        subtitleVtt.src = subtitleFile
        subtitleVtt.lang = video.main_lang
        subtitleVtt.save()
        return subtitleFile.file.path
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)


"""
Split wav file into segment of FFMPEG_SEG_TIME sec
"""
def av2segwave(video_id):
    try:
        video = Video.objects.get(id=video_id)
        split_dir = create_dir(video_id, "split")
        path_in = video.video.path
        path_out = os.path.join(split_dir, '%03d' + ".wav")
        options = "-f segment -segment_time %(seg_time)d \
                   -format wav \
                   -acodec pcm_s16le -ar 16000 \
                   -threads %(nb_threads)s" % {
                       'seg_time': FFMPEG_SEG_TIME,
                       'nb_threads': FFMPEG_NB_THREADS
                    }
        cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
               'ffmpeg': FFMPEG,
               'input': path_in,
               'options': options,
               'output': path_out
        }
        subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)

def av2segwave2(video_id):
    try:
        video = Video.objects.get(id=video_id)
        split_dir = create_dir(video_id, "split")
        path_in = video.video.path
        if VAD:
            v = VoiceActivityDetector(path_in)
            raw_detection = v.detect_speech(VAD_AGRESS)
            content = v.convert_windows_to_readible_labels(raw_detection)
        else:
            content = [True]
        options_base = "-format wav \
                   -acodec pcm_s16le -ar 16000 \
                   -threads %(nb_threads)s " % {'nb_threads': FFMPEG_NB_THREADS}
        for elem in content:
            if VAD:
                path_out = os.path.join(split_dir, format(elem['speech_begin'], '07.3f') + ".wav")
                options = options_base + "-ss %(begin)s \
                                          -to %(end)s" % {
                                            'begin': elem['speech_begin'],
                                            'end': elem['speech_end'],
                                          }
            else:
                path_out = os.path.join(split_dir, '%03d' + ".wav")
                options = options_base + "-f segment \
                                          -segment_time %(seg_time)s" % {'seg_time': FFMPEG_SEG_TIME}
            cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
                'ffmpeg': FFMPEG,
                'input': path_in,
                'options': options,
                'output': path_out
            }
            subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)




def av2segwave3(video_id):
    try:
        video = Video.objects.get(id=video_id)
        split_dir = create_dir(video_id, "split")
        path_in = video.video.path
        if VAD:
            av2SegWaveVAD(path_in, split_dir)
        else:
            av2SegWaveLinear(path_in, split_dir)
    except Video.DoesNotExist:
        print("%d is not a valid video id" % video_id)


def av2SegWaveLinear(path_in, split_dir):
    path_out = os.path.join(split_dir, '%03d' + ".wav")
    options = SEG_BASE_OPTIONS + "-f segment \
                                  -segment_time %(seg_time)s" % {
                                      'seg_time': FFMPEG_SEG_TIME
                                    }
    cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
           'ffmpeg': FFMPEG,
           'input': path_in,
           'options': options,
           'output': path_out
    }
    subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    
def av2SegWaveVAD(path_in, split_dir):
    vad = VoiceActivityDetector(path_in)
    raw_detection = vad.detect_speech(VAD_AGRESS)
    content = vad.convert_windows_to_readible_labels(raw_detection)
    for elem in content:
        path_out = os.path.join(split_dir, format(elem['speech_begin'], '07.3f') + ".wav")
        options = SEG_BASE_OPTIONS + "-ss %(begin)s \
                                      -to %(end)s" % {
                                          'begin': elem['speech_begin'],
                                          'end': elem['speech_end'],
                                        }
        cmd = "%(ffmpeg)s -i %(input)s %(options)s %(output)s" % {
            'ffmpeg': FFMPEG,
            'input': path_in,
            'options': options,
            'output': path_out
        }
        subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

