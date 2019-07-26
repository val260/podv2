from django.conf import settings

from django.core.files import File
from pod.video.models import Video
from pod.completion.models import Track

from scipy.io import wavfile
from webvtt import WebVTT, Caption
from tempfile import NamedTemporaryFile

import os
import billiard
import subprocess
import deepspeech
import bisect
import shutil
import time
import logmmse


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


# ######################################
# TRANSCRIPT VIDEO : DEEPSPEECH PROCESS
# ######################################
def deepspeech_run(video, output_dir):
    """
    Run the deepspeech process to the all segment of the video of id video_id
    """
    msg = ""
    lang = video.main_lang
    split_dir = create_named_dir(output_dir, 'split')
    split_rep = os.listdir(split_dir)
    for folder in split_rep:
        entry_dir = os.path.join(split_dir, folder)
        entry_rep = os.listdir(entry_dir)
        list_res = []
        p = billiard.Pool(processes=min(NB_WORKERS_POOL,
                            len(entry_rep)),
                            initializer=initfunc,
                            initargs=(lang,))
        for entry in entry_rep:
            entry_path = os.path.join(entry_dir, entry)
            p.apply_async(deepspeech_part,  args=(entry_path, lang),
                            callback=lambda x: bisect.insort(
                                list_res, x),
                            error_callback=print)
        p.close()
        p.join()
        msg += deepspeech2VTT(video, list_res, folder, output_dir)
    msg += mergeVTT(video, output_dir, lang)
    return msg

# ###########################################
# TRANSCRIPT VIDEO : WEBVTT FILES MANAGEMENT
# ###########################################
def deepspeech2VTT(video, content, name, output_dir):
    msg = "\ncreate temp subtitles vtt file"
    vtt_dir = create_named_dir(output_dir, 'vtt')
    path_out = os.path.join(vtt_dir, name + '.vtt')
    webvtt = WebVTT()
    for elem in content:
        s, e = elem[0], elem[1]
        trans = elem[2]
        start = format(s, '.3f')
        end = format(e, '.3f')
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
            '%s' % trans
        )
        webvtt.captions.append(caption)
    webvtt.save(path_out)
    msg += "\n- tmpsubtilesfilename :\n%s" % path_out
    return msg


def mergeVTT(video, content_l, lang):
    msg = "\ncreate subtitles vtt file"
    lang = video.main_lang
    webvtt = WebVTT()
    content_d = dict()
    for caption in captions:
        start = caption.start[:-2] + '00'
        end = caption.end[:-2] + '00'
        if start not in content_d:
            content_d[start] = [(end, caption.text)]
        elif (end, caption.text) not in content_d[start]:
            content_d[start].append((end, caption.text))
    content_l = []
    for start in content_d:
        for end, text in content_d[start]:
            bisect.insort(content_l, (start, end, text))
    for start, end, text in content_l:
        caption = Caption(
            '%s' % start,
            '%s' % end,
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


# ##########################################################
# TRANSCRIPT VIDEO : PRETRAITMENT FUNCTIONS
# ##########################################################
def reduce_noise(video_id):
    video_to_encode = Video.objects.get(id=video_id)
    path = video_to_encode.video.path
    name = os.path.splitext(os.path.basename(path))[0]
    path_in = os.path.join(
        os.path.dirname(path), "%04d" % video_id, name + "_ds.wav"
    )
    rate, data = wavfile.read(path_in)
    logmmse.logmmse(data, rate, path_in)


# #################################
# TRANSCRIPT VIDEO : MAIN FONCTION
# #################################
def run_transcriptor(video, output_dir):
    numpy.seterr(all='warn')
    msg = ""
    lang = video.main_lang
    if lang in MODELS_DIR:
        # delete_tmp(video, output_dir)
        msg += "\n- transcript lang :\n%s" % lang
        if DEBUG:
            print(" --> START CONVERTING TO WAVE 16BIT 16KHZ")
        msg += av2wav16b16k(video.video.path, output_dir)
        if DEBUG:
            print("   --> CONVERT : OK")
        if DEBUG and TRANSCRIPT_REDUCE_NOISE:
            print(" --> START REDUCING NOISES")
        if TRANSCRIPT_REDUCE_NOISE:
            reduce_noise(video_id)
        if DEBUG and TRANSCRIPT_REDUCE_NOISE:
            print("   --> RECUDING NOISES : OK")
        if DEBUG:
            print(" --> START EXTRACTING CHANNELS")
        msg += wave16b2mono(video.video.path, output_dir)
        if DEBUG:
            print("   --> EXTRACTING : OK")
        if DEBUG:
            print(" --> START TRANSCRIPTION PROCESS")
        msg += deepspeech_run(video, output_dir)
        if DEBUG:
            print("   --> DEEPSPEECH: OK")
        msg += delete_tmp(video, output_dir)
    else:
        msg += "\n- transcript lang not available :\n%s" % lang
    return msg
