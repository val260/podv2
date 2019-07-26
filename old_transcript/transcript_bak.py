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
import numpy
import bisect
import shutil
import time
import logmmse

if getattr(settings, 'USE_VAD', False):
    from voiceActivityDetector import VoiceActivityDetector
    VAD = True
else:
    FFMPEG_SEG_TIME = getattr(settings, 'FFMPEG_SEG_TIME', 5)
    VAD = False
SPLIT_METHOD = getattr(
    settings, 'SPLIT_METHOD', True)
TRANSCRIPT_REDUCE_NOISE = getattr(
    settings, 'TRANSCRIPT_REDUCE_NOISE', False)
TRANSCRIPT_SPLIT_CHANNELS = getattr(
    settings, 'TRANSCRIPT_SPLIT_CHANNELS', False)
TRANSCRIPT_AUTO_COMPUTE_THRESHOLD = getattr(
    settings, 'TRANSCRIPT_AUTO_COMPUTE_THRESHOLD', False)
if TRANSCRIPT_AUTO_COMPUTE_THRESHOLD:
    THRESHOLD_PERCENT = getattr(settings, 'THRESHOLD_PERCENT', 200)
else:
    THRESHOLD = getattr(settings, 'THRESHOLD', 0.5)

if getattr(settings, 'USE_PODFILE', False):
    from pod.podfile.models import CustomFileModel
    from pod.podfile.models import UserFolder
    FILEPICKER = True
else:
    FILEPICKER = False
    from pod.main.models import CustomFileModel

FFMPEG = getattr(settings, 'FFMPEG', 'ffmpeg')
FFPROBE = getattr(settings, 'FFPROBE', 'ffprobe')
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
    path = os.path.join(output_dir, "split")
    if os.path.exists(path):
        if DEBUG:
            print(" --> START DELETING SPLIT SEGMENTS")
        shutil.rmtree(path, ignore_errors=True)
        if DEBUG:
            print("   --> DELETE SEGMENTS : OK")
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


# ##############################################
# TRANSCRIPT VIDEO : AUDIO SEGMENTATION PROCESS
# ##############################################
def av2segwave(video, output_dir):
    """
    Split audio/video file into wave segment
    """
    msg = ""
    if VAD:
        msg += "\n- segmentation method :\nvad"
    else:
        msg += "\n- segmentation method :\nlinear %f sec" % FFMPEG_SEG_TIME
    lang = video.main_lang
    ch_dir = create_named_dir(output_dir, 'channels')
    rep = os.listdir(ch_dir)
    p = billiard.Pool(processes=NB_WORKERS_POOL)
    for entry in rep:
        name = os.path.splitext(entry)[0]
        path_in = os.path.join(ch_dir, entry)
        path_out = os.path.join('split', name)
        output_subdir = create_named_dir(output_dir, path_out)
        if VAD:
            p.apply_async(av2SegWaveVAD,
                          args=(path_in, output_dir, output_subdir, lang),
                          error_callback=print)
        else:
            p.apply_async(av2SegWaveLinear,
                          args=(path_in, output_dir, output_subdir),
                          error_callback=print)
    p.close()
    p.join()
    return msg


def av2SegWaveLinear(path_in, output_dir, output_subdir):
    """
    Split audio/video file into wave segment of FFMPEG_SEG_TIME sec
    using linear segmentation
    """
    path_out = os.path.join(output_subdir, '%03d' + ".wav")
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
    msg = "\nffmpegSegmentationCommand :\n%s" % cmd
    msg += "\n- Splitting Audio : %s" % time.ctime()
    ffmpegseg = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    msg += "\n- End Splitting : %s" % time.ctime()
    with open(output_dir + "/encoding.log", "ab") as f:
        f.write(b'\n\nffmpegsegment:\n\n')
        f.write(ffmpegseg.stdout)
    return msg


def av2SegWaveVAD(path_in, output_dir, output_subdir, lang):
    """
    Split audio/video file into wave segment of speech cutting arround blanks
    using Voice Activity Detection tool
    """
    msg = ""
    content = []
    vad = VoiceActivityDetector(path_in, lang)
    raw_detection = vad.detect_speech()
    content += vad.convert_windows_to_readible_labels(raw_detection)
    content.sort(key=lambda d: d['speech_begin'])
    content = clean_list(content)
    content = merge_seg(content)
    for idx, elem in enumerate(content):
        path_out = os.path.join(
            output_subdir, format(elem['speech_begin'], '07.3f') + ".wav"
        )
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
        msg += "\nffmpegSegmentationCommand %i :\n%s" % (idx, cmd)
        msg += "\n- Splitting Audio %i : %s" % (idx, time.ctime())
        ffmpegseg = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        with open(output_dir + "/encoding.log", "ab") as f:
            f.write(b'\n\nffmpegsegment%i:\n\n' % idx)
            f.write(ffmpegseg.stdout)
    return msg


# ######################################
# TRANSCRIPT VIDEO : DEEPSPEECH PROCESS
# ######################################
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


def deepspeech_part(path, lang):
    """
    Apply deepspeech to the file pointed by path
    """
    fs, audio = wavfile.read(path)
    audio_length = len(audio) / fs
    name = os.path.splitext(os.path.basename(path))[0]
    res = ds_model.stt(audio, fs)
    return (name, res, audio_length)


# ###########################################
# TRANSCRIPT VIDEO : WEBVTT FILES MANAGEMENT
# ###########################################
def deepspeech2VTT(video, content, name, output_dir):
    msg = "\ncreate temp subtitles vtt file"
    vtt_dir = create_named_dir(output_dir, 'vtt')
    path_out = os.path.join(vtt_dir, name + '.vtt')
    webvtt = WebVTT()
    for elem in content:
        if VAD and SPLIT_METHOD:
            s, e = float(elem[0]), float(elem[0]) + elem[2]
            trans = elem[1]
        elif not VAD and SPLIT_METHOD:
            s, e = float(elem[0]*FFMPEG_SEG_TIME), float((elem[0]+1)*FFMPEG_SEG_TIME)
            trans = elem[1]
        else:
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
        if trans:
            webvtt.captions.append(caption)
    webvtt.save(path_out)
    msg += "\n- tmpsubtilesfilename :\n%s" % path_out
    return msg


def mergeVTT(video, output_dir, lang):
    msg = "\ncreate subtitles vtt file"
    lang = video.main_lang
    vtt_dir = create_named_dir(output_dir, 'vtt')
    vtt_rep = os.listdir(vtt_dir)
    webvtt = WebVTT()
    content_d = dict()
    for vtt in vtt_rep:
        path_in = os.path.join(vtt_dir, vtt)
        captions = webvtt.read(path_in)
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


def compute_threshold(content):
    total_blank = 0
    for i in range(len(content)-1):
        end_current = content[i]['speech_end']
        begin_next = content[i+1]['speech_begin']
        total_blank += (begin_next - end_current)
    avg_blank = total_blank / len(content)
    return avg_blank / THRESHOLD_PERCENT


def clean_list(content):
    content_d = dict()
    for elem in content:
        str_begin = str(elem['speech_begin'])
        if ((str_begin not in content_d)
                or (content_d[str_begin] < elem['speech_end'])):
            content_d[str_begin] = elem['speech_end']
    new_content = [{'speech_begin': float(sb), 'speech_end': content_d[sb]}
                   for sb in content_d]
    return sorted(new_content, key=lambda e: e['speech_begin'])


def merge_seg(content):
    if len(content) > 1:
        if TRANSCRIPT_AUTO_COMPUTE_THRESHOLD:
            thr = compute_threshold(content)
        else:
            thr = THRESHOLD
        res = []
        label = {}
        speech_end = 0
        for elem in content:
            gap = elem['speech_begin'] - speech_end
            if (((gap <= thr) or (elem['speech_begin'] <= speech_end))
                    and (speech_end > 0)):
                speech_end = elem['speech_end']
            elif (gap > thr) and (speech_end > 0):
                label['speech_end'] = speech_end
                res.append(label)
            if (gap > thr) or (speech_end <= 0):
                label = {}
                label['speech_begin'] = elem['speech_begin']
                speech_end = elem['speech_end']
        if not ('speech_end' in label):
            label['speech_end'] = speech_end
            res.append(label)
        return res
    else:
        return content


# #################################
# TRANSCRIPT VIDEO : MAIN FONCTION
# #################################
def run_transcriptor(video, output_dir):
    numpy.seterr(all='warn')
    msg = ""
    lang = video.main_lang
    if lang in MODELS_DIR:
        delete_tmp(video, output_dir)
        msg += "\n- transcript lang :\n%s" % lang
        if DEBUG:
            print(" --> START CONVERTING TO WAVE 16BIT 16KHZ")
        msg += av2wav16b16k(video.video.path, output_dir)
        if DEBUG:
            print("   --> CONVERT : OK")
        if TRANSCRIPT_REDUCE_NOISE:
            reduce_noise(video_id)
        if DEBUG:
            print(" --> START EXTRACTING CHANNELS")
        msg += wave16b2mono(video.video.path, output_dir)
        if DEBUG:
            print("   --> EXTRACTING : OK")
        if DEBUG:
            print(" --> START SPLITTING")
        av2segwave(video, output_dir)
        if DEBUG:
            print("   --> SPLIT : OK")
        if DEBUG:
            print(" --> START DEEPSPEECH PROCESS")
        msg += deepspeech_run(video, output_dir)
        if DEBUG:
            print("   --> DEEPSPEECH: OK")
        msg += delete_tmp(video, output_dir)
    else:
        msg += "\n- transcript lang not available :\n%s" % lang
    return msg


def main(video_id):
    video = Video.objects.get(id=video_id)
    dirname = os.path.dirname(video.video.path)
    output_dir = os.path.join(dirname, "%04d" % video_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msg = run_transcriptor(video, output_dir)
    print(msg)
