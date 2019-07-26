import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import math
import python_speech_features
import scipy as sp
from scipy import signal

from utils5 import *


'''------------------------------------
FILE READER:
    receives filename,
    returns audio time series (y) and sampling rate of y (sr)
------------------------------------'''
def read_file(file_name, sample_directory):
    #sample_file = file_name
    #sample_directory = '00_samples/'
    sample_path = sample_directory + file_name

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(sample_path)

    return y, sr

'''------------------------------------
NOISE REDUCTION USING POWER:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean


'''------------------------------------
NOISE REDUCTION USING CENTROID ANALYSIS:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_centroid_s(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=100, slope=0.5).highshelf(gain=-12.0, frequency=10000, slope=0.5).limiter(gain=6.0)

    y_cleaned = less_noise(y)

    return y_cleaned

def reduce_noise_centroid_mb(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    # less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
    y_cleaned = less_noise(y)


    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)

    # boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)#.lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


'''------------------------------------
NOISE REDUCTION USING MFCC:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_mfcc_down(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.6).limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def reduce_noise_mfcc_up(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

'''------------------------------------
NOISE REDUCTION USING MEDIAN:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


'''------------------------------------
AUDIO ENHANCER:
    receives an audio matrix,
    returns the same matrix after audio manipulation
------------------------------------'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=10.0, frequency=260, slope=0.1).reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced

'''------------------------------------
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
------------------------------------'''
def output_file(destination ,filename, y, sr, ext=""):
    destination = destination + filename[:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)


samples = ['osr_us_000_0060_8k_ds.wav']
for s in samples:
    folder = 'pod/media/videos/1b2385219d50b162c9451b5cd47d337ca794d719dc159bc61c1b1c797134445d/0001/'
    # reading a file
    filename = s
    y, sr = read_file(filename, folder)

    # reducing noise using db power
    y_reduced_power = reduce_noise_power(y, sr)
    y_reduced_centroid_s = reduce_noise_centroid_s(y, sr)
    y_reduced_centroid_mb = reduce_noise_centroid_mb(y, sr)
    y_reduced_mfcc_up = reduce_noise_mfcc_up(y, sr)
    y_reduced_mfcc_down = reduce_noise_mfcc_down(y, sr)
    y_reduced_median = reduce_noise_median(y, sr)

    # output_file(folder+'new/' ,filename, y_reduced_power, sr, '_pwr')
    # princ(folder+'new/'+filename[:-4] + '_pwr' + '.wav')
    output_file(folder+'new/' ,filename, y_reduced_centroid_s, sr, '_ctr_s')
    princ(folder+'new/'+filename[:-4] + '_ctr_s' + '.wav')
    # output_file(folder+'new/' ,filename, y_reduced_centroid_mb, sr, '_ctr_mb')
    # princ(folder+'new/'+filename[:-4] + '_ctr_mb' + '.wav')
    # output_file(folder+'new/' ,filename, y_reduced_mfcc_up, sr, '_mfcc_up')
    # princ(folder+'new/'+filename[:-4] + '_mfcc_up' + '.wav')
    # output_file(folder+'new/' ,filename, y_reduced_mfcc_down, sr, '_mfcc_down')
    # princ(folder+'new/'+filename[:-4] + '_mfcc_down' + '.wav')
    # output_file(folder+'new/' ,filename, y_reduced_median, sr, '_median')
    # princ(folder+'new/'+filename[:-4] + '_median' + '.wav')
    # output_file(folder+'new/' ,filename, y, sr, '_org')
    # princ(folder+'new/'+filename[:-4] + '_org' + '.wav')