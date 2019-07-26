from django.conf import settings

import deepspeech

MODELS_DIR = getattr(settings, 'MODELS_DIR', '')

BEAM_WIDTH = getattr(settings, 'BEAM_WIDTH', 500)
LM_ALPHA = getattr(settings, 'LM_ALPHA', 0.75)
LM_BETA = getattr(settings, 'LM_BETA', 1.85)
N_FEATURES = getattr(settings, 'N_FEATURES', 26)
N_CONTEXT = getattr(settings, 'N_CONTEXT', 9)


class Vad():

    def __init__(self, fs, lang):
        self.rate = fs
        self.lang = lang
        self.ds_model = deepspeech.Model(
            MODELS_DIR[lang]['model'], N_FEATURES, N_CONTEXT,
            MODELS_DIR[lang]['alphabet'], BEAM_WIDTH)
        if 'lm' in MODELS_DIR[lang] and 'trie' in MODELS_DIR[lang]:
            self.ds_model.enableDecoderWithLM(
                MODELS_DIR[lang]['alphabet'], MODELS_DIR[lang]['lm'],
                MODELS_DIR[lang]['trie'], LM_ALPHA, LM_BETA)

    def is_speech(self, data):
        res = self.ds_model.stt(data, self.rate)
        return res != ""
