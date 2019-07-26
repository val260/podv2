from django.conf import settings

import webrtcvad

VAD_AGRESSIVITY = getattr(settings, 'VAD_AGRESSIVITY', 0)


class Vad():

    def __init__(self, fs, lang):
        self.rate = fs
        self.lang = lang
        self.vad = webrtcvad.Vad(VAD_AGRESSIVITY)
    
    def is_speech(self, data):
        return self.vad.is_speech(data, self.rate)
