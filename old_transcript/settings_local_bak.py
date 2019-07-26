USE_PODFILE=True
NB_WORKERS_POOL=4

# TRANSCRIPTION PARAMETERS
USE_TRANSCRIPTION=True
SPLIT_METHOD=False
USE_VAD=True
VAD_VERSION=5
TRANSCRIPT_REDUCE_NOISE=False
TRANSCRIPT_SPLIT_CHANNELS=False
TRANSCRIPT_AUTO_COMPUTE_THRESHOLD=True

# VAD PARAMETERS
SAMPLE_WINDOW = 2
SAMPLE_OVERLAP = 0.5
SAMPLE_OVERLAP_RIGHT = 0.5  #utils2
SAMPLE_OVERLAP_LEFT = 0.22 #utils2
# PARAMETERS FOR VAD1
SPEECH_WINDOW = 0.2
SPEECH_ENERGY_THRESHOLD = 0.65
SPEECH_START_BAND = 250
SPEECH_END_BAND = 3500
# PARAMETER FOR VAD2
VAD_AGRESSIVITY = 3

# SEGMENTS TRAITMENT PARAMETERS
THRESHOLD = 0.01
THRESHOLD_PERCENT = 1000

# DEEPSPEECH PARAMETERS
BEAM_WIDTH = 500  # Beam width used in the CTC decoder when building candidate transcriptions
LM_ALPHA = 0.75  # The alpha hyperparameter of the CTC decoder. Language Model weight
LM_BETA = 1.85  # The beta hyperparameter of the CTC decoder. Word insertion bonus.
N_FEATURES = 50  # Number of MFCC features to use
N_CONTEXT = 15 # Size of the context window used for producing timesteps in the input vector

# DEEPSPEECH MODELS FOLDERS
MODELS_HOME = '/home/pod/models'
MODELS_DIR = {
    'en': {
        'alphabet': '%s/en/alphabet.txt' % MODELS_HOME,
        'model': '%s/en/output_graph.pbmm' % MODELS_HOME,
        'lm': '%s/en/lm.binary' % MODELS_HOME,
        'trie': '%s/en/trie' % MODELS_HOME
    },
    'fr': {
        'alphabet': '%s/fr/alphabet.txt' % MODELS_HOME,
        'model': '%s/fr/output_graph.pbmm' % MODELS_HOME,
        'lm': '%s/fr/lm.binary' % MODELS_HOME,
        'trie': '%s/fr/trie' % MODELS_HOME
    }
}
