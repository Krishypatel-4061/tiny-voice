"""
config_ext.py
=============
Settings for the TWO NEW features added on top of the working TinyVoice system:

    Option A - a learned FEATURE DENOISER (cleans the MFCC "image" before the
               DS-CNN sees it, so recognition survives a noisy room)
    Option B - a two-stage CASCADE: a tiny learned VOICE-ACTIVITY DETECTOR (VAD)
               decides "is anyone speaking?" and only then wakes up the bigger
               DS-CNN (saves CPU/power and rejects loud non-speech noise)

WHY A SEPARATE FILE?
Your original config.py is untouched and still drives the working demo. Every
new script imports BOTH:
        import config          # the old settings (sample rate, labels, paths)
        import config_ext      # the new settings in this file
That way the add-on can never break the system you already showed.
"""

import os

import config

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# Put YOUR OWN room-noise recordings here (see record_noise.py). Real noise from
# the demo room is the most valuable data in the whole add-on.
NOISE_DIR = os.path.join(config.DATASET_DIR, "noise")

# Processed arrays for the denoiser + the train/val/test file lists.
ENH_DIR = os.path.join(config.DATASET_DIR, "enhancer")
SPLITS_PATH = os.path.join(ENH_DIR, "splits.json")

# Where benchmark tables / plots are written (for your report).
RESULTS_DIR = os.path.join(config.PROJECT_ROOT, "results")

# Option A model files
ENHANCER_KERAS_PATH = os.path.join(config.MODELS_DIR, "enhancer.keras")
ENHANCER_FLOAT_PATH = os.path.join(config.MODELS_DIR, "enhancer_float32.tflite")
ENHANCER_INT8_PATH = os.path.join(config.MODELS_DIR, "enhancer_int8.tflite")

# Option B model files
VAD_KERAS_PATH = os.path.join(config.MODELS_DIR, "vad.keras")
VAD_FLOAT_PATH = os.path.join(config.MODELS_DIR, "vad_float32.tflite")
VAD_INT8_PATH = os.path.join(config.MODELS_DIR, "vad_int8.tflite")
VAD_META_PATH = os.path.join(config.MODELS_DIR, "vad_meta.json")  # chosen threshold

# ---------------------------------------------------------------------------
# 2. NOISE SETTINGS (used to build noisy training/test audio)
# ---------------------------------------------------------------------------
# Synthetic noise types we can generate in code, so you need no big download.
#   white  : hiss (all frequencies equal)             pink : softer hiss, more bass
#   hum    : 50 Hz mains hum + harmonics (India)      fan  : low rumble that wobbles
#   babble : several overlapping "voices"             clicks: claps / door bangs
NOISE_KINDS = ["white", "pink", "hum", "fan", "babble", "clicks"]

# Kinds that make sense as a CONTINUOUS room background (clicks are one-off events).
BACKGROUND_KINDS = ["white", "pink", "hum", "fan", "babble"]

# SNR = Signal-to-Noise Ratio in decibels. 20 dB = quiet room, 0 dB = noise is as
# loud as the speech, -5 dB = noise is louder than the speech.
TRAIN_SNRS_DB = [-5, 0, 5, 10, 15, 20]
EVAL_SNRS_DB = [20, 10, 5, 0]

# ---------------------------------------------------------------------------
# 3. OPTION A - DENOISER DATA SETTINGS
# ---------------------------------------------------------------------------
ENH_COPIES_TRAIN = 8      # noisy copies made from every clean training clip
ENH_COPIES_VAL = 2        # ... and from every validation clip
ENH_MAX_SC_CLIPS = 1500   # cap on extra Google Speech Commands clips (if present)
ENH_EPOCHS = 40
ENH_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# 4. OPTION B - VAD SETTINGS
# ---------------------------------------------------------------------------
# The VAD uses a MUCH cheaper feature than the MFCC: the 1 s window is cut into
# 40 frames of 25 ms (400 samples, no overlap), and we keep the energy in just 8
# frequency bands. One batched FFT in numpy - a small fraction of an MFCC.
VAD_FRAME = 400
VAD_NUM_FRAMES = config.CLIP_SAMPLES // VAD_FRAME          # = 40
VAD_NUM_BANDS = 8
VAD_BAND_EDGES_HZ = [100, 300, 500, 800, 1200, 2000, 3200, 5000, 8000]  # 9 edges -> 8 bands
VAD_FEATURE_SHAPE = (VAD_NUM_FRAMES, VAD_NUM_BANDS, 1)

VAD_EPOCHS = 30
VAD_BATCH_SIZE = 64
VAD_DEFAULT_THRESHOLD = 0.40   # used if vad_meta.json does not exist yet.
                               # Deliberately LOW: a missed command is worse than
                               # a false wake-up (the DS-CNN rejects false wake-ups).
