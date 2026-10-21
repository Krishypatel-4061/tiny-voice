"""
config.py
=========
ONE place where every tunable number for TinyVoice lives.

WHY THIS FILE EXISTS (important ML engineering idea):
A speech model only works if the features you feed it at RUNTIME are computed
EXACTLY the same way as the features you used at TRAINING time. If training used
40 ms windows and the Raspberry Pi uses 25 ms windows, accuracy collapses and you
will waste days debugging "the model works on my laptop but not on the Pi".
This mismatch has a name in industry: TRAIN/SERVE SKEW.

The fix is simple: every script imports its numbers from here. Never hardcode
audio or feature numbers anywhere else.
"""

import os

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
# os.path.dirname(__file__) = the folder this config.py file sits in.
# Building paths this way means the project works no matter which directory you
# run the scripts from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CUSTOM_AUDIO_DIR = os.path.join(DATASET_DIR, "custom")            # your own recordings
SPEECH_COMMANDS_DIR = DATASET_DIR  # Google dataset
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")            # .npy feature arrays
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "tinyvoice.keras")
TFLITE_FLOAT_PATH = os.path.join(MODELS_DIR, "tinyvoice_float32.tflite")
TFLITE_INT8_PATH = os.path.join(MODELS_DIR, "tinyvoice_int8.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# ---------------------------------------------------------------------------
# 2. LABELS (the classes the neural network can output)
# ---------------------------------------------------------------------------
# The 5 commands from the proposal.
COMMAND_WORDS = ["on", "off", "up", "down", "stop"]

# Two EXTRA classes that are not commands. Beginners often skip these and then
# wonder why the system fires randomly during a demo.
#   _unknown_   -> any other spoken word ("hello", "seven", "left", ...)
#   _silence_   -> room tone, fan hum, chair scrape, no speech at all
# With these two classes the model can actively say "that wasn't a command",
# instead of being forced to pick one of the 5 commands for every sound it hears.
UNKNOWN_LABEL = "_unknown_"
SILENCE_LABEL = "_silence_"

# Final ordered label list. ORDER MATTERS: index 0 of the model output always
# corresponds to LABELS[0]. Never reorder this after training a model.
LABELS = COMMAND_WORDS + [UNKNOWN_LABEL, SILENCE_LABEL]
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------------------
# 3. AUDIO SETTINGS
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz. 16 kHz is the standard for speech ML.
                             # Human speech information lives below 8 kHz, and
                             # by Nyquist we need 2x that. Higher rates only
                             # waste CPU and RAM.
CLIP_DURATION_S = 1.0        # Every training clip and every inference window is
                             # exactly 1 second long. Short commands fit easily.
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)   # = 16000 samples
AUDIO_CHANNELS = 1           # Mono. Two channels would double data for no gain.

# ---------------------------------------------------------------------------
# 4. FEATURE (MFCC) SETTINGS
# ---------------------------------------------------------------------------
# We do NOT feed raw audio to the network. 16000 raw numbers per second is far
# too much. Instead we compute MFCCs (Mel-Frequency Cepstral Coefficients):
# a compact, perceptually-motivated summary of "which frequencies were loud, in
# which short slice of time". This is the same front-end used by nearly every
# keyword-spotting system.
MFCC_WINDOW_S = 0.040        # 40 ms analysis window ("frame")
MFCC_STRIDE_S = 0.020        # move forward 20 ms each frame (so frames overlap)
MFCC_NUM_CEPSTRA = 13        # how many MFCC numbers per frame
MFCC_NUM_FILTERS = 40        # mel filter bank size (the resolution of the
                             # frequency analysis before the cepstral step)
MFCC_NFFT = 1024             # FFT size. MUST be >= window length in samples
                             # (0.040 * 16000 = 640), so 1024 is the next power of 2.
MFCC_LOW_FREQ = 20           # Hz. Ignore below 20 Hz - it is DC offset/rumble.
MFCC_HIGH_FREQ = 7800        # Hz. Just under Nyquist (8000) to avoid edge effects.
MFCC_PREEMPH = 0.97          # pre-emphasis boosts high frequencies, standard value.

# Number of frames produced for one 1-second clip.
# Formula from the framing maths: 1 + floor((total_samples - window) / stride)
NUM_FRAMES = 1 + (CLIP_SAMPLES - int(MFCC_WINDOW_S * SAMPLE_RATE)) // int(
    MFCC_STRIDE_S * SAMPLE_RATE
)   # = 1 + (16000 - 640)//320 = 49

# So every clip becomes a small "image" of shape (49, 13, 1) -> perfect for a CNN.
FEATURE_SHAPE = (NUM_FRAMES, MFCC_NUM_CEPSTRA, 1)

# ---------------------------------------------------------------------------
# 5. TRAINING SETTINGS
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
VALIDATION_SPLIT_SPEAKERS = 1   # how many of your teammates are held out for
                                # validation (speaker-independent evaluation)
TEST_SPLIT_SPEAKERS = 1         # and how many for the final test set
RANDOM_SEED = 1337              # fixing the seed makes your results reproducible,
                                # which examiners like to see.

# How many augmented copies to generate per original custom recording.
# Augmentation = deliberately degrading your clean recordings (noise, shifts,
# volume changes) so the model learns to survive a noisy lab during the demo.
AUGMENTATIONS_PER_CLIP = 4

# Cap how many Google Speech Commands clips we pull in, so the 5 real commands
# are not drowned out by thousands of "unknown" clips (class imbalance).
MAX_UNKNOWN_CLIPS = 1500
MAX_SILENCE_CLIPS = 800
MAX_SC_CLIPS_PER_WORD = 1200    # for up / down / stop, which exist in the dataset

# ---------------------------------------------------------------------------
# 6. RUNTIME / INFERENCE SETTINGS (Raspberry Pi)
# ---------------------------------------------------------------------------
INFERENCE_HOP_S = 0.25       # run the model 4x per second on a sliding window.
                             # Smaller = more responsive but more CPU.
CONFIDENCE_THRESHOLD = 0.80  # below this we refuse to act (matches the proposal)
SMOOTHING_WINDOW = 3         # average the probabilities over the last N windows.
                             # This kills one-off spurious spikes. Standard trick
                             # in keyword spotting, called "posterior smoothing".
REFRACTORY_S = 1.5           # after firing a command, ignore everything for this
                             # long so one spoken word does not trigger 4 times.
RMS_SILENCE_THRESHOLD = 0.012  # if the window is quieter than this we skip
                               # inference entirely and save CPU. Tune on the Pi
                               # using tools/test_audio.py.

# ---------------------------------------------------------------------------
# 7. GPIO PIN MAP (BCM numbering, Raspberry Pi 4)
# ---------------------------------------------------------------------------
# BCM numbering = the GPIO number printed in pinout diagrams, NOT the physical
# pin position. gpiozero always uses BCM.
#
# These pins deliberately AVOID:
#   GPIO 2, 3   -> I2C (used by the optional OLED)
#   GPIO 18,19,20,21 -> I2S (used only if you pick the INMP441 mic instead of USB)
#   GPIO 14, 15 -> UART (serial console)
PIN_LED = 17            # LED  -> "light" command
PIN_MOTOR_IN1 = 23      # L298N IN1  -> "fan" direction
PIN_MOTOR_IN2 = 24      # L298N IN2
PIN_MOTOR_ENA = 25      # L298N ENA (PWM speed pin)
PIN_SERVO = 12          # servo signal -> "up" / "down"

MOTOR_SPEED = 0.7       # 0.0 to 1.0 duty cycle for the fan motor
SERVO_STEP_DEG = 20     # how far the servo moves per up/down command
SERVO_MIN_DEG = -90
SERVO_MAX_DEG = 90

USE_OLED = False        # flip to True only after the OLED is wired and tested
