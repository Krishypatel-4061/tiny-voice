"""
features.py
===========
Turns 1 second of raw audio into the small "image" the neural network eats.

THIS FILE IS IMPORTED BY BOTH THE TRAINING SCRIPTS AND THE RASPBERRY PI RUNTIME.
That is the whole point - identical code on both sides means zero train/serve skew.

Pipeline implemented here:
    raw audio (16000 floats)
      -> fix length to exactly 1 s
      -> pre-emphasis + framing + FFT + mel filter bank + log + DCT   (the MFCC)
      -> per-clip normalisation (CMVN)
      -> shape (49, 13, 1) float32

We use `python_speech_features` rather than `librosa` because it is pure NumPy:
it installs cleanly on Raspberry Pi OS, has no heavy audio dependencies, and is
fast enough to run in a few milliseconds on a Pi 4.
"""

import numpy as np
from python_speech_features import mfcc as psf_mfcc

import config


def fix_length(audio, target_len=config.CLIP_SAMPLES):
    """
    Force an audio array to be exactly `target_len` samples.

    Recordings are never exactly 1.000 s. Neural networks need a fixed input
    size, so we either:
      - pad with zeros (silence) at the END if the clip is too short, or
      - take the FIRST target_len samples if the clip is too long.

    Parameters
    ----------
    audio : 1-D numpy array of float32, values roughly in [-1.0, 1.0]
    target_len : int, desired number of samples

    Returns
    -------
    1-D numpy array of exactly target_len float32 samples.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()

    if len(audio) > target_len:
        return audio[:target_len]

    if len(audio) < target_len:
        # np.pad with mode="constant" fills the new space with 0.0 = silence.
        pad_amount = target_len - len(audio)
        return np.pad(audio, (0, pad_amount), mode="constant")

    return audio


def audio_to_mfcc(audio, sample_rate=config.SAMPLE_RATE):
    """
    Convert one audio clip into an MFCC feature array of shape (49, 13, 1).

    Step-by-step of what python_speech_features does internally:
      1. Pre-emphasis filter  - boosts high frequencies that the mic attenuates.
      2. Framing              - chops the 1 s into 49 overlapping 40 ms frames.
      3. Windowing + FFT      - gets the frequency content of each frame.
      4. Mel filter bank      - squashes the FFT bins into 40 perceptual bands
                                (the mel scale mirrors how human hearing works:
                                fine resolution at low frequency, coarse at high).
      5. log()                - because loudness perception is logarithmic.
      6. DCT                  - decorrelates the 40 bands into 13 compact
                                coefficients = the MFCCs.

    Returns
    -------
    numpy array, shape (NUM_FRAMES, MFCC_NUM_CEPSTRA, 1), dtype float32
    """
    audio = fix_length(audio)

    features = psf_mfcc(
        audio,
        samplerate=sample_rate,
        winlen=config.MFCC_WINDOW_S,
        winstep=config.MFCC_STRIDE_S,
        numcep=config.MFCC_NUM_CEPSTRA,
        nfilt=config.MFCC_NUM_FILTERS,
        nfft=config.MFCC_NFFT,
        lowfreq=config.MFCC_LOW_FREQ,
        highfreq=config.MFCC_HIGH_FREQ,
        preemph=config.MFCC_PREEMPH,
        ceplifter=22,        # "liftering" - mild smoothing of the cepstrum
        appendEnergy=True,   # replace coefficient 0 with frame log-energy
        winfunc=np.hamming,  # Hamming window reduces spectral leakage
    )

    # --- Per-clip mean/variance normalisation (CMVN) --------------------------
    # Different microphones, distances and room acoustics shift the MFCC values
    # up or down as a block. Subtracting this clip's own mean and dividing by its
    # own standard deviation removes most of that channel effect, so a model
    # trained on your laptop mic still works on the Pi's mic.
    # The 1e-8 stops a divide-by-zero on a perfectly silent clip.
    features = (features - np.mean(features, axis=0)) / (
        np.std(features, axis=0) + 1e-8
    )

    # A CNN expects a "channel" dimension, like an image has RGB channels.
    # Ours is grayscale, so we add a trailing dimension of size 1.
    features = features.astype(np.float32)
    return np.expand_dims(features, axis=-1)


def rms(audio):
    """
    Root-Mean-Square loudness of a clip - a single number for "how loud is this".

    Used at runtime as a cheap gate: if the room is silent there is no point
    burning CPU on a neural network inference.
    """
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


# Quick self-test: run `python features.py` to confirm shapes are what we expect.
if __name__ == "__main__":
    dummy = np.random.randn(config.CLIP_SAMPLES).astype(np.float32) * 0.1
    feats = audio_to_mfcc(dummy)
    print("Input samples :", dummy.shape)
    print("Feature shape :", feats.shape, "(expected", config.FEATURE_SHAPE, ")")
    print("Feature dtype :", feats.dtype)
    print("RMS of dummy  :", round(rms(dummy), 4))
    assert feats.shape == config.FEATURE_SHAPE, "Shape mismatch - check config.py"
    print("OK - features.py is consistent with config.py")
