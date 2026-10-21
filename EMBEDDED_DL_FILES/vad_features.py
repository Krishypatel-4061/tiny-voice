"""
vad_features.py   (OPTION B)
============================
The CHEAP features the Voice-Activity Detector looks at.

WHY NOT JUST REUSE THE MFCC?
The whole point of a cascade is that stage 1 must cost far less than stage 2.
The MFCC pipeline (pre-emphasis, overlapping frames, FFT, mel filters, log, DCT)
is a noticeable part of the CPU budget. So the VAD uses a deliberately lighter
recipe:

    1 s of audio (16000 samples)
      -> reshape into 40 frames x 400 samples (25 ms each, NO overlap)
      -> Hann window each frame
      -> ONE batched FFT for all 40 frames at once (numpy does this in C)
      -> power spectrum, summed into just 8 frequency bands
      -> log10
      -> subtract each band's average over time

That last step is the clever one. A steady noise (fan, hum, hiss) has nearly
constant energy over time, so after subtracting the time-average almost nothing
is left. Speech keeps changing (syllables, formants), so it leaves a strong
pattern. The VAD therefore keys on "does the sound CHANGE like speech does?"
rather than "is it loud?" - which is exactly why it can reject a loud fan that a
simple loudness (RMS) gate would let straight through.

Output shape: (40, 8, 1) float32.
"""

import numpy as np

import config
import config_ext

_FRAME = config_ext.VAD_FRAME
_NFRAMES = config_ext.VAD_NUM_FRAMES
assert _FRAME * _NFRAMES == config.CLIP_SAMPLES, "VAD framing must tile the 1 s clip"

# Hann window, computed once.
_WINDOW = np.hanning(_FRAME).astype(np.float32)


def _build_band_matrix():
    """
    A (201 x 8) matrix of 0/1. Multiplying the power spectrum (201 FFT bins,
    20 Hz apart) by it sums the bins belonging to each band in one matrix product.
    """
    freqs = np.fft.rfftfreq(_FRAME, 1.0 / config.SAMPLE_RATE)       # 201 values
    edges = config_ext.VAD_BAND_EDGES_HZ
    matrix = np.zeros((len(freqs), config_ext.VAD_NUM_BANDS), dtype=np.float32)
    for b in range(config_ext.VAD_NUM_BANDS):
        in_band = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if b == config_ext.VAD_NUM_BANDS - 1:
            in_band |= freqs >= edges[-1]            # include the top edge
        matrix[in_band, b] = 1.0
    return matrix


_BAND_MATRIX = _build_band_matrix()


def vad_features(audio):
    """1 s of float32 audio -> VAD feature array of shape (40, 8, 1)."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) < config.CLIP_SAMPLES:
        audio = np.pad(audio, (0, config.CLIP_SAMPLES - len(audio)))
    audio = audio[:config.CLIP_SAMPLES]

    frames = audio.reshape(_NFRAMES, _FRAME) * _WINDOW          # (40, 400)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2             # (40, 201)
    bands = power @ _BAND_MATRIX                                 # (40, 8)
    logb = np.log10(bands + 1e-10)
    logb = logb - logb.mean(axis=0, keepdims=True)               # remove steady level
    return logb.astype(np.float32)[..., np.newaxis]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    steady = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32) * 0.1
    f = vad_features(steady)
    print("shape:", f.shape, "dtype:", f.dtype)
    print("steady-noise feature spread (std):", round(float(f.std()), 3),
          "(small = good, noise leaves little pattern)")
    assert f.shape == config_ext.VAD_FEATURE_SHAPE
    print("OK - vad_features.py works")
