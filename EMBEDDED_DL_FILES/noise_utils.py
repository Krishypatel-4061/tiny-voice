"""
noise_utils.py
==============
Everything about NOISE lives here: generating it, loading real recordings of it,
and mixing it into clean speech at an exact Signal-to-Noise Ratio (SNR).

Used by: build_enhancer_data.py, train_vad.py, eval_noise_robustness.py,
         benchmark_cascade.py

KEY IDEA - SNR in decibels
--------------------------
    SNR_dB = 10 * log10( speech_power / noise_power )

    +20 dB -> speech is 100x more powerful than noise (quiet room)
      0 dB -> speech and noise equally powerful (very noisy)
     -5 dB -> noise is stronger than the speech (brutal)

To hit a target SNR we do NOT guess a volume: we measure the speech power, then
scale the noise so the ratio comes out exactly right. That is `mix_at_snr`.

Every noise generator returns audio normalised to UNIT RMS (loudness = 1.0).
That makes mixing trivial: multiply by whatever level you want.
"""

import os

import numpy as np
from scipy.io import wavfile

import config
import config_ext

SR = config.SAMPLE_RATE


# ===========================================================================
# 1. SYNTHETIC NOISE GENERATORS
# ===========================================================================
def _unit_rms(x):
    """Scale an array so its RMS loudness is exactly 1.0."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.sqrt(np.mean(x ** 2) + 1e-12)


def _colored_noise(n, exponent, rng):
    """
    Noise whose POWER falls off as 1 / f^exponent.
        exponent 0 -> white noise (flat spectrum, a hiss)
        exponent 1 -> pink noise  (more low-frequency energy, like rain/wind)
        exponent 2 -> brown noise (heavy rumble)

    Method: make white noise, go to the frequency domain with an FFT, divide each
    frequency by f^(exponent/2) (amplitude = sqrt(power)), come back with an
    inverse FFT.
    """
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = freqs[1]                      # avoid dividing by zero at 0 Hz
    spectrum = spectrum / (freqs ** (exponent / 2.0))
    return np.fft.irfft(spectrum, n)


def _band_limit(x, low_hz, high_hz):
    """Keep only frequencies between low_hz and high_hz (FFT brick-wall filter)."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    return np.fft.irfft(spectrum, len(x))


def _slow_envelope(n, cutoff_hz, rng):
    """A random slowly-varying positive curve (used to make noise 'wobble')."""
    env = np.abs(_band_limit(rng.standard_normal(n), 0.1, cutoff_hz))
    return env / (env.max() + 1e-9)


def synth_noise(kind, n, rng):
    """
    Generate `n` samples of synthetic noise of the given kind, at unit RMS.
    `rng` is a numpy Generator (np.random.default_rng(seed)) so results are
    reproducible.
    """
    if kind == "white":
        x = rng.standard_normal(n)

    elif kind == "pink":
        x = _colored_noise(n, 1.0, rng)

    elif kind == "hum":
        # Electrical mains hum: 50 Hz and its harmonics, each with a random phase,
        # plus a whisper of hiss so it is not a perfectly clean tone.
        t = np.arange(n) / SR
        x = np.zeros(n)
        for h in range(1, 8):
            x += (1.0 / h) * np.sin(2 * np.pi * 50.0 * h * t + rng.uniform(0, 2 * np.pi))
        x += 0.05 * rng.standard_normal(n)

    elif kind == "fan":
        # Brown rumble + some pink "whoosh", wobbling slowly in loudness.
        rumble = _colored_noise(n, 2.0, rng)
        rumble = _unit_rms(rumble)
        whoosh = _unit_rms(_colored_noise(n, 1.0, rng)) * 0.3
        wobble = 1.0 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.3, 2.0) * np.arange(n) / SR
                                     + rng.uniform(0, 2 * np.pi))
        x = (rumble + whoosh) * wobble

    elif kind == "babble":
        # Crowd chatter: 5 "voices", each = speech-band noise chopped by a
        # syllable-rate (3-6 Hz) envelope. Not real words, but it has the same
        # on/off texture that makes real babble hard for a recogniser.
        x = np.zeros(n)
        for _ in range(5):
            voice = _band_limit(_colored_noise(n, 1.0, rng), 200, 3400)
            syllables = _slow_envelope(n, rng.uniform(3.0, 6.0), rng) ** 2
            x += voice * syllables

    elif kind == "clicks":
        # Claps / door bangs: a few short decaying bursts on near-silence.
        x = 0.002 * rng.standard_normal(n)
        for _ in range(int(rng.integers(3, 12))):
            length = int(rng.uniform(0.005, 0.03) * SR)
            start = int(rng.integers(0, max(n - length, 1)))
            burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.3 * length))
            x[start:start + length] += rng.uniform(0.5, 1.5) * burst[: n - start]

    else:
        raise ValueError(f"Unknown noise kind: {kind}")

    return _unit_rms(x)


# ===========================================================================
# 2. REAL NOISE RECORDINGS
# ===========================================================================
def load_wav_mono(path):
    """Load a WAV as float32 mono in [-1, 1]. Returns None if not 16 kHz."""
    try:
        rate, data = wavfile.read(path)
    except Exception:
        return None
    if rate != SR:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    return data


def load_real_noise_pool():
    """
    Collect REAL noise recordings:
      * dataset/noise/*.wav                       <- recorded in YOUR room
      * dataset/speech_commands/_background_noise_ <- Google's noise clips (if downloaded)
    Returns a list of float32 arrays (can be empty - synthetic noise still works).
    """
    folders = [config_ext.NOISE_DIR,
               os.path.join(config.SPEECH_COMMANDS_DIR, "_background_noise_")]
    pool = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".wav"):
                audio = load_wav_mono(os.path.join(folder, name))
                if audio is not None and len(audio) >= SR // 2:
                    pool.append(audio)
    return pool


def _real_slice(pool, n, rng):
    """Cut a random n-sample slice out of a random real recording (unit RMS)."""
    rec = pool[int(rng.integers(len(pool)))]
    if len(rec) <= n:
        reps = int(np.ceil(n / len(rec)))          # loop short recordings
        rec = np.tile(rec, reps)
    start = int(rng.integers(0, len(rec) - n + 1))
    chunk = rec[start:start + n]
    if np.sqrt(np.mean(chunk ** 2)) < 1e-4:        # (almost) digital silence
        return None
    return _unit_rms(chunk)


def random_noise(rng, real_pool=None, n=config.CLIP_SAMPLES, kind=None,
                 p_real=0.5, exclude=()):
    """
    Pick a noise source and return (name, unit-RMS noise array of length n).

      kind=None    -> random: real recording (probability p_real, if any exist)
                      or a random synthetic kind
      kind="real"  -> force a real recording
      kind="fan"   -> force that synthetic kind
      exclude      -> synthetic kinds to avoid (e.g. ("clicks",))
    """
    real_pool = real_pool or []

    if kind == "real" or (kind is None and real_pool and rng.random() < p_real):
        if real_pool:
            chunk = _real_slice(real_pool, n, rng)
            if chunk is not None:
                return "real", chunk
        kind = None                                  # fall back to synthetic

    if kind is None or kind == "real":
        choices = [k for k in config_ext.NOISE_KINDS if k not in exclude]
        kind = choices[int(rng.integers(len(choices)))]
    return kind, synth_noise(kind, n, rng)


# ===========================================================================
# 3. MIXING AT A TARGET SNR
# ===========================================================================
def mix_at_snr(clean, noise_unit_rms, snr_db):
    """
    Return clean + noise, with the noise scaled to give exactly `snr_db`.

        noise_power_needed = speech_power / 10^(snr/10)
        noise_gain         = sqrt(noise_power_needed)       (noise is unit RMS)

    If the sum would clip (peak > 0.99) the WHOLE mix is turned down together,
    which keeps the SNR unchanged.
    """
    clean = np.asarray(clean, dtype=np.float32)
    speech_power = float(np.mean(clean ** 2)) + 1e-12
    gain = np.sqrt(speech_power / (10.0 ** (snr_db / 10.0)))
    mix = clean + gain * noise_unit_rms[: len(clean)]
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    return mix.astype(np.float32)


# Quick self-test: python noise_utils.py
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(config.CLIP_SAMPLES) / SR
    fake_speech = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for kind in config_ext.NOISE_KINDS:
        noise = synth_noise(kind, config.CLIP_SAMPLES, rng)
        for snr in (20, 0):
            mix = mix_at_snr(fake_speech, noise, snr)
            print(f"{kind:7s} unit-rms={np.sqrt(np.mean(noise**2)):.3f}  "
                  f"mix peak@{snr:>2d}dB={np.max(np.abs(mix)):.3f}")
    # Verify the SNR maths on a case with no clipping.
    noise = synth_noise("white", config.CLIP_SAMPLES, rng)
    mix = mix_at_snr(fake_speech, noise, 10)
    measured = 10 * np.log10(np.mean(fake_speech ** 2) / np.mean((mix - fake_speech) ** 2))
    print(f"\nRequested 10 dB, measured {measured:.2f} dB")
    assert abs(measured - 10) < 0.2
    print("OK - noise_utils.py works")
