"""
tools/test_audio.py
===================
RUN THIS FIRST on hardware day, before anything else.

It answers the three questions that block every audio project:
  1. Does the Pi see my microphone, and at which device index?
  2. Is it actually capturing sound (not just silence)?
  3. What RMS level does my voice produce, so I can set RMS_SILENCE_THRESHOLD?

USAGE (from the project root)
    python tools/test_audio.py --list
    python tools/test_audio.py --meter
    python tools/test_audio.py --record test.wav --device 1
"""

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Allow importing the project modules from the parent folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                       # noqa: E402
import features as feat             # noqa: E402


def list_devices():
    print("ALL AUDIO DEVICES")
    print(sd.query_devices())
    print("\nUse the number in the leftmost column as --device N")
    print("An input device is one with 'in' channels > 0.")


def level_meter(device, seconds=30):
    """
    Live loudness bar. Watch the numbers while you:
      - stay silent  -> note the RMS. Set RMS_SILENCE_THRESHOLD ~3x that value.
      - say "light"  -> RMS should jump well above it.
      - shout        -> peak should stay below 1.00 (no clipping).
    """
    print(f"\nLevel meter for {seconds}s. Ctrl+C to stop early.\n")
    print(f"{'RMS':>8s} {'PEAK':>8s}  meter")

    end = time.time() + seconds
    try:
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                            dtype="float32", device=device,
                            blocksize=int(config.SAMPLE_RATE * 0.1)) as stream:
            while time.time() < end:
                block, overflowed = stream.read(int(config.SAMPLE_RATE * 0.1))
                audio = block[:, 0]
                r = feat.rms(audio)
                peak = float(np.max(np.abs(audio)))

                bars = int(min(r * 200, 40))
                flag = ""
                if peak > 0.99:
                    flag = "  <-- CLIPPING! reduce gain"
                elif overflowed:
                    flag = "  <-- overflow"
                print(f"{r:8.4f} {peak:8.3f}  {'#' * bars:<40s}{flag}",
                      end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    print("\n\nSuggested config.py value:")
    print("  RMS_SILENCE_THRESHOLD = about 3x your SILENT-room RMS")


def record_test(path, device, seconds=3):
    print(f"Recording {seconds}s... speak now.")
    audio = sd.rec(int(seconds * config.SAMPLE_RATE),
                   samplerate=config.SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    audio = audio.flatten()

    wavfile.write(path, config.SAMPLE_RATE,
                  (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    print(f"Saved {path}")
    print(f"  RMS  : {feat.rms(audio):.4f}")
    print(f"  Peak : {np.max(np.abs(audio)):.3f}")

    if np.max(np.abs(audio)) < 0.01:
        print("\n  WARNING: this is essentially silence.")
        print("  - wrong --device index?")
        print("  - USB mic muted? run:  alsamixer  and raise the capture level")
        print("  - I2S mic wiring or dtoverlay wrong? see the manual")
    else:
        print("\n  Looks good. Play it back with:  aplay " + path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--meter", action="store_true")
    p.add_argument("--record", metavar="OUT.wav")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--seconds", type=float, default=3)
    a = p.parse_args()

    if a.list or not (a.meter or a.record):
        list_devices()
    if a.meter:
        level_meter(a.device)
    if a.record:
        record_test(a.record, a.device, a.seconds)
