"""
record_noise.py
===============
Records REAL background noise from the room where you will demo.

WHY THIS IS THE MOST VALUABLE 5 MINUTES IN THE ADD-ON
-----------------------------------------------------
Synthetic noise (white, hum, fan...) is a good approximation, but nothing beats
the real thing. Real noise recordings are automatically picked up by
build_enhancer_data.py, train_vad.py, eval_noise_robustness.py and
benchmark_cascade.py (they all read dataset/noise/*.wav).

WHAT TO RECORD - one 30-60 s file for each situation:
    * the room with the fan/AC running
    * the room with people talking quietly in the background
    * the motor and servo RUNNING (your own hardware is a noise source!)
    * a door closing, chairs moving, keyboard typing

DO NOT SPEAK COMMANDS while recording - this must be "no command" audio.

USAGE (from the project root)
    python record_noise.py --name fan_on --seconds 45
    python record_noise.py --name motor_running --seconds 30 --device 1
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

import config
import config_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="room", help="short label for the file name")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(config_ext.NOISE_DIR, exist_ok=True)
    print(f"Recording {args.seconds:.0f}s of '{args.name}' - stay quiet, let the room be itself.")
    for i in (3, 2, 1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1)

    audio = sd.rec(int(args.seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE,
                   channels=1, dtype="float32", device=args.device)
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    path = os.path.join(config_ext.NOISE_DIR, f"{args.name}_{int(time.time())}.wav")
    wavfile.write(path, config.SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))

    print(f"\nSaved {path}")
    print(f"  RMS {rms:.4f}   peak {peak:.3f}")
    print(f"  (your RMS_SILENCE_THRESHOLD is {config.RMS_SILENCE_THRESHOLD}; "
          f"{'this room is ABOVE it - the loudness gate will stay open, so the VAD matters here' if rms > config.RMS_SILENCE_THRESHOLD else 'this room is below it'})")
    if peak < 0.005:
        print("  WARNING: nearly silent - check the mic / --device index.")


if __name__ == "__main__":
    main()
