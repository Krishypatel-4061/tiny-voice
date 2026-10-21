"""
benchmark_cascade.py   (OPTION B, step 2 - THE RESULT FOR YOUR REPORT)
======================================================================
Compares four system configurations on the SAME simulated noisy-room recording:

    1. baseline            : RMS gate -> DS-CNN               (your original system)
    2. + denoiser          : RMS gate -> denoiser -> DS-CNN             (Option A)
    3. + VAD cascade       : RMS gate -> VAD -> DS-CNN                  (Option B)
    4. + VAD + denoiser    : RMS gate -> VAD -> denoiser -> DS-CNN      (A + B)

THE SIMULATED STREAM (a few minutes of "room" built in seconds)
  * continuous background noise (fan / hum / hiss / babble / real recordings),
    deliberately LOUDER than the RMS gate threshold - like a real classroom,
    where a loudness gate is permanently open and saves nothing
  * held-out command clips dropped in at known times            (ground truth)
  * loud NON-speech distractors (claps, bangs, bursts) at known times - things
    that must NOT trigger any action

For each configuration we replay the stream through `Pipeline.process()` (the
exact code the live program uses), sliding a 1 s window every 0.25 s, and report:

    NN windows run     how often the expensive classifier actually ran
    CPU x real-time    CPU seconds spent per second of audio (lower = cheaper;
                       < 1.0 means it keeps up in real time with room to spare)
    recall             fraction of real commands correctly executed
    false triggers     actions fired when NO command was spoken (or wrong word)

RUN IT ON THE PI to get the numbers that matter: CPU cost is hardware-specific.

USAGE
-----
    python benchmark_cascade.py
    python benchmark_cascade.py --snr 5 --noise-level 0.05 --duration 180
"""

import argparse
import csv
import os
import time

import numpy as np

import config
import config_ext
import noise_utils as nu
from pipeline import Pipeline
# pyrefly: ignore[missing-import]
from tflite_engine import TinyVoiceEngine
from tflite_runner import TFLiteRunner

SR = config.SAMPLE_RATE


# ===========================================================================
# 1. BUILD THE SIMULATED ROOM RECORDING
# ===========================================================================
def build_stream(duration_s, n_commands, n_distractors, noise_level, snr_db,
                 distractor_level, audios, labels, real_pool, seed):
    """
    Returns (stream, events) where
        stream : float32 array (the whole recording)
        events : list of dicts {"t": start_seconds, "kind": "command"/"distractor",
                                "label": label_index or None}
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)

    # ---- continuous background, built from 5-second pieces ---------------------
    stream = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        take = min(5 * SR, n - pos)
        _, noise = nu.random_noise(rng, real_pool, n=take,
                                   exclude=("clicks",))
        stream[pos:pos + take] = noise * noise_level
        pos += take

    # ---- choose event times: slots 3 s apart so events never overlap --------------
    slots = np.arange(2.0, duration_s - 3.0, 3.0)
    total_events = min(n_commands + n_distractors, len(slots))
    chosen = rng.choice(slots, size=total_events, replace=False)
    chosen.sort()
    kinds = (["command"] * n_commands + ["distractor"] * n_distractors)[:total_events]
    rng.shuffle(kinds)

    events = []
    for t, kind in zip(chosen, kinds):
        start = int((t + rng.uniform(-0.3, 0.3)) * SR)
        if kind == "command":
            i = int(rng.integers(len(audios)))
            clip = audios[i].copy()
            # Scale the clip so (clip RMS) / (background RMS) matches the SNR.
            clip_rms = np.sqrt(np.mean(clip ** 2)) + 1e-9
            clip *= (noise_level * 10 ** (snr_db / 20.0)) / clip_rms
            end = min(start + len(clip), n)
            stream[start:end] += clip[: end - start]
            events.append({"t": start / SR, "kind": "command", "label": int(labels[i])})
        else:
            # A 0.5 s loud NON-speech burst with soft fade-in/out.
            kind_name = ["white", "clicks", "pink", "babble"][int(rng.integers(4))]
            length = SR // 2
            burst = nu.synth_noise(kind_name, length, rng) * np.hanning(length)
            burst *= distractor_level
            end = min(start + length, n)
            stream[start:end] += burst[: end - start]
            events.append({"t": start / SR, "kind": "distractor", "label": None})

    return np.clip(stream, -1.0, 1.0).astype(np.float32), events


# ===========================================================================
# 2. REPLAY A STREAM THROUGH A PIPELINE AND SCORE IT
# ===========================================================================
def replay(pipe, stream, events):
    """Slide a 1 s window every INFERENCE_HOP_S and score the detections."""
    pipe.reset()
    hop = int(config.INFERENCE_HOP_S * SR)
    detections = []                                   # (time_of_window_end, label_name)

    cpu_start = time.process_time()
    for end in range(SR, len(stream) + 1, hop):
        now = end / SR
        result = pipe.process(stream[end - SR:end], now)
        if result["command"] is not None:
            detections.append((now, result["command"]))
    cpu = time.process_time() - cpu_start

    # ---- scoring ------------------------------------------------------------------
    # A detection at window-end time t is CORRECT for a command event starting at
    # s if the label matches and s+0.3 <= t <= s+2.0 (the word is inside the window).
    matched_events = set()
    false_triggers = 0
    for t, label_name in detections:
        hit = None
        for k, ev in enumerate(events):
            if ev["kind"] == "command" and k not in matched_events \
                    and ev["t"] + 0.3 <= t <= ev["t"] + 2.0 \
                    and config.LABELS[ev["label"]] == label_name:
                hit = k
                break
        if hit is None:
            false_triggers += 1
        else:
            matched_events.add(hit)

    n_cmd = sum(1 for e in events if e["kind"] == "command")
    audio_seconds = len(stream) / SR
    return {
        "nn_windows": pipe.counts["classified"],
        "total_windows": pipe.counts["windows"],
        "cpu_x_realtime": cpu / audio_seconds,
        "recall": len(matched_events) / max(n_cmd, 1),
        "hits": len(matched_events),
        "commands": n_cmd,
        "false_triggers": false_triggers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0, help="seconds of audio")
    parser.add_argument("--commands", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=12)
    parser.add_argument("--noise-level", type=float, default=0.03,
                        help="background RMS. Default is ABOVE the RMS gate (noisy room)")
    parser.add_argument("--snr", type=float, default=10.0, help="command SNR vs background (dB)")
    parser.add_argument("--distractor-level", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--enhancer-model", default=config_ext.ENHANCER_FLOAT_PATH)
    parser.add_argument("--vad-model", default=config_ext.VAD_FLOAT_PATH)
    parser.add_argument("--vad-threshold", type=float, default=None)
    args = parser.parse_args()

    audio_path = os.path.join(config_ext.ENH_DIR, "test_audio.npy")
    if not os.path.exists(audio_path):
        raise SystemExit("Run build_enhancer_data.py first (it creates the test clips).")
    audios = np.load(audio_path)
    labels = np.load(os.path.join(config_ext.ENH_DIR, "test_labels.npy"))

    # VAD threshold: command line > saved by train_vad.py > default.
    thr = args.vad_threshold
    if thr is None:
        if os.path.exists(config_ext.VAD_META_PATH):
            import json
            with open(config_ext.VAD_META_PATH) as f:
                thr = json.load(f)["threshold"]
        else:
            thr = config_ext.VAD_DEFAULT_THRESHOLD

    engine = TinyVoiceEngine()
    enhancer = TFLiteRunner(args.enhancer_model)
    vad = TFLiteRunner(args.vad_model)
    print("DS-CNN   :", engine.describe())
    print("Denoiser :", enhancer.describe())
    print("VAD      :", vad.describe(), f"| threshold {thr:.2f}")

    real_pool = nu.load_real_noise_pool()
    stream, events = build_stream(args.duration, args.commands, args.distractors,
                                  args.noise_level, args.snr, args.distractor_level,
                                  audios, labels, real_pool, args.seed)
    n_cmd = sum(e["kind"] == "command" for e in events)
    n_dis = sum(e["kind"] == "distractor" for e in events)
    print(f"\nSimulated room: {args.duration:.0f}s, background RMS {args.noise_level}, "
          f"commands at {args.snr:g} dB SNR")
    print(f"Events: {n_cmd} real commands + {n_dis} loud non-speech distractors "
          f"(RMS gate threshold = {config.RMS_SILENCE_THRESHOLD})\n")

    configs = [
        ("1. baseline (RMS gate)",       None,     None),
        ("2. + denoiser (A)",            enhancer, None),
        ("3. + VAD cascade (B)",         None,     vad),
        ("4. + VAD + denoiser (A+B)",    enhancer, vad),
    ]

    rows = []
    header = (f"{'configuration':<28s}{'NN windows':>14s}{'CPU x real-time':>17s}"
              f"{'recall':>10s}{'false trig.':>13s}")
    print(header)
    print("-" * len(header))
    for name, enh, vd in configs:
        pipe = Pipeline(engine, enhancer=enh, vad=vd, vad_threshold=thr)
        r = replay(pipe, stream, events)
        rows.append({"configuration": name, **r})
        print(f"{name:<28s}{r['nn_windows']:>6d}/{r['total_windows']:<7d}"
              f"{r['cpu_x_realtime']:>17.4f}{100 * r['recall']:>9.0f}%"
              f"{r['false_triggers']:>13d}")

    base = rows[0]
    print("\nHOW TO READ THIS")
    print("  * 'NN windows' = windows that reached the DS-CNN. The cascade should cut this a lot.")
    print("  * 'CPU x real-time' = CPU seconds per audio second. Lower = less CPU/power.")
    for r in rows[1:]:
        if base["cpu_x_realtime"] > 0:
            change = 100 * (1 - r["cpu_x_realtime"] / base["cpu_x_realtime"])
            cpu_text = (f"{change:.0f}% LESS CPU" if change >= 0 else f"{-change:.0f}% MORE CPU")
            print(f"  * {r['configuration']:<28s} {cpu_text} than baseline | "
                  f"recall {100 * r['recall']:.0f}% (baseline {100 * base['recall']:.0f}%) | "
                  f"false triggers {r['false_triggers']} (baseline {base['false_triggers']})")

    os.makedirs(config_ext.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config_ext.RESULTS_DIR, "benchmark_cascade.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {path}")
    print("Run this ON THE RASPBERRY PI for the real CPU numbers.")


if __name__ == "__main__":
    main()
