"""
eval_noise_robustness.py   (OPTION A, step 3 - THE RESULT FOR YOUR REPORT)
==========================================================================
Answers the question: "Does the learned denoiser make the recogniser survive
noise?"

THE EXPERIMENT
--------------
Take the held-out CLEAN test clips (never used in training any model), add noise
at several SNR levels, then classify each noisy clip THREE ways:

    1. BASELINE      : noisy MFCC  -> DS-CNN                      (your existing system)
    2. + DENOISER f32: noisy MFCC  -> denoiser (float32) -> DS-CNN
    3. + DENOISER i8 : noisy MFCC  -> denoiser (int8)    -> DS-CNN

All three see the EXACT SAME noisy audio, so the comparison is fair. We print an
accuracy table, save it as CSV, and draw the accuracy-vs-SNR plot.

IMPORTANT HONESTY NOTES (say these in your report, examiners respect it)
  * Accuracy here = argmax over the 7 classes equals the true command.
  * With few speakers the test clips may be from a voice the DS-CNN has heard.
    The COMPARISON (baseline vs denoiser) is still fair, but the absolute
    numbers are optimistic until you have >= 3 speakers recorded.
  * Synthetic noise is only an approximation of your real room. Add real room
    recordings (record_noise.py) and re-run for a more convincing result.

USAGE
-----
    python eval_noise_robustness.py
    python eval_noise_robustness.py --repeats 5 --per-noise --per-noise-snr 5
"""

import argparse
import csv
import os
import time

import numpy as np

import config
import config_ext
import features as feat
import noise_utils as nu
from tflite_engine import TinyVoiceEngine
from tflite_runner import TFLiteRunner


def evaluate_condition(engine, enhancers, audios, labels, snr_db, kind, repeats,
                       real_pool, seed):
    """
    Accuracy of every system under ONE noise condition.

    enhancers : dict  name -> TFLiteRunner
    snr_db    : None means "clean, no noise added"
    kind      : None = random mix of noise kinds, or a specific kind name
    Returns   : dict  system_name -> accuracy in [0, 1]
    """
    rng = np.random.default_rng(seed)
    systems = ["baseline"] + list(enhancers.keys())
    correct = {s: 0 for s in systems}
    total = 0

    for _ in range(1 if snr_db is None else repeats):
        for audio, label in zip(audios, labels):
            if snr_db is None:
                noisy = audio
            else:
                _, noise = nu.random_noise(rng, real_pool, kind=kind,
                                           exclude=("clicks",) if kind is None else ())
                noisy = nu.mix_at_snr(audio, noise, snr_db)

            mfcc = feat.audio_to_mfcc(noisy)
            if int(np.argmax(engine.predict(mfcc))) == label:
                correct["baseline"] += 1
            for name, runner in enhancers.items():
                cleaned = runner.predict(mfcc)
                if int(np.argmax(engine.predict(cleaned))) == label:
                    correct[name] += 1
            total += 1

    return {s: correct[s] / max(total, 1) for s in systems}


def measure_latency(runner, runs=200):
    """Median milliseconds for one denoiser call."""
    dummy = np.random.randn(*config.FEATURE_SHAPE).astype(np.float32)
    for _ in range(20):
        runner.predict(dummy)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        runner.predict(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3,
                        help="noisy versions generated per test clip per SNR")
    parser.add_argument("--per-noise", action="store_true",
                        help="also print a table per noise type")
    parser.add_argument("--per-noise-snr", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    # ---------------- load things -------------------------------------------
    audio_path = os.path.join(config_ext.ENH_DIR, "test_audio.npy")
    if not os.path.exists(audio_path):
        raise SystemExit("Run build_enhancer_data.py first (it creates the test set).")
    audios = np.load(audio_path)
    labels = np.load(os.path.join(config_ext.ENH_DIR, "test_labels.npy"))
    print(f"Test clips: {len(audios)}")

    engine = TinyVoiceEngine()
    print("DS-CNN :", engine.describe())

    enhancers = {}
    if os.path.exists(config_ext.ENHANCER_FLOAT_PATH):
        enhancers["+denoiser (float32)"] = TFLiteRunner(config_ext.ENHANCER_FLOAT_PATH)
    if os.path.exists(config_ext.ENHANCER_INT8_PATH):
        enhancers["+denoiser (int8)"] = TFLiteRunner(config_ext.ENHANCER_INT8_PATH)
    if not enhancers:
        raise SystemExit("No denoiser models found. Run train_enhancer.py first.")
    for name, r in enhancers.items():
        print(f"{name:20s}: {r.describe()}  | {measure_latency(r):.3f} ms/call")

    real_pool = nu.load_real_noise_pool()
    print(f"Real noise recordings available: {len(real_pool)}\n")

    systems = ["baseline"] + list(enhancers.keys())

    # ---------------- main table: accuracy vs SNR ----------------------------
    rows = []
    conditions = [("clean", None)] + [(f"{s} dB", float(s)) for s in config_ext.EVAL_SNRS_DB]
    header = f"{'condition':>10s}" + "".join(f"{s:>22s}" for s in systems)
    print("ACCURACY vs NOISE LEVEL (random mix of noise types)")
    print(header)
    print("-" * len(header))
    for label, snr in conditions:
        acc = evaluate_condition(engine, enhancers, audios, labels, snr, None,
                                 args.repeats, real_pool, args.seed)
        rows.append({"condition": label, **{s: round(100 * acc[s], 2) for s in systems}})
        print(f"{label:>10s}" + "".join(f"{100 * acc[s]:>21.1f}%" for s in systems))

    # ---------------- optional: per noise type -------------------------------
    per_noise_rows = []
    if args.per_noise:
        print(f"\nACCURACY PER NOISE TYPE at {args.per_noise_snr:g} dB")
        print(f"{'noise':>10s}" + "".join(f"{s:>22s}" for s in systems))
        kinds = [k for k in config_ext.NOISE_KINDS if k != "clicks"]
        if real_pool:
            kinds.append("real")
        for kind in kinds:
            acc = evaluate_condition(engine, enhancers, audios, labels,
                                     args.per_noise_snr, kind, args.repeats,
                                     real_pool, args.seed)
            per_noise_rows.append({"noise": kind, **{s: round(100 * acc[s], 2) for s in systems}})
            print(f"{kind:>10s}" + "".join(f"{100 * acc[s]:>21.1f}%" for s in systems))

    # ---------------- save CSV + plot ------------------------------------------
    os.makedirs(config_ext.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config_ext.RESULTS_DIR, "noise_robustness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition"] + systems)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved table -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(len(rows)))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in systems:
            ax.plot(x, [r[s] for r in rows], marker="o", label=s)
        ax.set_xticks(x)
        ax.set_xticklabels([r["condition"] for r in rows])
        ax.set_xlabel("Noise level (SNR, lower = noisier)")
        ax.set_ylabel("Command accuracy (%)")
        ax.set_title("TinyVoice: accuracy under noise, with vs without learned denoiser")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(config_ext.RESULTS_DIR, "noise_robustness.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved plot  -> {png_path}")
    except ImportError:
        print("(matplotlib not installed - skipped the plot)")

    print("\nPut the table and plot straight into your report's results section.")


if __name__ == "__main__":
    main()
