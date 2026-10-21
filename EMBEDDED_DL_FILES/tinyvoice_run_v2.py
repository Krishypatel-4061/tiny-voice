"""
tinyvoice_run_v2.py
===================
The LIVE program with the two new features. Your original tinyvoice_run.py is
untouched and remains your safe fallback for the demo.

WHAT CHANGED vs tinyvoice_run.py
--------------------------------
Nothing in the decision logic. The recognition chain now lives in pipeline.py
(shared with benchmark_cascade.py) and you switch the new features on with flags:

    --cascade   Option B: a tiny learned VAD decides "is this speech?" BEFORE the
                expensive stages run (saves CPU, rejects loud non-speech noise)
    --enhance   Option A: a learned denoiser cleans the MFCC features before the
                DS-CNN sees them (helps in noisy rooms)

USAGE
-----
    python tinyvoice_run_v2.py --no-gpio --debug               # laptop, nothing new
    python tinyvoice_run_v2.py --cascade --enhance --debug     # everything on
    python tinyvoice_run_v2.py --cascade                       # only Option B
    python tinyvoice_run_v2.py --enhance --enhancer-int8       # int8 denoiser
    python tinyvoice_run_v2.py                                 # same as the original
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import config
import config_ext
# pyrefly: ignore[missing-import]
from audio_input import MicrophoneStream
# pyrefly: ignore[missing-import]
from gpio_control import ActuatorController
from pipeline import Pipeline
# pyrefly: ignore[missing-import]
from tflite_engine import TinyVoiceEngine
from tflite_runner import TFLiteRunner
# pyrefly: ignore[missing-import]
from tinyvoice_run import OledDisplay, _sleep_remaining   # reuse, do not duplicate


def load_vad_threshold(cli_value):
    """command line > value saved by train_vad.py > default"""
    if cli_value is not None:
        return cli_value
    if os.path.exists(config_ext.VAD_META_PATH):
        with open(config_ext.VAD_META_PATH) as f:
            return float(json.load(f)["threshold"])
    return config_ext.VAD_DEFAULT_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="TinyVoice live recognition (v2)")
    parser.add_argument("--model", default=config.TFLITE_INT8_PATH)
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--rms-threshold", type=float, default=config.RMS_SILENCE_THRESHOLD)
    parser.add_argument("--no-gpio", action="store_true", help="simulate actuators")
    parser.add_argument("--debug", action="store_true", help="live status line")
    parser.add_argument("--device", type=int, default=None, help="audio input device index")
    parser.add_argument("--cascade", action="store_true", help="enable the VAD gate (Option B)")
    parser.add_argument("--vad-model", default=config_ext.VAD_FLOAT_PATH)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--enhance", action="store_true", help="enable the denoiser (Option A)")
    parser.add_argument("--enhancer-int8", action="store_true",
                        help="use the int8 denoiser instead of float32")
    parser.add_argument("--enhancer-model", default=None)
    args = parser.parse_args()

    print("=" * 66)
    print("  TinyVoice v2 - Edge ML speech commands on Raspberry Pi 4")
    print("=" * 66)

    engine = TinyVoiceEngine(model_path=args.model)
    print("DS-CNN     :", engine.describe())

    vad = enhancer = None
    vad_thr = None
    if args.cascade:
        vad = TFLiteRunner(args.vad_model)
        vad_thr = load_vad_threshold(args.vad_threshold)
        print(f"VAD (B)    : {vad.describe()} | threshold {vad_thr:.2f}")
    if args.enhance:
        path = args.enhancer_model or (config_ext.ENHANCER_INT8_PATH if args.enhancer_int8
                                       else config_ext.ENHANCER_FLOAT_PATH)
        enhancer = TFLiteRunner(path)
        print("Denoiser(A):", enhancer.describe())
    print(f"Features   : cascade={'ON' if vad else 'off'}  denoiser={'ON' if enhancer else 'off'}")

    pipe = Pipeline(engine, enhancer=enhancer, vad=vad,
                    vad_threshold=vad_thr if vad_thr is not None else 0.4,
                    confidence_threshold=args.threshold,
                    rms_threshold=args.rms_threshold)

    actuators = ActuatorController(simulate=args.no_gpio or None)
    oled = OledDisplay(config.USE_OLED)
    mic = MicrophoneStream(device=args.device)

    print("\nListening... (Ctrl+C to stop)\n")
    try:
        mic.start()
        time.sleep(config.CLIP_DURATION_S)         # let the ring buffer fill

        while True:
            loop_start = time.perf_counter()
            window = mic.get_window()
            res = pipe.process(window, time.time())

            if args.debug:
                if res["stage"] == "classified":
                    top3 = " | ".join(f"{l}:{c:.2f}" for l, c in res["top3"])
                    line = f"  [NN ] {top3}"
                elif res["stage"] == "vad_rejected":
                    line = f"  [VAD] not speech (p={res['vad_prob']:.2f})"
                else:
                    line = "  [   ] silent"
                print(f"{line:<60s}", end="\r", flush=True)

            if res["command"] is not None:
                result = actuators.handle_command(res["command"])
                print(f"\n[{time.strftime('%H:%M:%S')}] >>> {res['command'].upper():6s} "
                      f"({res['confidence'] * 100:.1f}%)  ->  {result}")
                print(f"           {actuators.status_line()}")
                oled.show(["TinyVoice", f"CMD: {res['command'].upper()}",
                           f"Conf: {res['confidence'] * 100:.0f}%", result[:21]])

            _sleep_remaining(loop_start)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        mic.stop()
        actuators.cleanup()
        print("\n--- SESSION SUMMARY ---")
        print(pipe.summary())
        if mic.overflow_count:
            print(f"Audio overflows    : {mic.overflow_count} "
                  "(CPU struggling - raise INFERENCE_HOP_S in config.py)")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
