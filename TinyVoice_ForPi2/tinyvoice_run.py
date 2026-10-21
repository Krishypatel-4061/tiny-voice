"""
tinyvoice_run.py
================
THE MAIN PROGRAM. Run this on the Raspberry Pi for the live demo.

WHAT IT DOES, EVERY 250 ms
--------------------------
    1. grab the most recent 1 second of audio from the ring buffer
    2. cheap loudness gate  - if the room is quiet, skip everything else
    3. MFCC feature extraction
    4. neural network inference -> 7 probabilities
    5. posterior smoothing      - average with the previous few results
    6. confidence threshold     - act only if >= 0.80
    7. refractory period        - ignore new triggers for 1.5 s after acting
    8. drive the LED / motor / servo

WHY STEPS 5-7 EXIST (this is the part most student projects get wrong)
----------------------------------------------------------------------
Because we classify 4 times per second on OVERLAPPING windows, a single spoken
word produces several consecutive detections. Naively acting on each one would
toggle your LED three times for one "light".

  * SMOOTHING averages the last 3 probability vectors. A real word stays strong
    across consecutive windows; a random blip does not survive the averaging.
    This is the standard "posterior handling" from Google's original
    keyword-spotting paper.
  * The REFRACTORY PERIOD then guarantees one action per utterance.

USAGE
-----
    python tinyvoice_run.py                  # normal demo
    python tinyvoice_run.py --debug          # show live top-3 predictions
    python tinyvoice_run.py --no-gpio        # laptop testing, prints actions
    python tinyvoice_run.py --threshold 0.9  # stricter
"""

import argparse
import collections
import sys
import time

import numpy as np

import config
import features as feat
from audio_input import MicrophoneStream
from gpio_control import ActuatorController
from tflite_engine import TinyVoiceEngine


# ---------------------------------------------------------------------------
# Optional OLED display
# ---------------------------------------------------------------------------
class OledDisplay:
    """
    Thin wrapper around an SSD1306 I2C OLED.
    Every call is wrapped so a missing or unwired display can never crash the
    demo - it just silently disables itself.
    """

    def __init__(self, enabled):
        self.ok = False
        if not enabled:
            return
        try:
            from board import SCL, SDA
            import busio
            import adafruit_ssd1306
            from PIL import Image, ImageDraw, ImageFont

            i2c = busio.I2C(SCL, SDA)
            self.disp = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
            self.Image, self.ImageDraw = Image, ImageDraw
            self.font = ImageFont.load_default()
            self.ok = True
            print("[OLED] Display ready.")
        except Exception as e:
            print(f"[OLED] Not available ({e}). Continuing without it.")

    def show(self, lines):
        if not self.ok:
            return
        try:
            img = self.Image.new("1", (128, 64))
            draw = self.ImageDraw.Draw(img)
            for i, line in enumerate(lines[:5]):
                draw.text((0, i * 12), line[:21], font=self.font, fill=255)
            self.disp.image(img)
            self.disp.show()
        except Exception:
            self.ok = False


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TinyVoice live recognition")
    parser.add_argument("--model", default=config.TFLITE_INT8_PATH)
    parser.add_argument("--threshold", type=float,
                        default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-gpio", action="store_true",
                        help="Simulate actuators (for laptop testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Print top-3 predictions for every window")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (see tools/test_audio.py)")
    parser.add_argument("--rms-threshold", type=float,
                        default=config.RMS_SILENCE_THRESHOLD)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Set-up
    # -----------------------------------------------------------------------
    print("=" * 66)
    print("  TinyVoice - offline speech command recognition on Raspberry Pi 4")
    print("=" * 66)

    engine = TinyVoiceEngine(model_path=args.model)
    print("Model      :", engine.describe())

    actuators = ActuatorController(simulate=args.no_gpio or None)
    oled = OledDisplay(config.USE_OLED)

    print(f"Commands   : {', '.join(config.COMMAND_WORDS)}")
    print(f"Threshold  : {args.threshold}")
    print(f"Window     : {config.CLIP_DURATION_S}s, evaluated every "
          f"{config.INFERENCE_HOP_S}s")
    print(f"Smoothing  : average of last {config.SMOOTHING_WINDOW} windows")
    print("\nListening... (press Ctrl+C to stop)\n")

    # A deque with maxlen automatically discards the oldest entry when full -
    # exactly the behaviour we want for a rolling average.
    prob_history = collections.deque(maxlen=config.SMOOTHING_WINDOW)

    last_action_time = 0.0      # when we last fired a command
    window_count = 0
    detection_count = 0
    latencies = []

    mic = MicrophoneStream(device=args.device)

    try:
        mic.start()
        # Let the ring buffer fill with real audio before the first inference,
        # otherwise the first window is mostly startup silence.
        time.sleep(config.CLIP_DURATION_S)

        while True:
            loop_start = time.perf_counter()
            window_count += 1

            audio = mic.get_window()
            level = feat.rms(audio)

            # --- Gate 1: loudness ------------------------------------------
            if level < args.rms_threshold:
                # The room is quiet. Clear the history so that stale
                # probabilities from a previous word cannot combine with a new
                # one and produce a phantom detection.
                prob_history.clear()
                if args.debug:
                    print(f"  [silent] rms={level:.4f}", end="\r", flush=True)
                _sleep_remaining(loop_start)
                continue

            # --- Inference --------------------------------------------------
            t0 = time.perf_counter()
            mfcc = feat.audio_to_mfcc(audio)
            probs = engine.predict(mfcc)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            # --- Gate 2: smoothing ------------------------------------------
            prob_history.append(probs)
            smoothed = np.mean(np.stack(prob_history, axis=0), axis=0)

            label, confidence = engine.top_prediction(smoothed)

            if args.debug:
                top3 = " | ".join(f"{l}:{c:.2f}"
                                  for l, c in engine.top_k(smoothed, 3))
                print(f"  rms={level:.3f}  {top3}          ", end="\r", flush=True)

            # --- Gate 3: is it an actual command, confidently? --------------
            is_command = label in config.COMMAND_WORDS
            confident = confidence >= args.threshold

            # --- Gate 4: refractory period ----------------------------------
            now = time.time()
            cooled_down = (now - last_action_time) >= config.REFRACTORY_S

            # We also require a full smoothing window of evidence, so a command
            # cannot fire on a single frame right after silence.
            enough_history = len(prob_history) >= config.SMOOTHING_WINDOW

            if is_command and confident and cooled_down and enough_history:
                detection_count += 1
                last_action_time = now
                result = actuators.handle_command(label)

                stamp = time.strftime("%H:%M:%S")
                print(f"\n[{stamp}] >>> {label.upper():6s} "
                      f"({confidence * 100:.1f}%)  ->  {result}")
                print(f"           {actuators.status_line()}")

                oled.show([
                    "TinyVoice",
                    f"CMD: {label.upper()}",
                    f"Conf: {confidence * 100:.0f}%",
                    result[:21],
                ])

                # Clear history so the same utterance, still sitting in the
                # buffer, cannot immediately trigger again.
                prob_history.clear()

            _sleep_remaining(loop_start)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        mic.stop()
        actuators.cleanup()

        print("\n--- SESSION SUMMARY ---")
        print(f"Windows processed : {window_count}")
        print(f"Commands executed : {detection_count}")
        if latencies:
            arr = np.array(latencies)
            print(f"Inference latency : mean {arr.mean():.2f} ms, "
                  f"median {np.median(arr):.2f} ms, "
                  f"p95 {np.percentile(arr, 95):.2f} ms")
        if mic.overflow_count:
            print(f"Audio overflows   : {mic.overflow_count} "
                  "(CPU is struggling - raise INFERENCE_HOP_S in config.py)")
        print("Goodbye.")


def _sleep_remaining(loop_start):
    """
    Sleep just long enough that the loop runs exactly every INFERENCE_HOP_S.

    Sleeping for a fixed 0.25 s would make the real period 0.25 s PLUS the
    processing time, so the loop would slowly drift. Subtracting the elapsed
    time keeps the cadence honest.
    """
    elapsed = time.perf_counter() - loop_start
    remaining = config.INFERENCE_HOP_S - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
