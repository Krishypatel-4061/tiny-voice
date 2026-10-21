"""
pipeline.py
===========
ONE class that processes ONE 1-second window through the whole recognition chain.

    window
      |-> Gate 0: RMS loudness   (free)              "is it even audible?"
      |-> Gate 1: VAD            (Option B, optional) "is it SPEECH?"
      |-> MFCC features
      |-> Denoiser               (Option A, optional) "clean the features"
      |-> DS-CNN classifier
      |-> smoothing -> confidence threshold -> refractory period -> command

WHY A SHARED CLASS?
Both the LIVE program (tinyvoice_run_v2.py) and the OFFLINE benchmark
(benchmark_cascade.py) call `Pipeline.process()`. So the benchmark measures
EXACTLY the logic that runs on the Pi - no "the benchmark does something
slightly different from the demo" doubts. Time is passed in as an argument
(`now`) so the benchmark can run a 2-minute recording in seconds of real time.
"""

import collections
import time

import numpy as np

import config
import features as feat
from vad_features import vad_features


class Pipeline:
    def __init__(self, engine, enhancer=None, vad=None, *,
                 vad_threshold=0.4,
                 confidence_threshold=config.CONFIDENCE_THRESHOLD,
                 rms_threshold=config.RMS_SILENCE_THRESHOLD,
                 smoothing=config.SMOOTHING_WINDOW,
                 refractory_s=config.REFRACTORY_S):
        self.engine = engine            # TinyVoiceEngine (the DS-CNN)
        self.enhancer = enhancer        # TFLiteRunner or None  (Option A)
        self.vad = vad                  # TFLiteRunner or None  (Option B)
        self.vad_threshold = vad_threshold
        self.confidence_threshold = confidence_threshold
        self.rms_threshold = rms_threshold
        self.smoothing = smoothing
        self.refractory_s = refractory_s
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Forget history and zero the statistics (call before each experiment)."""
        self.history = collections.deque(maxlen=self.smoothing)
        self.last_fire = -1e9
        self.counts = collections.Counter()
        self.seconds = collections.defaultdict(float)   # cumulative time per stage

    # ------------------------------------------------------------------
    def process(self, window, now):
        """
        Run one window through the chain.

        window : float32 array, 16000 samples
        now    : current time in seconds (real or simulated) - for the refractory period

        Returns a dict:
            stage      : "silent" | "vad_rejected" | "classified"
            command    : the command word if one FIRED this window, else None
            confidence : smoothed confidence of the top class (when classified)
            label      : top class name (when classified)
            vad_prob   : VAD probability (when VAD is enabled)
        """
        out = {"stage": None, "command": None, "confidence": 0.0,
               "label": None, "vad_prob": None}
        self.counts["windows"] += 1

        # ---- Gate 0: loudness (almost free) ---------------------------------
        if feat.rms(window) < self.rms_threshold:
            self.history.clear()      # stale evidence must not combine with new
            self.counts["silent"] += 1
            out["stage"] = "silent"
            return out

        # ---- Gate 1: voice-activity detector (Option B) ------------------------
        if self.vad is not None:
            t0 = time.perf_counter()
            p = float(np.ravel(self.vad.predict(vad_features(window)))[0])
            self.seconds["vad"] += time.perf_counter() - t0
            out["vad_prob"] = p
            if p < self.vad_threshold:
                self.history.clear()
                self.counts["vad_rejected"] += 1
                out["stage"] = "vad_rejected"
                return out

        # ---- Features -----------------------------------------------------------
        t0 = time.perf_counter()
        mfcc = feat.audio_to_mfcc(window)
        self.seconds["mfcc"] += time.perf_counter() - t0

        # ---- Denoiser (Option A) --------------------------------------------------
        if self.enhancer is not None:
            t0 = time.perf_counter()
            mfcc = self.enhancer.predict(mfcc)
            self.seconds["enhancer"] += time.perf_counter() - t0

        # ---- DS-CNN ----------------------------------------------------------------
        t0 = time.perf_counter()
        probs = self.engine.predict(mfcc)
        self.seconds["cnn"] += time.perf_counter() - t0
        self.counts["classified"] += 1
        out["stage"] = "classified"

        # ---- Decision logic (identical to the original tinyvoice_run.py) ------------
        self.history.append(probs)
        smoothed = np.mean(np.stack(self.history, axis=0), axis=0)
        label, confidence = self.engine.top_prediction(smoothed)
        out["label"], out["confidence"] = label, confidence
        out["top3"] = self.engine.top_k(smoothed, 3)

        is_command = label in config.COMMAND_WORDS
        confident = confidence >= self.confidence_threshold
        cooled_down = (now - self.last_fire) >= self.refractory_s
        enough_history = len(self.history) >= self.smoothing

        if is_command and confident and cooled_down and enough_history:
            self.last_fire = now
            self.history.clear()      # the same word still in the buffer must not re-fire
            self.counts["fired"] += 1
            out["command"] = label

        return out

    # ------------------------------------------------------------------
    def summary(self):
        """Human-readable statistics for the session / experiment."""
        c = self.counts
        w = max(c["windows"], 1)
        lines = [f"windows seen       : {c['windows']}",
                 f"  stopped by RMS   : {c['silent']}  ({100 * c['silent'] / w:.1f}%)"]
        if self.vad is not None:
            lines.append(f"  stopped by VAD   : {c['vad_rejected']}  ({100 * c['vad_rejected'] / w:.1f}%)")
        lines.append(f"  full classification: {c['classified']}  ({100 * c['classified'] / w:.1f}%)")
        lines.append(f"commands fired     : {c['fired']}")
        for stage in ("vad", "mfcc", "enhancer", "cnn"):
            n = c["classified"] if stage != "vad" else (c["classified"] + c["vad_rejected"])
            if self.seconds[stage] > 0 and n > 0:
                lines.append(f"avg {stage:8s} time : {1000 * self.seconds[stage] / n:.3f} ms/call")
        return "\n".join(lines)
