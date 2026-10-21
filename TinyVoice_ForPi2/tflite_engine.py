"""
tflite_engine.py
================
A thin, portable wrapper around the LiteRT / TensorFlow Lite interpreter.

TWO PROBLEMS THIS FILE SOLVES
-----------------------------

1) THE IMPORT PROBLEM.
   The same interpreter ships under three different package names depending on
   what you managed to install:
        ai-edge-litert   ->  from ai_edge_litert.interpreter import Interpreter
        tflite-runtime   ->  from tflite_runtime.interpreter import Interpreter
        tensorflow       ->  tf.lite.Interpreter
   Your laptop probably has the third; the Pi will have one of the first two.
   The try/except chain below means the SAME code runs in both places.

2) THE QUANTIZATION PROBLEM.
   Our deployed model takes int8 input and gives int8 output, but our features
   are float32 and we want float probabilities. This class does the conversion
   using the scale/zero_point stored inside the .tflite file, so you never have
   to think about it again:

        int8   = round(float / scale) + zero_point       (input)
        float  = (int8 - zero_point) * scale             (output)
"""

import json
import os

import numpy as np

import config


def _load_interpreter_class():
    """Find whichever TFLite interpreter is installed and report which one."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite-runtime"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow"
    except ImportError:
        pass

    raise ImportError(
        "No TFLite interpreter found. Install ONE of:\n"
        "   pip install ai-edge-litert      (recommended on Raspberry Pi)\n"
        "   pip install tflite-runtime\n"
        "   pip install tensorflow          (heavy, fine on a laptop)"
    )


class TinyVoiceEngine:
    """
    Loads a .tflite model and runs inference on MFCC features.

    Example
    -------
        engine = TinyVoiceEngine()
        probs = engine.predict(mfcc_features)     # shape (7,), sums to 1.0
        label, conf = engine.top_prediction(probs)
    """

    def __init__(self, model_path=None, labels_path=None, num_threads=2):
        InterpreterClass, backend = _load_interpreter_class()
        self.backend = backend

        self.model_path = model_path or config.TFLITE_INT8_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Did you copy models/ across from your training machine?"
            )

        # num_threads=2 is a good default on a Pi 4 (4 cores): the model gets
        # two cores, leaving headroom for the audio callback and the GPIO logic.
        try:
            self.interpreter = InterpreterClass(model_path=self.model_path,
                                                num_threads=num_threads)
        except TypeError:
            # Some older builds do not accept num_threads.
            self.interpreter = InterpreterClass(model_path=self.model_path)

        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        # Quantization parameters. For a float32 model these come back as
        # (0.0, 0), which our code below treats as "no quantization needed".
        self.in_scale, self.in_zero = self.input_detail["quantization"]
        self.out_scale, self.out_zero = self.output_detail["quantization"]
        self.is_quantized = self.input_detail["dtype"] in (np.int8, np.uint8)

        labels_path = labels_path or config.LABELS_PATH
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.labels = json.load(f)["labels"]
        else:
            self.labels = config.LABELS

    # ----------------------------------------------------------------------
    def predict(self, features):
        """
        Run one inference.

        Parameters
        ----------
        features : numpy array of shape (49, 13, 1), float32, from features.py

        Returns
        -------
        numpy array of shape (num_classes,) with probabilities in [0, 1].
        """
        # Add the batch dimension the interpreter expects: (1, 49, 13, 1)
        x = np.expand_dims(features.astype(np.float32), axis=0)

        if self.is_quantized:
            # float -> int8 using the model's own scale and zero-point.
            x = x / self.in_scale + self.in_zero
            # np.round then clip to the valid int8 range before casting,
            # otherwise out-of-range values wrap around into nonsense.
            x = np.clip(np.round(x), -128, 127).astype(self.input_detail["dtype"])

        self.interpreter.set_tensor(self.input_detail["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_detail["index"])[0]

        if self.is_quantized:
            # int8 -> float probabilities.
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale

        return y.astype(np.float32)

    # ----------------------------------------------------------------------
    def top_prediction(self, probs):
        """Return (label, confidence) for the highest-scoring class."""
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])

    def top_k(self, probs, k=3):
        """Return the k best (label, confidence) pairs - handy for debugging."""
        order = np.argsort(probs)[::-1][:k]
        return [(self.labels[i], float(probs[i])) for i in order]

    def describe(self):
        """One-line summary printed at startup so you know what is loaded."""
        size_kb = os.path.getsize(self.model_path) / 1024.0
        kind = "int8 quantized" if self.is_quantized else "float32"
        return (f"{os.path.basename(self.model_path)} | {kind} | "
                f"{size_kb:.1f} KB | backend={self.backend} | "
                f"classes={len(self.labels)}")
