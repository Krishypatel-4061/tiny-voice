"""
tflite_runner.py
================
A small, GENERIC wrapper for running any .tflite model (float32 or int8).

Your existing tflite_engine.py is specialised for the keyword classifier (it
returns labels, top-k, etc.). The two new models - the DENOISER (Option A) and
the VAD (Option B) - just need "array in, array out", so they share this
simpler class. It reuses the interpreter-detection trick from tflite_engine.py,
so it runs unchanged on your laptop (TensorFlow) and on the Pi (LiteRT).

Quantization reminder (see convert_tflite.py): an int8 model stores numbers as
8-bit integers with a scale and zero-point:
        real_value = scale * (int8_value - zero_point)
This class converts float -> int8 on the way in and int8 -> float on the way out,
so the rest of the code only ever sees ordinary float32 arrays.
"""

import os

import numpy as np

from tflite_engine import _load_interpreter_class


class TFLiteRunner:
    """
    Example
    -------
        runner = TFLiteRunner("models/enhancer_float32.tflite")
        cleaned = runner.predict(mfcc)     # mfcc shape (49, 13, 1), no batch axis
    """

    def __init__(self, model_path, num_threads=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Did you train it and copy the models/ folder across?"
            )
        InterpreterClass, backend = _load_interpreter_class()
        self.backend = backend
        self.model_path = model_path

        # One thread by default: these models are tiny, and threading overhead
        # would cost more than it saves. The big DS-CNN keeps its own 2 threads.
        try:
            self.interpreter = InterpreterClass(model_path=model_path,
                                                num_threads=num_threads)
        except TypeError:
            self.interpreter = InterpreterClass(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]
        self.in_quantized = self.inp["dtype"] in (np.int8, np.uint8)
        self.out_quantized = self.out["dtype"] in (np.int8, np.uint8)

    def predict(self, x):
        """
        x : float32 array WITHOUT the batch axis (we add and remove it here).
        Returns a float32 array, also without the batch axis.
        """
        x = np.expand_dims(np.asarray(x, dtype=np.float32), axis=0)

        if self.in_quantized:
            info = np.iinfo(self.inp["dtype"])
            x = np.round(x / self.in_scale + self.in_zero)
            # Clip BEFORE casting, otherwise out-of-range values wrap around.
            x = np.clip(x, info.min, info.max).astype(self.inp["dtype"])

        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])[0]

        if self.out_quantized:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        return y.astype(np.float32)

    def describe(self):
        kind = "int8" if self.in_quantized else "float32"
        kb = os.path.getsize(self.model_path) / 1024.0
        return f"{os.path.basename(self.model_path)} | {kind} | {kb:.1f} KB | {self.backend}"
