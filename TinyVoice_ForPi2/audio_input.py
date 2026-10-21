"""
audio_input.py
==============
Continuous microphone capture using a RING BUFFER.

THE PROBLEM THIS SOLVES
-----------------------
A naive loop would be: record 1 second -> classify -> record 1 second -> ...
That fails, because while you are classifying you are deaf. If the user says
"stop" during that gap, you miss it entirely. Worse, a word that straddles the
boundary gets cut in half and is never recognised.

THE FIX: a ring buffer (also called a circular buffer).
The sound card thread continuously writes small blocks of audio into a fixed
1-second array, overwriting the oldest samples. The main thread can ask for
"the most recent 1 second" at any moment, as often as it likes, without ever
interrupting capture. This is the standard streaming keyword-spotting design.

    [older samples ......... newest samples]
     ^ write pointer wraps around to the start

A threading.Lock guards the buffer because two threads touch it: the audio
callback (owned by PortAudio) and your main loop.
"""

import threading

import numpy as np
import sounddevice as sd

import config


class MicrophoneStream:
    """
    Usage
    -----
        mic = MicrophoneStream()
        mic.start()
        window = mic.get_window()     # newest 1 s, float32, shape (16000,)
        ...
        mic.stop()

    Or as a context manager, which guarantees the device is released even if
    your code crashes:

        with MicrophoneStream() as mic:
            window = mic.get_window()
    """

    def __init__(self, sample_rate=config.SAMPLE_RATE,
                 window_samples=config.CLIP_SAMPLES,
                 device=None, block_ms=50):
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.device = device                       # None = system default input
        self.block_size = int(sample_rate * block_ms / 1000)

        # The ring buffer itself, pre-filled with silence.
        self._buffer = np.zeros(window_samples, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None

        # Counts how many times the sound card had to throw audio away because
        # we were too slow. Should stay at 0; if it climbs, your Pi is
        # overloaded (see the troubleshooting section of the manual).
        self.overflow_count = 0

    # ----------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        """
        Called automatically by PortAudio on its own thread, roughly every
        `block_ms` milliseconds. It must be FAST - no printing, no model
        inference, no file I/O. Just copy the data and return.
        """
        if status:
            # `status` is truthy on input overflow/underflow.
            self.overflow_count += 1

        # indata has shape (frames, channels); take channel 0 as mono.
        new_samples = indata[:, 0].astype(np.float32)

        with self._lock:
            n = len(new_samples)
            if n >= self.window_samples:
                # Block bigger than the whole window (only with huge block_ms):
                # just keep the newest part.
                self._buffer[:] = new_samples[-self.window_samples:]
            else:
                # Shift the existing contents left by n, then append the new
                # samples at the end. np.roll + overwrite is clear and fast
                # enough at these sizes.
                self._buffer[:-n] = self._buffer[n:]
                self._buffer[-n:] = new_samples

    # ----------------------------------------------------------------------
    def start(self):
        """Open the input stream and begin filling the buffer."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self):
        """Close the stream and release the microphone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_window(self):
        """Return a COPY of the most recent `window_samples` of audio."""
        with self._lock:
            # .copy() matters: without it the caller would be handed the live
            # buffer, which the audio thread is actively rewriting underneath it.
            return self._buffer.copy()

    # Context-manager support
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ----------------------------------------------------------------------
    @staticmethod
    def list_devices():
        """Print all audio devices. Run this first on the Pi to find your mic."""
        print(sd.query_devices())
        try:
            default_in = sd.query_devices(kind="input")
            print(f"\nDefault input: {default_in['name']}")
        except Exception as e:
            print(f"\nNo default input device found: {e}")
