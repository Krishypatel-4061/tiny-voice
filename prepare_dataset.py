"""
prepare_dataset.py
==================
Builds the final training / validation / test arrays from two data sources:

  A) dataset/custom/<word>/*.wav          <- your team's recordings (all 5 words)
  B) dataset/speech_commands/<word>/*.wav <- Google Speech Commands v2 (optional
                                             but strongly recommended)

From (B) we take three things:
  * more examples of "up", "down", "stop" (those words exist in the dataset)
  * ~30 other words ("bird", "seven", "left", ...) -> the _unknown_ class
  * the _background_noise_ folder -> chopped into 1 s chunks -> _silence_ class

WHAT THIS SCRIPT PRODUCES
-------------------------
  dataset/processed/X_train.npy , y_train.npy
  dataset/processed/X_val.npy   , y_val.npy
  dataset/processed/X_test.npy  , y_test.npy
  dataset/processed/meta.json

SPEAKER-INDEPENDENT SPLITTING (the part that impresses examiners)
-----------------------------------------------------------------
The naive approach is to shuffle all clips and split 80/10/10. That is WRONG for
speech: clips from the same person end up in both train and test, so the model
can cheat by memorising voices and your reported accuracy is inflated.

Instead we hold out ENTIRE PEOPLE. If your model gets 92% on a speaker it has
never heard, that number is real and it will survive a live demo.

USAGE
-----
    python prepare_dataset.py
    python prepare_dataset.py --no-augment       (faster, for a quick test run)
"""

import argparse
import json
import os
import random

import numpy as np
from scipy.io import wavfile

import config
import features as feat

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ===========================================================================
# 1. LOADING AUDIO
# ===========================================================================

def load_wav(path):
    """
    Read a WAV file and return float32 samples in [-1, 1] at 16 kHz.

    Both our recordings and the Google dataset are 16 kHz int16 mono, so this
    stays simple. If a file has a different rate we skip it rather than silently
    feeding wrong-speed audio into the model.
    """
    try:
        rate, data = wavfile.read(path)
    except Exception:
        return None

    if rate != config.SAMPLE_RATE:
        return None

    # Some files are stereo (shape (N, 2)); average the channels down to mono.
    if data.ndim > 1:
        data = data.mean(axis=1)

    # int16 -> float32 in [-1, 1]. 32768 is 2^15, the int16 range.
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)

    return data


def parse_speaker(filename, source):
    """
    Work out which human produced a clip.

    Our own files:     light__akanksha__0007.wav        -> "akanksha"
    Google's files:    0ab3b47d_nohash_0.wav            -> "0ab3b47d"
    """
    base = os.path.basename(filename)
    if source == "custom" and "__" in base:
        return base.split("__")[1]
    return base.split("_")[0]


# ===========================================================================
# 2. AUGMENTATION
# ===========================================================================
# Augmentation creates realistic variations of a clip. It is the single highest
# value thing you can do when your dataset is small, which yours is.

def augment_waveform(audio, noise_pool):
    """
    Return ONE randomly augmented copy of `audio`.

    Three independent effects, each applied with some probability:

    1. TIME SHIFT - slide the word left or right by up to 100 ms.
       Teaches the model that a command is still a command if it starts slightly
       late. Without this, the model latches onto "the word always starts at
       0.2 s" and fails badly on live streaming audio.

    2. BACKGROUND NOISE MIX - add real room noise at a random low level.
       This is what makes the demo survive a noisy classroom.

    3. GAIN CHANGE - make it quieter or louder.
       Teaches robustness to distance from the microphone.
    """
    audio = feat.fix_length(audio)

    # --- 1. time shift -----------------------------------------------------
    max_shift = int(0.1 * config.SAMPLE_RATE)          # 100 ms in samples
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        # np.roll wraps around; we zero the wrapped part so no audio teleports
        # from the end of the clip to the beginning.
        audio = np.roll(audio, shift)
        audio[:shift] = 0.0
    elif shift < 0:
        audio = np.roll(audio, shift)
        audio[shift:] = 0.0

    # --- 2. background noise ----------------------------------------------
    if noise_pool and random.random() < 0.8:
        noise = random.choice(noise_pool)
        # Pick a random 1 s slice out of the (long) noise recording.
        if len(noise) > config.CLIP_SAMPLES:
            start = np.random.randint(0, len(noise) - config.CLIP_SAMPLES)
            noise = noise[start:start + config.CLIP_SAMPLES]
        noise = feat.fix_length(noise)
        # Noise volume between 0 and 15% - enough to be realistic, not enough to
        # bury the command entirely.
        noise_gain = np.random.uniform(0.0, 0.15)
        audio = audio + noise_gain * noise

    # --- 3. random gain ----------------------------------------------------
    audio = audio * np.random.uniform(0.7, 1.3)

    # Always clip back into the valid range so we never simulate impossible audio.
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ===========================================================================
# 3. GATHERING FILE LISTS
# ===========================================================================

def collect_custom_files():
    """Returns a list of (filepath, label, speaker) from your own recordings."""
    items = []
    if not os.path.isdir(config.CUSTOM_AUDIO_DIR):
        return items

    for word in config.COMMAND_WORDS:
        word_dir = os.path.join(config.CUSTOM_AUDIO_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        for name in sorted(os.listdir(word_dir)):
            if name.endswith(".wav"):
                path = os.path.join(word_dir, name)
                items.append((path, word, parse_speaker(path, "custom")))
    return items


def collect_speech_commands_files():
    """
    Returns (command_items, unknown_items, noise_file_paths) from the Google set.

    Folder layout after unzipping speech_commands_v0.02.tar.gz:
        speech_commands/up/....wav
        speech_commands/bird/....wav
        speech_commands/_background_noise_/doing_the_dishes.wav
        ...
    """
    command_items, unknown_items, noise_paths = [], [], []

    if not os.path.isdir(config.SPEECH_COMMANDS_DIR):
        print("  (Google Speech Commands folder not found - skipping. The model "
              "will still train, but _unknown_ rejection will be much weaker.)")
        return command_items, unknown_items, noise_paths

    for entry in sorted(os.listdir(config.SPEECH_COMMANDS_DIR)):
        entry_path = os.path.join(config.SPEECH_COMMANDS_DIR, entry)
        if not os.path.isdir(entry_path):
            continue

        wavs = [os.path.join(entry_path, f)
                for f in sorted(os.listdir(entry_path)) if f.endswith(".wav")]

        if entry == "_background_noise_":
            noise_paths.extend(wavs)

        elif entry in config.COMMAND_WORDS:
            # "up", "down", "stop" overlap with our vocabulary - free extra data.
            random.shuffle(wavs)
            for p in wavs[:config.MAX_SC_CLIPS_PER_WORD]:
                command_items.append((p, entry, parse_speaker(p, "sc")))

        elif not entry.startswith("_"):
            # Every other word becomes an _unknown_ example.
            for p in wavs:
                unknown_items.append((p, config.UNKNOWN_LABEL,
                                      parse_speaker(p, "sc")))

    random.shuffle(unknown_items)
    unknown_items = unknown_items[:config.MAX_UNKNOWN_CLIPS]
    return command_items, unknown_items, noise_paths


def make_silence_clips(noise_pool, count):
    """
    Build the _silence_ class by cutting random 1 s slices out of the long
    background-noise recordings, at low volume.

    Note "silence" here means "no command was spoken", not "digital zero". If you
    train on true zeros the model has never heard a quiet room and will behave
    unpredictably the moment the projector fan turns on.
    """
    clips = []
    if not noise_pool:
        # Fall back to very quiet random noise so the class is not empty.
        for _ in range(count):
            clips.append((np.random.randn(config.CLIP_SAMPLES).astype(np.float32)
                          * 0.003))
        return clips

    for _ in range(count):
        noise = random.choice(noise_pool)
        if len(noise) > config.CLIP_SAMPLES:
            start = np.random.randint(0, len(noise) - config.CLIP_SAMPLES)
            chunk = noise[start:start + config.CLIP_SAMPLES].copy()
        else:
            chunk = feat.fix_length(noise)
        chunk = chunk * np.random.uniform(0.1, 1.0)
        clips.append(chunk.astype(np.float32))
    return clips


# ===========================================================================
# 4. MAIN PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation (much faster, lower accuracy)")
    args = parser.parse_args()

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    label_to_index = {label: i for i, label in enumerate(config.LABELS)}

    print("STEP 1: scanning audio files")
    custom_items = collect_custom_files()
    sc_commands, sc_unknown, noise_paths = collect_speech_commands_files()
    print(f"  custom recordings      : {len(custom_items)}")
    print(f"  speech-commands words  : {len(sc_commands)}")
    print(f"  speech-commands unknown: {len(sc_unknown)}")
    print(f"  background noise files : {len(noise_paths)}")

    if not custom_items and not sc_commands:
        raise SystemExit("\nNo recordings found.\n")

    print("\nSTEP 2: loading background noise into memory")
    noise_pool = [n for n in (load_wav(p) for p in noise_paths) if n is not None]

    # -----------------------------------------------------------------------
    # Speaker-independent split.
    # -----------------------------------------------------------------------
    print("\nSTEP 3: splitting by SPEAKER (not by clip)")
    custom_speakers = sorted({sp for _, _, sp in custom_items})
    print(f"  speakers in custom data: {custom_speakers}")

    if len(custom_speakers) >= 3:
        # Enough people to hold out whole speakers - the correct way.
        shuffled = custom_speakers[:]
        random.shuffle(shuffled)
        test_speakers = set(shuffled[:config.TEST_SPLIT_SPEAKERS])
        val_speakers = set(shuffled[config.TEST_SPLIT_SPEAKERS:
                                    config.TEST_SPLIT_SPEAKERS +
                                    config.VALIDATION_SPLIT_SPEAKERS])
        split_mode = "speaker-independent"
    else:
        # Only 1-2 people recorded. Fall back to a clip-level split and SAY SO,
        # because the resulting accuracy number is optimistic.
        test_speakers, val_speakers = set(), set()
        split_mode = "clip-level (fewer than 3 speakers - results optimistic)"

    print(f"  mode : {split_mode}")
    if test_speakers:
        print(f"  test speakers : {sorted(test_speakers)}")
        print(f"  val speakers  : {sorted(val_speakers)}")

    def assign_split(speaker, fallback_rand):
        """Decide which set a clip belongs to."""
        if test_speakers:
            if speaker in test_speakers:
                return "test"
            if speaker in val_speakers:
                return "val"
            return "train"
        # fallback: 80/10/10 by random draw
        if fallback_rand < 0.8:
            return "train"
        if fallback_rand < 0.9:
            return "val"
        return "test"

    # -----------------------------------------------------------------------
    # Feature extraction.
    # -----------------------------------------------------------------------
    print("\nSTEP 4: extracting MFCC features")
    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}

    def add(audio, label, split):
        X, y = buckets[split]
        X.append(feat.audio_to_mfcc(audio))
        y.append(label_to_index[label])

    all_file_items = custom_items + sc_commands + sc_unknown
    total = len(all_file_items)

    for i, (path, label, speaker) in enumerate(all_file_items):
        if i % 500 == 0:
            print(f"  {i}/{total} files...")

        audio = load_wav(path)
        if audio is None:
            continue

        split = assign_split(speaker, random.random())
        add(audio, label, split)

        # IMPORTANT: augment TRAINING data only.
        # Augmenting validation/test would mean you are measuring performance on
        # synthetic audio, which tells you nothing about the real demo.
        is_custom = True
        if split == "train" and not args.no_augment and is_custom:
            for _ in range(config.AUGMENTATIONS_PER_CLIP):
                add(augment_waveform(audio, noise_pool), label, split)

    # Silence class - split proportionally across the three sets.
    print("  building _silence_ class...")
    silence_clips = make_silence_clips(noise_pool, config.MAX_SILENCE_CLIPS)
    for i, clip in enumerate(silence_clips):
        split = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")
        add(clip, config.SILENCE_LABEL, split)

    # -----------------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------------
    print("\nSTEP 5: saving arrays")
    summary = {}
    for split, (X, y) in buckets.items():
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)

        # Shuffle so that batches during training are not all one class.
        order = np.random.permutation(len(y_arr))
        X_arr, y_arr = X_arr[order], y_arr[order]

        np.save(os.path.join(config.PROCESSED_DIR, f"X_{split}.npy"), X_arr)
        np.save(os.path.join(config.PROCESSED_DIR, f"y_{split}.npy"), y_arr)

        counts = {config.LABELS[c]: int((y_arr == c).sum())
                  for c in range(config.NUM_CLASSES)}
        summary[split] = {"total": int(len(y_arr)), "per_class": counts}
        print(f"  {split:5s}: {len(y_arr):6d} examples  shape={X_arr.shape}")
        for lbl, n in counts.items():
            print(f"          {lbl:12s} {n}")

    meta = {
        "labels": config.LABELS,
        "feature_shape": list(config.FEATURE_SHAPE),
        "sample_rate": config.SAMPLE_RATE,
        "split_mode": split_mode,
        "test_speakers": sorted(test_speakers),
        "val_speakers": sorted(val_speakers),
        "summary": summary,
    }
    with open(os.path.join(config.PROCESSED_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Next:  python train_model.py")


if __name__ == "__main__":
    main()
