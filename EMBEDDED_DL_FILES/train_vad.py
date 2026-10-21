"""
train_vad.py   (OPTION B, step 1)
=================================
Trains the tiny Voice-Activity Detector (VAD): a binary classifier that answers
"is someone speaking in this 1-second window?"  (1 = speech, 0 = no speech)

WHY A LEARNED VAD INSTEAD OF YOUR EXISTING LOUDNESS (RMS) GATE?
  * RMS only measures LOUDNESS. A running fan, a clap or a passing conversation
    is loud, so the RMS gate opens and the big network wastes CPU on it - and may
    even mis-fire.
  * In a noisy room the noise floor is ABOVE the RMS threshold all the time, so
    the gate is open permanently and saves nothing.
  * A learned VAD looks at the spectral/temporal PATTERN (see vad_features.py),
    so it can say "loud, but not speech".

THE CASCADE (how it fits the system)
    window -> RMS gate (free) -> VAD (tiny NN, cheap) -> [enhancer] -> MFCC -> DS-CNN
                                    |
                       "no speech" -> stop here, skip all the expensive work

THE MODEL (~1.5 k parameters)
    conv 3x3 (8) -> maxpool -> conv 3x3 (16) -> [global average ++ global max]
    -> dense 16 -> dense 1 (sigmoid = probability of speech)
  We concatenate a global AVERAGE and a global MAX because a word only occupies
  part of the 1 s window: the average captures overall texture, the max catches a
  short burst of speech.

THRESHOLD CHOICE
  A missed command is worse than a false wake-up (the DS-CNN will reject a false
  wake-up anyway). So after training we pick the LARGEST threshold that still
  keeps >= 98% recall on the validation set, and store it in models/vad_meta.json.

DATA (all built here, on the fly)
  positives (speech)     : your command clips (+ Speech Commands words if present),
                           randomly shifted, re-gained, and mixed with noise
  negatives (no speech)  : synthetic + real noise at many loudness levels,
                           INCLUDING loud ones and clap/click events

USAGE
-----
    python train_vad.py
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
import config_ext
import features as feat
import noise_utils as nu
from vad_features import vad_features

tf.random.set_seed(config.RANDOM_SEED)


# ===========================================================================
# 1. DATASET
# ===========================================================================
def shift_audio(audio, rng, max_s=0.1):
    """Slide the clip left/right by up to 100 ms, filling the gap with zeros."""
    m = int(max_s * config.SAMPLE_RATE)
    s = int(rng.integers(-m, m + 1))
    out = np.roll(audio, s)
    if s > 0:
        out[:s] = 0.0
    elif s < 0:
        out[s:] = 0.0
    return out


def build_vad_dataset(paths, copies, n_neg_ratio, rng, real_pool):
    X, y = [], []

    # ---- positives: speech -------------------------------------------------
    for path in paths:
        audio = nu.load_wav_mono(path)
        if audio is None:
            continue
        audio = feat.fix_length(audio)
        for _ in range(copies):
            a = shift_audio(audio, rng) * rng.uniform(0.3, 1.2)
            if rng.random() < 0.7:                      # 70% get noise, 30% stay clean
                _, noise = nu.random_noise(rng, real_pool, exclude=("clicks",))
                a = nu.mix_at_snr(a, noise, float(rng.choice([0, 5, 10, 15, 20])))
            X.append(vad_features(a))
            y.append(1)

    # ---- negatives: no speech -------------------------------------------------
    n_neg = int(len(y) * n_neg_ratio)
    for _ in range(n_neg):
        _, noise = nu.random_noise(rng, real_pool)       # clicks INCLUDED here
        level = 10 ** rng.uniform(-2.7, -0.5)            # RMS from ~0.002 up to ~0.3
        X.append(vad_features(noise * level))
        y.append(0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    order = rng.permutation(len(y))
    return X[order], y[order]


# ===========================================================================
# 2. MODEL
# ===========================================================================
def build_vad():
    inputs = keras.Input(shape=config_ext.VAD_FEATURE_SHAPE, name="vad_input")
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    avg = layers.GlobalAveragePooling2D()(x)
    mx = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([avg, mx])
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_speech")(x)
    return keras.Model(inputs, outputs, name="tinyvoice_vad")


# ===========================================================================
# 3. THRESHOLD + CONVERSION
# ===========================================================================
def choose_threshold(probs, labels, min_recall=0.98):
    """
    Largest threshold that still catches >= min_recall of real speech.

    We only search 0.05 ... 0.50. On easy data a higher threshold would also
    keep 98% recall, but it would leave NO safety margin once real-world audio
    (noisier, less separable than the training data) arrives. A missed command
    is worse than a false wake-up, so we deliberately stay at or below 0.5.
    """
    best = None
    for t in np.linspace(0.05, 0.50, 10):
        pred = probs >= t
        recall = float(pred[labels == 1].mean()) if (labels == 1).any() else 0.0
        if recall >= min_recall:
            best = float(t)                              # keep going: want the largest
    return best if best is not None else 0.30


def convert(model, rep_data):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(config_ext.VAD_FLOAT_PATH, "wb") as f:
        f.write(conv.convert())

    def rep_gen():
        idx = np.random.choice(len(rep_data), size=min(300, len(rep_data)), replace=False)
        for i in idx:
            yield [rep_data[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    with open(config_ext.VAD_INT8_PATH, "wb") as f:
        f.write(conv.convert())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config_ext.VAD_EPOCHS)
    parser.add_argument("--copies", type=int, default=12,
                        help="augmented positive examples per speech clip")
    parser.add_argument("--neg-ratio", type=float, default=1.3)
    args = parser.parse_args()

    if not os.path.exists(config_ext.SPLITS_PATH):
        raise SystemExit("Run build_enhancer_data.py first (it writes the file splits).")
    with open(config_ext.SPLITS_PATH) as f:
        splits = json.load(f)

    rng = np.random.default_rng(config.RANDOM_SEED + 7)
    real_pool = nu.load_real_noise_pool()
    print(f"Real noise recordings: {len(real_pool)}")

    print("Building VAD datasets (train files only - test files are never used)...")
    X_tr, y_tr = build_vad_dataset(splits["train"], args.copies, args.neg_ratio, rng, real_pool)
    X_va, y_va = build_vad_dataset(splits["val"], max(args.copies // 3, 2),
                                   args.neg_ratio, rng, real_pool)
    print(f"  train {X_tr.shape} (speech fraction {y_tr.mean():.2f})   val {X_va.shape}")

    model = build_vad()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(config_ext.VAD_KERAS_PATH, monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                          min_lr=1e-5, verbose=1),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs,
              batch_size=config_ext.VAD_BATCH_SIZE, callbacks=callbacks, verbose=2)
    model.save(config_ext.VAD_KERAS_PATH)

    # ---- evaluate + choose the threshold on validation data -------------------
    probs = model.predict(X_va, verbose=0).flatten()
    thr = choose_threshold(probs, y_va)
    pred = probs >= thr
    recall = float(pred[y_va == 1].mean())
    false_alarm = float(pred[y_va == 0].mean())
    print(f"\nChosen threshold: {thr:.2f}")
    print(f"  speech caught (recall)           : {100 * recall:.1f}%")
    print(f"  noise wrongly passed (false alarm): {100 * false_alarm:.1f}%"
          "   <- each of these only costs a little CPU; the DS-CNN rejects them")

    with open(config_ext.VAD_META_PATH, "w") as f:
        json.dump({"threshold": thr, "val_recall": recall,
                   "val_false_alarm": false_alarm}, f, indent=2)

    print("\nConverting to TFLite...")
    convert(model, X_tr)
    for p in (config_ext.VAD_KERAS_PATH, config_ext.VAD_FLOAT_PATH, config_ext.VAD_INT8_PATH):
        print(f"  {os.path.basename(p):22s} {os.path.getsize(p) / 1024:8.1f} KB")
    print("\nNext:  python benchmark_cascade.py")


if __name__ == "__main__":
    main()
