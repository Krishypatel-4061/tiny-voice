"""
train_enhancer.py   (OPTION A, step 2)
======================================
Trains the DENOISER and converts it for the Raspberry Pi.

THE MODEL: a small RESIDUAL convolutional denoiser
--------------------------------------------------
Input  : the noisy MFCC "image"   (49 time frames x 13 coefficients x 1)
Output : a cleaned MFCC "image" of the SAME shape

        noisy MFCC ---> [conv] -> [conv] -> [conv] -> [conv] ---> correction
             |                                                        |
             +---------------------------(+)<-------------------------+
                                          |
                                    cleaned MFCC

WHY "RESIDUAL" (the skip connection + Add)?
  The network does not have to rebuild the whole image from scratch. It only
  learns the CORRECTION to add to the noisy input:   clean = noisy + correction.
  This is far easier to learn from little data, and it is SAFE: if the network
  is unsure it outputs a correction near zero and the features pass through
  almost unchanged - it cannot easily make things dramatically worse.
  (Same idea as the skip connections in ResNet.)

WHY CONVOLUTIONS?
  Noise corrupts neighbouring time frames and neighbouring coefficients in
  similar ways. A 2-D convolution looks at a small patch of both at once, so it
  can learn "this smear across 5 frames is noise, that sharp pattern is speech".

LOSS: Mean Squared Error between cleaned output and the true clean MFCC.

~11,000 parameters -> float32 .tflite ~ 50 KB, int8 ~ 20 KB. A fraction of a
millisecond to a couple of ms on the Pi.

USAGE
-----
    python train_enhancer.py
    python train_enhancer.py --epochs 20
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
import config_ext

tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def build_enhancer(input_shape=config.FEATURE_SHAPE):
    """Residual conv denoiser. Input and output shape are identical."""
    inputs = keras.Input(shape=input_shape, name="noisy_mfcc")

    # First layer is wider in TIME (5 frames = 100 ms of context) because noise
    # and speech differ mostly in how they evolve over time.
    x = layers.Conv2D(24, (5, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.Conv2D(24, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.Conv2D(24, (3, 3), padding="same", activation="relu", name="conv3")(x)

    # Last conv has NO activation: the correction must be able to be negative
    # as well as positive (ReLU would force it to be >= 0).
    correction = layers.Conv2D(1, (3, 3), padding="same", name="correction")(x)

    # The skip connection: output = input + learned correction.
    outputs = layers.Add(name="residual_add")([inputs, correction])
    return keras.Model(inputs, outputs, name="tinyvoice_enhancer")


def load(split):
    n = np.load(os.path.join(config_ext.ENH_DIR, f"X_noisy_{split}.npy"))
    c = np.load(os.path.join(config_ext.ENH_DIR, f"X_clean_{split}.npy"))
    return n, c


def convert(model, rep_data):
    """Write float32 and int8 .tflite files (same recipe as convert_tflite.py)."""
    # ---- float32 --------------------------------------------------------
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(config_ext.ENHANCER_FLOAT_PATH, "wb") as f:
        f.write(conv.convert())

    # ---- int8 (needs real data to measure activation ranges) --------------
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
    with open(config_ext.ENHANCER_INT8_PATH, "wb") as f:
        f.write(conv.convert())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config_ext.ENH_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config_ext.ENH_BATCH_SIZE)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(config_ext.ENH_DIR, "X_noisy_train.npy")):
        raise SystemExit("Run build_enhancer_data.py first.")
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    Xn_tr, Xc_tr = load("train")
    Xn_va, Xc_va = load("val")
    print(f"train pairs: {Xn_tr.shape}   val pairs: {Xn_va.shape}")

    # The number to beat: how far is the NOISY input from the CLEAN target if we
    # do nothing at all? A useful denoiser must get below this.
    mse_before = float(np.mean((Xn_va - Xc_va) ** 2))
    print(f"\nValidation MSE with NO denoising (noisy vs clean): {mse_before:.4f}")

    model = build_enhancer()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    callbacks = [
        keras.callbacks.ModelCheckpoint(config_ext.ENHANCER_KERAS_PATH,
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                      restore_best_weights=True, verbose=1),
    ]

    print("\nTraining the denoiser...")
    history = model.fit(Xn_tr, Xc_tr, validation_data=(Xn_va, Xc_va),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, verbose=2)

    model.save(config_ext.ENHANCER_KERAS_PATH)

    mse_after = float(np.mean((model.predict(Xn_va, verbose=0) - Xc_va) ** 2))
    gain = 100.0 * (1.0 - mse_after / mse_before)
    print(f"\nValidation MSE  before: {mse_before:.4f}   after: {mse_after:.4f}")
    print(f"=> the denoiser removes {gain:.1f}% of the feature error")
    if gain < 5:
        print("   (small gain: add more/real noise recordings or more speakers)")

    print("\nConverting to TFLite (float32 + int8)...")
    convert(model, Xn_tr)
    for path in (config_ext.ENHANCER_KERAS_PATH, config_ext.ENHANCER_FLOAT_PATH,
                 config_ext.ENHANCER_INT8_PATH):
        print(f"  {os.path.basename(path):26s} {os.path.getsize(path) / 1024:8.1f} KB")

    print("\nNext:  python eval_noise_robustness.py")


if __name__ == "__main__":
    main()
