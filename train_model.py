"""
train_model.py
==============
Trains the keyword-spotting neural network.

THE ARCHITECTURE: DS-CNN (Depthwise Separable Convolutional Neural Network)
--------------------------------------------------------------------------
This is the reference architecture for keyword spotting on constrained hardware,
introduced in "Hello Edge: Keyword Spotting on Microcontrollers" (Zhang et al.,
ARM, 2017). It is the honest, well-cited choice for this project - not a toy
model, but not a 100 MB transformer either.

Why depthwise-separable convolution?
  A normal Conv2D with C_in input channels, C_out output channels and a KxK
  kernel costs  K*K*C_in*C_out  multiplications per output pixel.
  A depthwise-separable convolution splits that into two cheaper steps:
     1. DEPTHWISE  - one KxK filter per input channel (spatial mixing only)
     2. POINTWISE  - a 1x1 convolution across channels (channel mixing only)
  Cost becomes  K*K*C_in + C_in*C_out, which is roughly 8-9x cheaper for a 3x3
  kernel with 64 channels, for almost the same accuracy.

  That is exactly the trade you want on a Raspberry Pi: near-CNN accuracy at a
  fraction of the compute, so inference stays in the low milliseconds and leaves
  the CPU free for audio capture and GPIO.

Result: about 25,000 parameters, ~30 KB as an int8 .tflite file.

USAGE
-----
    python train_model.py
    python train_model.py --epochs 80
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ===========================================================================
# 1. SPEC-AUGMENT LAYER
# ===========================================================================
class SpecAugment(layers.Layer):
    """
    Randomly blanks out a horizontal band (time) and a vertical band (frequency)
    of the MFCC "image" during training only.

    Intuition: it is dropout for spectrograms. By hiding a slice of the input,
    the model is forced to recognise "light" from several parts of the word
    rather than fixating on one instant. Introduced by Google for speech
    recognition (Park et al., 2019) and it consistently helps small datasets.

    Because it is a Keras layer with `training=` handling, it automatically
    switches OFF during validation, testing and deployment. You never have to
    remember to disable it.
    """

    def __init__(self, time_mask=6, freq_mask=3, **kwargs):
        super().__init__(**kwargs)
        self.time_mask = time_mask   # max frames to blank (out of 49)
        self.freq_mask = freq_mask   # max cepstral coefficients to blank (of 13)

    def call(self, inputs, training=None):
        if not training:
            return inputs          # identity at inference time

        shape = tf.shape(inputs)
        batch, n_time, n_freq = shape[0], shape[1], shape[2]

        def make_mask(length, max_width, axis_len):
            """Build a 0/1 mask that zeroes a random contiguous band."""
            width = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
            start = tf.random.uniform([], 0, tf.maximum(axis_len - width, 1),
                                      dtype=tf.int32)
            positions = tf.range(length)
            inside = tf.logical_and(positions >= start, positions < start + width)
            return tf.cast(tf.logical_not(inside), inputs.dtype)

        # Time mask: shape (n_time,) -> broadcast over freq and channel
        t_mask = make_mask(n_time, self.time_mask, n_time)
        t_mask = tf.reshape(t_mask, [1, -1, 1, 1])

        # Frequency mask: shape (n_freq,) -> broadcast over time and channel
        f_mask = make_mask(n_freq, self.freq_mask, n_freq)
        f_mask = tf.reshape(f_mask, [1, 1, -1, 1])

        _ = batch  # (kept for readability; broadcasting handles the batch dim)
        return inputs * t_mask * f_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"time_mask": self.time_mask, "freq_mask": self.freq_mask})
        return cfg


# ===========================================================================
# 2. MODEL DEFINITION
# ===========================================================================
def ds_cnn_block(x, filters, name):
    """
    One depthwise-separable convolution block:
        depthwise conv -> BN -> ReLU -> pointwise conv -> BN -> ReLU

    BatchNormalization (BN) re-centres activations each batch. It makes training
    noticeably more stable and lets us use a higher learning rate. It also folds
    into the convolution weights during TFLite conversion, so it costs nothing
    at inference time.
    """
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False,
                               name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)

    x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False,
                      name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_model(input_shape, num_classes):
    """Assemble the full DS-CNN."""
    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    # Augmentation first, so every layer after it sees the masked version.
    x = SpecAugment(name="specaugment")(inputs)

    # ---- Stem: one ordinary convolution ---------------------------------
    # A wide kernel (10 frames tall, 4 coefficients wide) lets the first layer
    # see a meaningful chunk of a phoneme straight away. Stride (2,2) halves the
    # feature map immediately, which is where most of the compute saving comes
    # from.
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same",
                      use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.Dropout(0.2, name="stem_drop")(x)

    # ---- Four depthwise-separable blocks ---------------------------------
    for i in range(4):
        x = ds_cnn_block(x, 64, name=f"ds{i + 1}")

    # ---- Head -------------------------------------------------------------
    # GlobalAveragePooling collapses the whole feature map to 64 numbers by
    # averaging. Compared to Flatten + Dense this removes ~90% of the parameters
    # and strongly reduces overfitting, which matters because our dataset is small.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="tinyvoice_dscnn")


# ===========================================================================
# 3. TRAINING
# ===========================================================================
def load_split(split):
    X = np.load(os.path.join(config.PROCESSED_DIR, f"X_{split}.npy"))
    y = np.load(os.path.join(config.PROCESSED_DIR, f"y_{split}.npy"))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("Loading processed data...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"  train: {X_train.shape}   val: {X_val.shape}")

    # -----------------------------------------------------------------------
    # CLASS WEIGHTS
    # -----------------------------------------------------------------------
    # _unknown_ will have far more examples than "fan". Without correction the
    # model learns "predicting _unknown_ is usually right" and ignores the rare
    # commands. Class weights make each class contribute equally to the loss.
    counts = np.bincount(y_train, minlength=config.NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0                       # avoid divide-by-zero
    class_weight = {i: float(len(y_train) / (config.NUM_CLASSES * counts[i]))
                    for i in range(config.NUM_CLASSES)}
    print("\nClass weights (higher = rarer class, gets more attention):")
    for i, label in enumerate(config.LABELS):
        print(f"  {label:12s} n={int(counts[i]):5d}  weight={class_weight[i]:.2f}")

    # -----------------------------------------------------------------------
    print("\nBuilding model...")
    model = build_model(config.FEATURE_SHAPE, config.NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        # sparse_* because our labels are integers (3), not one-hot ([0,0,0,1,...])
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        # Save the weights from the epoch with the BEST validation accuracy,
        # not the last epoch (which is often slightly overfitted).
        keras.callbacks.ModelCheckpoint(
            config.KERAS_MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        # Drop the learning rate when progress stalls - lets the model settle
        # into a better minimum instead of bouncing around it.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=6,
            min_lr=1e-5, verbose=1),
        # Stop early if nothing improves for 15 epochs, and restore the best weights.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # -----------------------------------------------------------------------
    print("\nSaving artefacts...")
    model.save(config.KERAS_MODEL_PATH)
    with open(config.LABELS_PATH, "w") as f:
        json.dump({"labels": config.LABELS}, f, indent=2)

    # Save the training curves as JSON so you can plot them in your report.
    hist_path = os.path.join(config.MODELS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    best_val = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val * 100:.2f}%")
    print(f"Model : {config.KERAS_MODEL_PATH}")
    print(f"Labels: {config.LABELS_PATH}")
    print("\nNext:  python convert_tflite.py")


if __name__ == "__main__":
    main()
