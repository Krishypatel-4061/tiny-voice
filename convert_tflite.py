"""
convert_tflite.py
=================
Converts the trained Keras model into the two deployable formats.

WHY CONVERT AT ALL?
-------------------
Keras/TensorFlow is a training framework - it is huge and assumes a desktop.
For deployment we use LiteRT (formerly TensorFlow Lite), a small interpreter
that loads a single flat `.tflite` file. On a Raspberry Pi this means a ~2 MB
runtime instead of a ~500 MB framework, and noticeably faster inference.

WHAT IS QUANTIZATION?
---------------------
The trained model stores every weight as a 32-bit float. Full-integer
quantization rewrites them as 8-bit integers by learning, for each tensor, a
scale and zero-point such that:

        real_value  ~=  scale * (int8_value - zero_point)

Effects:
  * model file becomes ~4x smaller
  * inference gets faster (integer maths, better cache use)
  * accuracy typically drops by well under 1% for a model this size

To pick good scales the converter must SEE real data flowing through the model.
That is what the "representative dataset" below is for - a few hundred genuine
training samples so the converter learns the true range of every activation.
Using random noise here is a classic mistake that silently wrecks accuracy.

USAGE
-----
    python convert_tflite.py
"""

import os

import numpy as np
import tensorflow as tf

import config


def representative_dataset_gen(X, num_samples=300):
    """
    A generator that yields single input samples to calibrate quantization.

    TFLite calls this repeatedly. Each yielded item must be a LIST of inputs
    (our model has one input) with a batch dimension of 1.
    """
    idx = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
    for i in idx:
        sample = X[i:i + 1].astype(np.float32)   # shape (1, 49, 13, 1)
        yield [sample]


def file_size_kb(path):
    return os.path.getsize(path) / 1024.0


def main():
    print("Loading Keras model...")
    # SpecAugment is a custom layer, so Keras needs to be told how to rebuild it.
    from train_model import SpecAugment
    model = tf.keras.models.load_model(
        config.KERAS_MODEL_PATH,
        custom_objects={"SpecAugment": SpecAugment},
    )

    X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))

    # -----------------------------------------------------------------------
    # A) FLOAT32 .tflite - the safe baseline
    # -----------------------------------------------------------------------
    print("\n[1/2] Converting to float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    with open(config.TFLITE_FLOAT_PATH, "wb") as f:
        f.write(tflite_float)
    print(f"  saved {config.TFLITE_FLOAT_PATH}  "
          f"({file_size_kb(config.TFLITE_FLOAT_PATH):.1f} KB)")

    # -----------------------------------------------------------------------
    # B) INT8 .tflite - the deployment model
    # -----------------------------------------------------------------------
    print("\n[2/2] Converting to full-integer (int8) TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # DEFAULT optimisations = weight quantization + operator fusion.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Feed real data so activation ranges are measured, not guessed.
    converter.representative_dataset = lambda: representative_dataset_gen(X_train)

    # Force EVERY operation to be integer. If any op cannot be quantized the
    # conversion fails loudly - which is what we want, rather than silently
    # producing a mixed model that behaves differently on the Pi.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    with open(config.TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_int8)
    print(f"  saved {config.TFLITE_INT8_PATH}  "
          f"({file_size_kb(config.TFLITE_INT8_PATH):.1f} KB)")

    # -----------------------------------------------------------------------
    # Print the quantization parameters. You will need these numbers to
    # explain your pipeline in the report, and the runtime uses them to convert
    # float MFCCs into int8 before inference.
    # -----------------------------------------------------------------------
    interpreter = tf.lite.Interpreter(model_content=tflite_int8)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print("\nQuantization parameters (record these for your report):")
    print(f"  input  dtype={inp['dtype'].__name__}  shape={inp['shape']}")
    print(f"         scale={inp['quantization'][0]:.8f}  "
          f"zero_point={inp['quantization'][1]}")
    print(f"  output dtype={out['dtype'].__name__}  shape={out['shape']}")
    print(f"         scale={out['quantization'][0]:.8f}  "
          f"zero_point={out['quantization'][1]}")

    size_ratio = (file_size_kb(config.TFLITE_FLOAT_PATH) /
                  file_size_kb(config.TFLITE_INT8_PATH))
    print(f"\nCompression vs float32: {size_ratio:.2f}x smaller")
    print("\nNext:  python evaluate_model.py   (verify the int8 model is still accurate)")


if __name__ == "__main__":
    main()
