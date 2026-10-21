"""
build_enhancer_data.py   (OPTION A, step 1)
===========================================
Builds the training data for the DENOISER.

THE IDEA
--------
A denoiser learns a mapping:   NOISY features  ->  CLEAN features.
We do not have recordings of the same word both "clean" and "noisy", but we do
not need them: we take a CLEAN recording and ADD noise ourselves at a known SNR.
Then we have a perfect pair:

        input  (X_noisy) = MFCC( clean_audio + noise )
        target (X_clean) = MFCC( clean_audio )

and we repeat with many different noise types / loudness levels per clip.
(This is called "synthetic noisy-clean pair generation" - standard practice.)

WHAT IT PRODUCES
----------------
  dataset/enhancer/X_noisy_{train,val}.npy   shape (N, 49, 13, 1)
  dataset/enhancer/X_clean_{train,val}.npy   shape (N, 49, 13, 1)
  dataset/enhancer/test_audio.npy            held-out CLEAN test clips (N, 16000)
  dataset/enhancer/test_labels.npy           their label indices (0-4)
  dataset/enhancer/splits.json               which file went to which split

THE TEST SET IS NEVER TOUCHED DURING TRAINING of either new model - the VAD
training script reads splits.json and also stays away from the test files.

SPLITTING (same philosophy as prepare_dataset.py)
-------------------------------------------------
  * If prepare_dataset.py already held out a TEST speaker (meta.json) and that
    speaker still exists, we reuse that same person, so the DS-CNN and the new
    models agree on who counts as "unseen".
  * Otherwise, with >= 3 speakers we hold out one test and one validation
    speaker; with fewer we split clip-by-clip and say so (results optimistic).

USAGE
-----
    python build_enhancer_data.py
    python build_enhancer_data.py --copies 4        (faster, fewer noisy copies)
"""

import argparse
import json
import os

import numpy as np

import config
import config_ext
import features as feat
import noise_utils as nu


def parse_speaker(filename):
    """light__akanksha__0007.wav -> 'akanksha'"""
    base = os.path.basename(filename)
    return base.split("__")[1] if "__" in base else "unknown"


def list_custom_clips():
    """All of YOUR recordings as (path, word, speaker)."""
    items = []
    for word in config.COMMAND_WORDS:
        word_dir = os.path.join(config.CUSTOM_AUDIO_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        for name in sorted(os.listdir(word_dir)):
            if name.endswith(".wav"):
                path = os.path.join(word_dir, name)
                items.append((path, word, parse_speaker(path)))
    return items


def list_speech_commands_clips(rng, cap):
    """Extra speech from Google's dataset (optional). Returns a list of paths."""
    if not os.path.isdir(config.SPEECH_COMMANDS_DIR):
        return []
    paths = []
    for entry in sorted(os.listdir(config.SPEECH_COMMANDS_DIR)):
        folder = os.path.join(config.SPEECH_COMMANDS_DIR, entry)
        if not os.path.isdir(folder) or entry.startswith("_"):
            continue
        paths += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.endswith(".wav")]
    rng.shuffle(paths)
    return paths[:cap]


def split_custom(items, rng):
    """Return (train_items, val_items, test_items, mode_string)."""
    speakers = sorted({sp for _, _, sp in items})

    # --- reuse the DS-CNN's held-out speakers when possible ----------------
    meta_path = os.path.join(config.PROCESSED_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        test_sp = set(meta.get("test_speakers", [])) & set(speakers)
        val_sp = set(meta.get("val_speakers", [])) & set(speakers)
        if test_sp:
            tr = [i for i in items if i[2] not in test_sp | val_sp]
            va = [i for i in items if i[2] in val_sp]
            te = [i for i in items if i[2] in test_sp]
            return tr, va, te, f"speaker-independent (reused from prepare_dataset: test={sorted(test_sp)})"

    # --- own speaker-independent split -------------------------------------
    if len(speakers) >= 3:
        order = list(rng.permutation(speakers))
        test_sp, val_sp = {order[0]}, {order[1]}
        tr = [i for i in items if i[2] not in test_sp | val_sp]
        va = [i for i in items if i[2] in val_sp]
        te = [i for i in items if i[2] in test_sp]
        return tr, va, te, f"speaker-independent (test={sorted(test_sp)}, val={sorted(val_sp)})"

    # --- not enough people: clip-level split (be honest about it) ----------
    idx = rng.permutation(len(items))
    n_te = max(1, int(0.15 * len(items)))
    n_va = max(1, int(0.15 * len(items)))
    te = [items[i] for i in idx[:n_te]]
    va = [items[i] for i in idx[n_te:n_te + n_va]]
    tr = [items[i] for i in idx[n_te + n_va:]]
    return tr, va, te, "clip-level (fewer than 3 speakers - results optimistic)"


def make_pairs(paths, copies, rng, real_pool):
    """
    For every clean clip make `copies` (noisy, clean) MFCC pairs.
    Returns two arrays of shape (N, 49, 13, 1).
    """
    noisy_list, clean_list = [], []
    for i, path in enumerate(paths):
        audio = nu.load_wav_mono(path)
        if audio is None:
            continue
        audio = feat.fix_length(audio)
        clean_feat = feat.audio_to_mfcc(audio)            # the TARGET (computed once)

        for _ in range(copies):
            _, noise = nu.random_noise(rng, real_pool)
            snr = float(rng.choice(config_ext.TRAIN_SNRS_DB))
            noisy_audio = nu.mix_at_snr(audio, noise, snr)
            noisy_list.append(feat.audio_to_mfcc(noisy_audio))   # the INPUT
            clean_list.append(clean_feat)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(paths)} clips...")

    return (np.asarray(noisy_list, dtype=np.float32),
            np.asarray(clean_list, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=config_ext.ENH_COPIES_TRAIN,
                        help="noisy copies per clean TRAIN clip")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(config_ext.ENH_DIR, exist_ok=True)

    print("STEP 1: finding clean speech")
    items = list_custom_clips()
    if not items:
        raise SystemExit("No recordings in dataset/custom/. Record first "
                         "(record_dataset.py), then run prepare_dataset.py.")
    train_i, val_i, test_i, mode = split_custom(items, rng)
    print(f"  custom clips: {len(items)}   split mode: {mode}")

    sc_paths = list_speech_commands_clips(rng, config_ext.ENH_MAX_SC_CLIPS)
    n_val_sc = len(sc_paths) // 10                      # 10% of extras -> validation
    sc_val, sc_train = sc_paths[:n_val_sc], sc_paths[n_val_sc:]
    print(f"  extra Speech Commands clips: {len(sc_paths)} "
          f"({'used' if sc_paths else 'folder not found - skipped, fine'})")

    train_paths = [p for p, _, _ in train_i] + sc_train
    val_paths = [p for p, _, _ in val_i] + sc_val
    test_paths = [p for p, _, _ in test_i]
    print(f"  train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")
    if len(test_paths) == 0 or len(val_paths) == 0:
        raise SystemExit("Not enough clips to make val/test sets. Record more.")

    print("\nSTEP 2: loading real noise recordings (if any)")
    real_pool = nu.load_real_noise_pool()
    print(f"  real noise files: {len(real_pool)} "
          f"({'plus synthetic' if real_pool else 'synthetic noise only'})")

    print("\nSTEP 3: building noisy/clean MFCC pairs")
    print(f"  train ({args.copies} noisy copies per clip)")
    Xn_tr, Xc_tr = make_pairs(train_paths, args.copies, rng, real_pool)
    print(f"  val   ({config_ext.ENH_COPIES_VAL} noisy copies per clip)")
    Xn_va, Xc_va = make_pairs(val_paths, config_ext.ENH_COPIES_VAL, rng, real_pool)

    print("\nSTEP 4: saving")
    np.save(os.path.join(config_ext.ENH_DIR, "X_noisy_train.npy"), Xn_tr)
    np.save(os.path.join(config_ext.ENH_DIR, "X_clean_train.npy"), Xc_tr)
    np.save(os.path.join(config_ext.ENH_DIR, "X_noisy_val.npy"), Xn_va)
    np.save(os.path.join(config_ext.ENH_DIR, "X_clean_val.npy"), Xc_va)

    # Held-out CLEAN test audio + labels (noise is added later, at any SNR we like).
    test_audio, test_labels = [], []
    for path, word, _ in test_i:
        audio = nu.load_wav_mono(path)
        if audio is not None:
            test_audio.append(feat.fix_length(audio))
            test_labels.append(config.LABELS.index(word))
    np.save(os.path.join(config_ext.ENH_DIR, "test_audio.npy"),
            np.asarray(test_audio, dtype=np.float32))
    np.save(os.path.join(config_ext.ENH_DIR, "test_labels.npy"),
            np.asarray(test_labels, dtype=np.int32))

    with open(config_ext.SPLITS_PATH, "w") as f:
        json.dump({"mode": mode, "train": train_paths, "val": val_paths,
                   "test": test_paths}, f, indent=2)

    print(f"  X_noisy_train {Xn_tr.shape}   X_noisy_val {Xn_va.shape}")
    print(f"  test clips kept aside: {len(test_audio)}")
    print("\nDone. Next:  python train_enhancer.py")


if __name__ == "__main__":
    main()
