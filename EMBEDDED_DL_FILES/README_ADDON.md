# TinyVoice Add-On — Learned Denoiser (A) + VAD Cascade (B)

Two new deep-learning stages for your working system. **Nothing you already have
is modified** — every file below is NEW. Your original `tinyvoice_run.py` stays as
your safe fallback for the demo.

---

## 1. What you are adding (and the one-paragraph pitch)

```
BEFORE:  mic -> ring buffer -> RMS gate ---------------------> MFCC -> DS-CNN -> action

AFTER :  mic -> ring buffer -> RMS gate -> VAD (B) -> MFCC -> DENOISER (A) -> DS-CNN -> action
                                            |
                                  "not speech" -> stop here (saves CPU)
```

**Option A — Learned feature denoiser.** A ~11k-parameter *residual convolutional
network* that takes the noisy 49×13 MFCC "image" and outputs a cleaned one. It
learns `clean = noisy + correction`, so it is safe by design (a near-zero
correction leaves the features untouched). Trained on noisy/clean pairs we create
by mixing noise into clean recordings at known SNRs.

**Option B — Two-stage cascade with a learned VAD.** A ~1.8k-parameter CNN
(Voice-Activity Detector) looks at a cheap 40×8 spectral summary and decides "is
this speech?" *before* the expensive stages run. Unlike a loudness gate, it keys
on how the sound *changes over time* (speech does, steady noise does not), so it
can reject a loud fan or a clap that a loudness gate lets straight through.

**Pitch line:**
> "We extended our edge keyword spotter with two learned stages — a feature-domain
> denoiser for noise robustness and a tiny voice-activity detector that wakes the
> classifier only when speech is present — both quantized to int8 and benchmarked
> on the Raspberry Pi for accuracy, false triggers, CPU cost and latency."

Be honest about novelty: denoising front-ends and VAD cascades exist in the
literature. Your contribution is the **integrated, quantized, measured system on
real edge hardware** with an honest accuracy-vs-compute trade-off analysis.

---

## 2. Files in this add-on

| File | Runs on | Purpose |
|---|---|---|
| `config_ext.py` | both | All new settings (paths, SNR levels, VAD sizes) |
| `noise_utils.py` | both | Synthetic noise + real-noise loading + exact-SNR mixing |
| `record_noise.py` | PC/Pi | Record real room noise (most valuable 5 minutes) |
| `build_enhancer_data.py` | PC | **A1** — noisy/clean pairs + held-out test set + file splits |
| `train_enhancer.py` | PC | **A2** — train + convert the denoiser |
| `eval_noise_robustness.py` | PC | **A3** — accuracy vs SNR table + plot (report result) |
| `vad_features.py` | both | Cheap VAD features (shared train/runtime) |
| `train_vad.py` | PC | **B1** — train + convert the VAD, choose its threshold |
| `tflite_runner.py` | both | Generic float/int8 LiteRT runner for the two new models |
| `pipeline.py` | both | The whole recognition chain (shared by demo + benchmark) |
| `benchmark_cascade.py` | **Pi** | **B2** — 4-configuration CPU / recall / false-trigger table |
| `tinyvoice_run_v2.py` | **Pi** | Live demo with `--cascade` and `--enhance` flags |

**Installation:** copy ALL of these into your `EmbeddedProject` folder — the same
folder as `config.py`. **No new `pip install` is needed** (numpy, scipy,
tensorflow, matplotlib, sounddevice are already in your requirements).

---

## 3. THE THREE-DAY PLAN

### Day 1 — Data, denoiser, and the first real result

**0. (Do first, 5 min) Record real room noise** where you'll demo. One file per
situation, ~30–45 s each, **nobody speaking commands**:
```powershell
python record_noise.py --name fan_on --seconds 45
python record_noise.py --name people_talking --seconds 45
python record_noise.py --name motor_running --seconds 30
```
These land in `dataset\noise\` and are picked up automatically by everything below.

**1. Make sure the base model data is current.** If teammates' recordings have
arrived since you last ran it, re-run these so the DS-CNN and the new models agree
on who the "unseen" test speaker is:
```powershell
python prepare_dataset.py
python train_model.py
python convert_tflite.py
```

**2. Build the denoiser data:**
```powershell
python build_enhancer_data.py
```
Read the printout: it shows the split mode. With **< 3 speakers** it says
`clip-level ... results optimistic` — that is expected and fine for now.

**3. Train the denoiser** (a few minutes on CPU):
```powershell
python train_enhancer.py
```
It prints `Validation MSE before / after` and the % of feature error removed.
A clearly positive number means it is learning.

**4. Get your headline Option-A result:**
```powershell
python eval_noise_robustness.py --per-noise --per-noise-snr 0
```
Outputs `results\noise_robustness.csv` and `results\noise_robustness.png`.

### Day 2 — The VAD cascade

**5. Train the VAD:**
```powershell
python train_vad.py
```
Look for `speech caught (recall)` ≈ 98%+ and note the chosen threshold
(saved to `models\vad_meta.json`).

**6. Run the benchmark on your laptop first** (to confirm it works):
```powershell
python benchmark_cascade.py
```
Try a harsher room: `python benchmark_cascade.py --snr 5 --noise-level 0.05`

**7. Test the live program on your laptop, no hardware:**
```powershell
python tinyvoice_run_v2.py --no-gpio --cascade --enhance --debug
```

### Day 3 — Deploy to the Pi and collect the numbers that count

**8. Copy to the Pi.** The Pi needs (no TensorFlow required):
```
config_ext.py  noise_utils.py  vad_features.py  tflite_runner.py
pipeline.py    tinyvoice_run_v2.py   benchmark_cascade.py
models\enhancer_float32.tflite  enhancer_int8.tflite
models\vad_float32.tflite       vad_int8.tflite   vad_meta.json
dataset\enhancer\test_audio.npy  test_labels.npy      (for the benchmark)
```
(plus everything it already has from before). Use `scp` or a USB drive.

**9. On the Pi:**
```bash
source venv/bin/activate
python benchmark_cascade.py                       # CPU cost on the REAL hardware
python benchmark_cascade.py --vad-model models/vad_int8.tflite \
                            --enhancer-model models/enhancer_int8.tflite
python tinyvoice_run_v2.py --cascade --enhance --debug
```
**CPU cost is hardware-specific — the Pi numbers are the ones for your report.**

**10. Demo flags cheat-sheet:**

| Command | Meaning |
|---|---|
| `python tinyvoice_run.py` | your original system (safe fallback) |
| `python tinyvoice_run_v2.py` | identical behaviour, new code path |
| `... --cascade` | Option B only |
| `... --enhance` | Option A only |
| `... --cascade --enhance` | both |
| `... --enhance --enhancer-int8` | int8 denoiser |

---

## 4. WHAT TO PUT IN THE REPORT

| Result | File | What it shows |
|---|---|---|
| Accuracy vs SNR (baseline vs denoiser f32 vs int8) | `results/noise_robustness.csv/.png` | Option A's benefit |
| Per-noise-type table | `--per-noise` output | *Which* noises it helps with |
| 4-way configuration table | `results/benchmark_cascade.csv` | CPU, recall, false triggers (run on Pi) |
| Denoiser/VAD size + latency | printed by the scripts | edge cost of each stage |
| float32 vs int8 for the new models | both evals | quantization trade-off |
| VAD threshold + recall | `train_vad.py` output | safety-margin reasoning |

Run the benchmark at 2–3 difficulty settings (e.g. `--snr 15`, `--snr 10`, `--snr 5`)
and show the trend, not one number.

---

## 5. HONEST LIMITATIONS (say these before the examiner asks)

1. **Single-speaker data.** With < 3 speakers the test split is clip-level, so
   absolute accuracy is optimistic. The *comparison* (baseline vs denoiser) is
   still fair because both see the same clips. Add teammates' recordings to fix it.
2. **Synthetic noise is an approximation.** Real room recordings
   (`record_noise.py`) make the result far more credible. Re-run after adding them.
3. **The DS-CNN was not retrained on denoised features.** The denoiser is a drop-in
   front-end. *Future work:* fine-tune the DS-CNN on enhanced features (joint
   training) for further gain.
4. **Without Google Speech Commands, the `_unknown_` class is empty**, so the
   baseline false-trigger count is high. Download it (manual §4.2) and re-run
   `prepare_dataset.py` — all new scripts will use it automatically.
5. **The denoiser costs extra CPU** (~0.2–0.7 ms plus). The honest claim is a
   *trade-off*: slightly more compute for better noisy-room accuracy; the VAD
   cascade more than pays for it by skipping most windows.

---

## 6. LIKELY VIVA QUESTIONS

**Why denoise features rather than raw audio?** Far cheaper: the network works on a
49×13 grid (637 values) instead of 16 000 samples, so it adds fractions of a
millisecond on the Pi, and it plugs in between two stages that already exist.

**Why a residual connection?** The network only learns the correction; if unsure it
does almost nothing, so it can't badly damage clean input, and it trains well from
little data.

**Why not just raise the RMS threshold?** A loudness gate can't tell speech from a
loud fan or clap, and in a noisy room the noise floor is above any usable threshold
so the gate is permanently open. The VAD uses spectral *change over time*.

**Why is the VAD threshold so low?** A missed command is worse than a false
wake-up, because the DS-CNN rejects false wake-ups anyway. We choose the largest
threshold that still keeps ≥ 98% validation recall, capped at 0.5.

**How do you know the test is fair?** Test clips are held out of every model's
training (splits.json), noise realizations differ from training, and all systems are
scored on identical noisy audio.

**What did quantization cost?** Compare the f32 and int8 rows in the evaluation —
that is exactly why both are reported.

---

## 7. TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: config` | Run from the project root; files must sit beside `config.py` |
| `Run build_enhancer_data.py first` | You skipped step 2 — it also creates the test set and splits |
| `Not enough clips to make val/test sets` | Record more clips (≥ ~30 total per speaker is comfortable) |
| Denoiser gain `< 5%` | Add real noise recordings and more speakers; re-run steps 2–3 |
| VAD recall `< 95%` | Add real speech data (Speech Commands) / more recordings; or pass `--vad-threshold 0.3` |
| Benchmark recall low for ALL configs | Normal on tiny data: words are short vs. the 3-window smoothing. Compare configs against each other |
| Pi: `No TFLite interpreter` | `pip install ai-edge-litert` (or `tflite-runtime`) |
| v2 crashes on `OledDisplay` import | Make sure the original `tinyvoice_run.py` is in the same folder |
| Demo misbehaves | Fall back instantly: `python tinyvoice_run.py` |

---

## 8. DEMO-DAY CHECKLIST

- [ ] `python tinyvoice_run_v2.py --no-gpio --cascade --enhance --debug` works on the Pi
- [ ] Benchmark CSV generated **on the Pi**
- [ ] `noise_robustness.png` and the 4-way table ready as slides
- [ ] Real room noise recorded and the evals re-run with it
- [ ] Original `tinyvoice_run.py` tested as fallback
- [ ] You can explain, in one sentence each: MFCC, DS-CNN, residual denoiser, VAD, int8 quantization
