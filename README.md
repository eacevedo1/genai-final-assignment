# EnCodec Streaming Reconstruction - Generative AI Final Assignment

## Installation

Create and activate a Conda environment with Python 3.10:

```bash
conda create -n genai-final-assignment python=3.10 -y
conda activate genai-final-assignment
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchaudio`, `encodec`, `descript-audio-codec`, `soundfile`, `ruff`.

---

## Data

This project uses the [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset. Download the `fma_small` subset (~7.2 GB) with:

```bash
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip -d data/fma_small
```

---

## streaming.py

Command-line script that encodes an audio file with EnCodec in a streaming (chunk-by-chunk) fashion, reconstructs it via overlap-add synthesis, and compares the result against a full (non-streaming) reconstruction.

**Usage:**

```bash
python streaming.py <input_audio> [options]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `input_audio` | — | Path to the input audio file |
| `--codec` | `encodec48` | EnCodec model variant: `encodec48` or `encodec24` |
| `--bandwidth` | `12.0` | Target bandwidth in kbps |
| `--chunk-seconds` | `1.0` | Streaming window size in seconds |
| `--hop-seconds` | `0.5` | Step between consecutive windows in seconds |
| `--window` | `hann` | Synthesis window for overlap-add: `hann` or `rect` |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, or `mps` |

**Example:**

```bash
python streaming.py data/audio/000/track.mp3 --codec encodec48 --bandwidth 6.0 --chunk-seconds 1.0 --hop-seconds 0.5
```

The script saves two output files in the current directory:
- `<stem>_<codec>_bw<bw>_full.wav` — full (non-streaming) reconstruction
- `<stem>_<codec>_bw<bw>_stream.wav` — streaming overlap-add reconstruction

It also prints quality metrics (SI-SDR, SNR, L1, MSE) comparing both reconstructions against the original.

---

## src/ package

Helper modules used by `streaming.py`.

### `src/device.py`

Provides `get_device() -> str`, which selects the best available PyTorch accelerator in priority order: **CUDA → MPS → CPU**.

### `src/audio.py`

Audio I/O and waveform utilities:

### `src/eval.py`

Waveform quality metrics (all operate on raw waveform tensors):
### `src/overlap_add.py`

Streaming overlap-add reconstruction from pre-encoded EnCodec frames:

---

## smooth.py

Applies **EMA smoothing** to the per-chunk EnCodec scale factor during streaming overlap-add reconstruction, reducing audible amplitude jumps at chunk boundaries.

**Usage:**

```bash
python smooth.py <input_audio> [options]
```

| Argument | Default | Description |
|---|---|---|
| `input_audio` | — | Path to the input audio file |
| `--codec` | `encodec48` | `encodec48` or `encodec24` |
| `--bandwidth` | `6.0` | Target bandwidth in kbps |
| `--chunk-seconds` | `1.0` | Window size in seconds |
| `--hop-seconds` | `0.5` | Step between windows in seconds |
| `--window` | `hann` | Synthesis window: `hann` or `rect` |
| `--alpha` | `0.3` | EMA smoothing factor in (0, 1] |
| `--smooth-mode` | `linear` | `linear` (EMA on scale) or `log` (EMA on log-scale) |
| `--compare-alphas` | — | Extra alpha values for a comparison table |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

`alpha=1.0` disables smoothing; lower values smooth more aggressively (`0.3` recommended for music).

**Example:**

```bash
python smooth.py data/audio/000/000002.wav \
    --codec encodec48 --chunk-seconds 0.5 --hop-seconds 0.25 \
    --alpha 0.3 --compare-alphas 1.0 0.5 0.1
```

Prints SI-SDR / SNR / L1 / MSE for: full reconstruct, streaming OLA (no smoothing), linear EMA, and log EMA.

---

## learned_scale_predictor.py

Trains and evaluates a **small causal GRU** (`ScalePredictor`) that predicts the per-chunk EnCodec scale from encoder latents, enabling causal streaming without look-ahead. The GRU hidden state is carried across chunks to maintain loudness memory.

**Commands:**

```bash
python learned_scale_predictor.py eval <input_audio> <predictor.pt> [options]
python learned_scale_predictor.py train <audio_dir> [save_path] [options]
python learned_scale_predictor.py demo
```

**`eval` options:**

| Argument | Default | Description |
|---|---|---|
| `--codec` | `encodec48` | `encodec48` or `encodec24` |
| `--bandwidth` | `6.0` | Target bandwidth in kbps |
| `--chunk-seconds` | `1.0` | Chunk size in seconds |
| `--hop-seconds` | `0.5` | Hop size in seconds |
| `--window` | `hann` | Synthesis window: `hann` or `rect` |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

**`train` options:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `4` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--hidden-dim` | `64` | GRU hidden size |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

EnCodec is frozen; only the predictor is trained with MSE loss in log-scale space. `demo` runs a forward-pass smoke test with random tensors.

**Example:**

```bash
python learned_scale_predictor.py eval data/audio/000/000002.wav scale_predictor_000.pt \
    --codec encodec48 --chunk-seconds 0.5 --hop-seconds 0.25 --window hann --device cpu

python learned_scale_predictor.py train ./music_wavs/ predictor.pt --epochs 20
```

Prints SI-SDR / SNR / L1 / MSE for: full reconstruct, streaming OLA (raw scale), and predicted-scale OLA.
