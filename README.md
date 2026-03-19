# genai-final-assignment

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
