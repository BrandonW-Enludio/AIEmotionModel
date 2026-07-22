# AI Emotion Model — Local Voice Pipeline

Real-time, local voice NPC that listens to you, detects emotion from speech, and answers with spoken dialogue. Everything can run on your machine (no cloud API required).

**Pipeline:** mic → VAD → speech-to-text → speech emotion recognition → LLM → text-to-speech

Default models:

| Stage | Model |
|-------|--------|
| STT | faster-whisper `tiny` |
| SER | `superb/wav2vec2-base-superb-er` |
| LLM | OpenAI-compatible client → **llama.cpp** serving a **GGUF** (e.g. Qwen2.5-7B-Instruct) |
| TTS | Chatterbox Turbo |

The LLM runs in a **separate** `llama-server` process. The Python app calls it over an OpenAI-style HTTP API (`llm_openai.py`). You can swap back to in-process Hugging Face Transformers via `pipeline_config.py` (`"llm": "qwen2_5_7b"`).

---

## Requirements

- **Windows** (setup steps below are for PowerShell)
- **Python 3.11** (e.g. 3.11.9)
- **NVIDIA GPU** with a recent driver (tested with RTX 4070 / 4080). About **12–16 GB VRAM** is a comfortable fit when llama.cpp and the voice app share one GPU.
- Working **microphone** and speakers/headphones
- Disk space and internet for model downloads (Hugging Face / GGUF / llama.cpp binaries)

`nvidia-smi` may show CUDA 12.x or 13.x — that number is your **driver’s max supported CUDA**. This project installs **PyTorch CUDA 12.4** wheels, which is fine on newer drivers.

---

## Setup

### 1. Create and activate a virtual environment

From the project root:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt should show `(venv)`. Always activate this venv before installing or running.

If activation is blocked by PowerShell policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install dependencies

With the venv **activated**:

```powershell
python setup.py
```

This upgrades pip, then installs (in order):

1. PyTorch / torchvision / torchaudio (CUDA 12.4)
2. `chatterbox-tts`
3. `sounddevice`
4. `numpy`, `silero-vad`, `faster-whisper`, `transformers`, `accelerate`, `bitsandbytes`

There is no `requirements.txt` — use `setup.py` only.

### 3. Check the GPU

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

You want `True` and your GPU name. If CUDA is `False`, you likely installed CPU torch or the driver/GPU is unavailable — re-run `setup.py` inside the activated venv, or fix the NVIDIA driver.

### 4. Install llama.cpp (LLM server)

You do **not** need to build from source. Download a **prebuilt Windows** release from:

[https://github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)

- Prefer a **CUDA** build if you have an NVIDIA GPU (filename like `llama-*-bin-win-cuda-*-x64.zip`).
- Unzip somewhere convenient (e.g. `C:\tools\llama.cpp\`).
- You will use `llama-server.exe` from that folder.

### 5. Download a GGUF model

llama.cpp loads **GGUF** files (not the Hugging Face safetensors used by in-process Transformers).

Example — Qwen2.5-7B-Instruct Q5 (good baseline):

```powershell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF `
  qwen2.5-7b-instruct-q5_k_m.gguf `
  --local-dir K:\AI\models
```

Notes:

- Exact filenames vary by repo; check the Hugging Face file list and pick a single **`Q4_K_M`** or **`Q5_K_M`** Instruct GGUF.
- If the download is split (`-00001-of-00002.gguf`), point the server at the **`00001`** file.
- Store GGUFs on a drive with enough space (they are multi‑GB).

Other useful Instruct GGUFs for A/B tests: Qwen2.5-14B-Instruct (try **Q4_K_M** on 16 GB), Llama 3.1 8B Instruct (e.g. [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)).

---

## Run

### 1. Start llama-server (keep this terminal open)

From the folder that contains `llama-server.exe`:

```powershell
.\llama-server.exe -m "K:\AI\models\YOUR-MODEL.gguf" --port 8080 -ngl 99 -c 4096
```

| Flag | Meaning |
|------|---------|
| `-m` | Path to your GGUF |
| `--port 8080` | API at `http://127.0.0.1:8080` |
| `-ngl 99` | Offload layers to GPU (`0` = CPU only) |
| `-c 4096` | Context size |

Check that it is up: open `http://127.0.0.1:8080` or `http://127.0.0.1:8080/v1/models`.

Stop the server with **Ctrl+C** in that window.

Optional env vars for the Python client (usually not required — the app can auto-discover the model from `/v1/models`):

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
$env:OPENAI_API_KEY  = "sk-local"
# $env:OPENAI_MODEL  = "K:\AI\models\YOUR-MODEL.gguf"   # optional override
```

### 2. Start the voice pipeline

In a **second** terminal, with the project venv activated:

```powershell
python voice_loop.py
```

Default `"llm": "openai_compat"` expects llama-server to already be running. You should see models loading (STT / SER / TTS), then `Listening...`. Speak into the mic; when you pause, the pipeline transcribes, detects emotion, generates a short NPC reply via llama.cpp, and plays it back. Stop with **Ctrl+C**.

**First run** of Whisper / Chatterbox downloads weights and can take several minutes. Later runs reuse the local cache.

On startup, TTS loads the included clone clip at `voices/reference.wav` (one-time conditioning; does not add per-sentence latency). Replace that file to change the voice, or pass `voice_prompt_path` into `TTSHandler`.

---

## Swapping models

Defaults live in `pipeline_config.py` (`DEFAULT_PIPELINE`).

| Want | What to change |
|------|----------------|
| Different GGUF | Stop llama-server, restart with a new `-m` path; keep `"llm": "openai_compat"` |
| In-process HF LLM (no llama.cpp) | Set `"llm": "qwen2_5_7b"` (or `qwen3_1_7b`) — do **not** also load a large GGUF on the same GPU unless you have headroom |
| STT size | `"stt": "whisper_tiny"` / `"whisper_base"` / … |
| Save each NPC turn as a WAV | `"save_tts_wavs": True` → writes `recordings/turn_<id>.wav` |

You can also override when constructing the pipeline:

```python
VoicePipeline({**DEFAULT_PIPELINE, "stt": "whisper_base", "llm": "qwen2_5_7b"})
```

See `AVAILABLE` / the registries in `pipeline_config.py` for valid keys. Alternate LLM scripts are under `AlternateModels/`.

Latency notes for streaming vs blocking LLM: `LATENCY_NOTES.md`.

---

## Project layout (main pieces)

| File | Role |
|------|------|
| `voice_loop.py` | Main loop (VAD + turn handling) |
| `stt.py` / `ser.py` / `llm.py` / `tts.py` | Pipeline stages |
| `llm_openai.py` | OpenAI-compatible LLM client (llama.cpp / cloud-style endpoints) |
| `pipeline_config.py` | Model registry / defaults |
| `scenario.py` | NPC scenario prompts / emotion hints |
| `interfaces.py` | Shared stage interfaces |
| `setup.py` | Dependency install |
| `TestScripts/` | Stage-level experiments |
| `AlternateModels/` | Extra LLM / TTS options |
| `recordings/` | Optional per-turn TTS WAVs when `save_tts_wavs` is on |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `ModuleNotFoundError` | Activate `venv`, then `python setup.py` again |
| CUDA not available | Confirm `nvidia-smi` works; reinstall torch via `setup.py` (do not install torch from plain PyPI without the cu124 index) |
| `OpenAI-compat unreachable` | Start `llama-server` first; confirm port `8080` and `OPENAI_BASE_URL` |
| GGUF fails to load | Update llama.cpp prebuilt release; confirm Instruct GGUF path; try Q4_K_M if VRAM is tight |
| No mic / no audio | Check Windows privacy settings for mic; confirm default input device |
| Out of VRAM | Smaller Whisper size; smaller / lower-quant GGUF; don’t run HF 7B and llama-server together |
| Slow first start | Normal while models download; later starts should be faster |
