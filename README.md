# AI Emotion Model — Local Voice Pipeline

Real-time, local voice NPC that listens to you, detects emotion from speech, and answers with spoken dialogue. Everything runs on your machine (no cloud API required for the default stack).

**Pipeline:** mic → VAD → speech-to-text → speech emotion recognition → LLM → text-to-speech

Default models:

| Stage | Model |
|-------|--------|
| STT | faster-whisper `small` |
| SER | `superb/wav2vec2-base-superb-er` |
| LLM | Qwen3-1.7B (4-bit) |
| TTS | Chatterbox Turbo |

---

## Requirements

- **Windows** (setup steps below are for PowerShell)
- **Python 3.11** (e.g. 3.11.9)
- **NVIDIA GPU** with a recent driver (tested with RTX 4070). About **12 GB VRAM** is a comfortable fit for the defaults.
- Working **microphone** and speakers/headphones
- Disk space and internet for the first model downloads (Hugging Face / model weights)

`nvidia-smi` may show CUDA 12.x or 13.x — that number is your **driver’s max supported CUDA**. This project installs **PyTorch CUDA 12.4** wheels, which is fine on newer drivers.

---

## Setup

### 1. Create and activate a virtual environment

From the project root (`AIEmotionModel`):

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

---

## Run

With the venv activated:

```powershell
python voice_loop.py
```

You should see models loading, then `Listening...`. Speak into the mic; when you pause, the pipeline transcribes, detects emotion, generates a short NPC reply, and plays it back. Stop with **Ctrl+C**.

**First run** downloads model weights and can take several minutes. Later runs reuse the local cache.

---

## Swapping models

Defaults live in `pipeline_config.py` (`DEFAULT_PIPELINE`). You can change STT / LLM / TTS keys there, or override when constructing the pipeline in `voice_loop.py`:

```python
VoicePipeline({**DEFAULT_PIPELINE, "stt": "whisper_base", "llm": "gemma2_2b"})
```

See `AVAILABLE` / the registries in `pipeline_config.py` for valid keys. Alternate LLM scripts are under `AlternateModels/`.

Latency notes for streaming vs blocking LLM: `LATENCY_NOTES.md`.

---

## Project layout (main pieces)

| File | Role |
|------|------|
| `voice_loop.py` | Main loop (VAD + turn handling) |
| `stt.py` / `ser.py` / `llm.py` / `tts.py` | Pipeline stages |
| `pipeline_config.py` | Model registry / defaults |
| `interfaces.py` | Shared stage interfaces |
| `setup.py` | Dependency install |
| `TestScripts/` | Stage-level experiments |
| `AlternateModels/` | Extra LLM / TTS options |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `ModuleNotFoundError` | Activate `venv`, then `python setup.py` again |
| CUDA not available | Confirm `nvidia-smi` works; reinstall torch via `setup.py` (do not install torch from plain PyPI without the cu124 index) |
| No mic / no audio | Check Windows privacy settings for mic; confirm default input device |
| Out of VRAM | Use a smaller Whisper size in `pipeline_config.py` (`whisper_base` or `whisper_tiny`) |
| Slow first start | Normal while models download; later starts should be faster |
