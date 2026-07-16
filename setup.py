# setup.py — install packages needed to run voice_loop.py
#
# Prerequisites (do these yourself first):
#   1. Python 3.11
#   2. Create a venv:  python -m venv venv
#   3. Activate it:    venv\Scripts\activate   (Windows)
#   4. Then run:       python setup.py
#
# Assumes NVIDIA GPU + driver that can run CUDA 12.4 PyTorch wheels
# (nvidia-smi may show a newer CUDA version; that is fine).

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"→ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    py = sys.executable

    print("Setting up Local Voice Pipeline (minimal install)...\n")

    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    # CUDA torch first (not on default PyPI).
    run([
        py, "-m", "pip", "install",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])

    run([py, "-m", "pip", "install", "chatterbox-tts"])
    run([py, "-m", "pip", "install", "sounddevice"])
    run([
        py, "-m", "pip", "install",
        "numpy",
        "silero-vad",
        "faster-whisper",
        "transformers",
        "accelerate",
        "bitsandbytes",
    ])

    print("\nSetup complete. First run of voice_loop.py will download models.")
    print("Run:  python voice_loop.py")


if __name__ == "__main__":
    main()
