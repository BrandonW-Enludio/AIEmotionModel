# setup.py — recreate the project virtualenv on a new machine
import subprocess
import sys


def run_command(cmd):
    print(f"→ {cmd}")
    subprocess.check_call(cmd, shell=True)


print("Setting up Local Voice Pipeline (Python 3.11 + CUDA 12.4)...\n")

run_command(f'"{sys.executable}" -m pip install --upgrade pip')

# Install CUDA torch builds first (not on default PyPI).
run_command(
    f'"{sys.executable}" -m pip install '
    "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
    "--index-url https://download.pytorch.org/whl/cu124"
)

run_command(f'"{sys.executable}" -m pip install -r requirements.txt')

print("\nInstalling Chatterbox from GitHub...")
run_command(
    f'"{sys.executable}" -m pip install '
    "git+https://github.com/resemble-ai/chatterbox.git"
)

print("\nInstalling spaCy English model (Chatterbox dependency)...")
run_command(f'"{sys.executable}" -m spacy download en_core_web_sm')

print("\n✅ Setup complete! Activate your venv, then run: python voice_loop.py")
