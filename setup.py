# setup.py
import subprocess
import sys

def run_command(cmd):
    print(f"→ {cmd}")
    subprocess.check_call(cmd, shell=True)

print("Setting up Local Voice Pipeline...\n")

run_command("pip install --upgrade pip")
run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
run_command("pip install -r requirements.txt")

# Special install for Chatterbox
print("\nInstalling Chatterbox from GitHub...")
run_command("pip install git+https://github.com/resemble-ai/chatterbox.git")

print("\n✅ Setup complete! Run: python voice_loop.py")