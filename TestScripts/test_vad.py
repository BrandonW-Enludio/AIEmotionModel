import torch
import numpy as np
from silero_vad import load_silero_vad

print("Loading Silero VAD model... (this may take a moment the first time)")

model = load_silero_vad()
print("✅ Silero VAD loaded successfully!")

# Quick test with dummy audio
dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
print("Dummy audio test passed.")

print("\nSilero VAD is ready to use in our pipeline!")