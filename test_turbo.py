import time
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

text = (
    "Welcome to the world of streaming text to speech. "
    "This audio should be generated in real time chunks."
)
print("Warming up...")
# warmup
model.generate(text)

torch.cuda.synchronize()
start = time.perf_counter()

audio = model.generate(text)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"Turbo generation: {elapsed:.3f}s")
print(f"Audio duration: {audio.shape[-1]/model.sr:.3f}s")