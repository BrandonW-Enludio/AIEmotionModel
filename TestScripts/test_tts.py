import torch
from qwen_tts import Qwen3TTSModel

print("Loading...")

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda",
    dtype=torch.bfloat16
)

print("Loaded!")

print(tts.get_supported_speakers())
print(tts.get_supported_languages())