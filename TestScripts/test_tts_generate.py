import torch
import soundfile as sf
import simpleaudio as sa

from qwen_tts import Qwen3TTSModel


print("Loading Qwen3-TTS...")

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda",
    dtype=torch.float32
)

print("Loaded!")

text = "Hello, this is a test of the Qwen three text to speech system."

audio_list, sample_rate = tts.generate_custom_voice(
    text=text,
    speaker="vivian",
    language="english",
    instruct="Speak naturally and warmly.",
    do_sample=False
)

audio = audio_list[0]

print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)

output = "test_output.wav"

sf.write(
    output,
    audio,
    sample_rate
)

print("Saved:", output)

wave = sa.WaveObject.from_wave_file(output)

play = wave.play()
play.wait_done()

print("Finished!")