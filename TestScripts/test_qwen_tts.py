import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import simpleaudio as sa
import time

print("Loading Qwen3-TTS 1.7B...")
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
print("✅ Loaded!\n")

emotions = {
    "happy": "Speak with a cheerful, friendly tone and a slight smile.",
    "sad": "Speak softly with a subdued and gentle tone.",
    "angry": "Speak firmly with controlled frustration.",
    "fear": "Speak cautiously with slight nervousness.",
    "curious": "Speak with an interested, inquisitive tone.",
    "neutral": "Speak naturally and conversationalally."
}

test_sentences = [
    "Hello, how are you today?",
    "I'm really excited about this!",
    "That makes me so sad...",
    "I'm a bit scared right now.",
    "Tell me more about that.",
    "It's just a normal day."
]

for i, (emotion, instruct) in enumerate(emotions.items()):
    print(f"Testing {emotion}...")

    audio_list, sr = tts.generate_custom_voice(
        text=test_sentences[i],
        speaker="vivian",
        language="english",
        instruct=instruct
    )

    audio = audio_list[0]

    output_path = f"test_{emotion}_1.7b.wav"
    sf.write(output_path, audio, sr)

    print(f"  Saved: {output_path}")
    print(f"  Playing {emotion} sample...")

    wave_obj = sa.WaveObject.from_wave_file(output_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    time.sleep(0.5)

print("\nQwen3-TTS 1.7B emotion demo complete!")