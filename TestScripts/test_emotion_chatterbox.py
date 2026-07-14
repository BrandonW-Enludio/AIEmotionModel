import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
import soundfile as sf
import simpleaudio as sa
import time

tts = ChatterboxTurboTTS.from_pretrained(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

emotions = {
    "happy": "[laugh] I'm so excited to see you today!",
    "sad": "[sigh] I'm feeling a bit down today.",
    "angry": "[shout] Why did you do that?!",
    "fear": "[gasp] I'm really scared right now.",
    "curious": "[curious] Tell me more about that.",
    "neutral": "Hello, how are you doing today?"
}

print("Testing emotions with Chatterbox-Turbo...\n")

for emotion, prompt in emotions.items():
    print(f"Testing {emotion}...")
    
    audio = tts.generate(prompt, speaker="male1")
    
    output_path = f"test_{emotion}_chatterbox_turbo.wav"
    sf.write(output_path, audio.squeeze().cpu().numpy(), tts.sr)
    
    print(f"  Saved: {output_path}")
    print(f"  Playing {emotion} sample...")
    
    wave_obj = sa.WaveObject.from_wave_file(output_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    
    time.sleep(0.5)  # Short pause between samples

print("\nDemo complete!")