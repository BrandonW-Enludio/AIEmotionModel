import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
import soundfile as sf
import simpleaudio as sa
import time

class TTSHandler:
    def __init__(self):
        print("Loading Chatterbox-Turbo...")
        self.tts = ChatterboxTurboTTS.from_pretrained(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = self.tts.sr
        print("✅ Chatterbox-Turbo loaded!")

    def speak(self, text: str, emotion: str = "neutral", intensity: float = 1.0):
        start_time = time.time()

        # Clean any leaked emotion tags
        clean_text = text.replace(f"({emotion})", "").replace(f"[{emotion}]", "").strip()

        # Simple emotion prompting
        prompt_text = clean_text
        if emotion != "neutral":
            prompt_text = f"[{emotion}] {clean_text}"

        audio = self.tts.generate(prompt_text)

        output_path = "response.wav"
        sf.write(output_path, audio.squeeze().cpu().numpy(), self.sample_rate)

        tts_latency = time.time() - start_time
        print(f"🔊 Speaking ({emotion}): {clean_text}")
        print(f"⏱️ TTS generation latency: {tts_latency:.2f}s")

        wave_obj = sa.WaveObject.from_wave_file(output_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        return output_path