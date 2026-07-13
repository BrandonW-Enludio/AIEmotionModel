import re
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


    def clean_text(self, text: str):

        clean_text = text.strip()

        for marker in ["Assistant:", "assistant:", "NPC:", "Response:", "You are", "Emotion:", "User:"]:
            if marker in clean_text:
                clean_text = clean_text.split(marker)[-1].strip()

        if '.' in clean_text:
            clean_text = clean_text.split('.')[0] + '.'

        clean_text = clean_text.strip('"').strip()

        if not clean_text or len(clean_text) < 5:
            clean_text = "I am here. How can I assist you?"

        return clean_text



    def speak(
        self,
        text: str,
        emotion: str = "neutral",
        intensity: float = 1.0
    ):
        full_text = text

        clean_text = self.clean_text(text)

        # Simple emotion tags for TTS
        emotion_tags = {
            "hap": "[laugh]",
            "sad": "[sigh]",
            "ang": "[shout]",
            "fear": "[gasp]",
            "curious": "[curious]",
            "neutral": ""
        }

        tag = emotion_tags.get(emotion, "")

        prompt_text = f"{tag} {clean_text}".strip()

        print("\n========== WHAT TTS IS RECEIVING ==========")
        print(prompt_text)
        print("===========================================\n")

        generation_start = time.time()

        audio = self.tts.generate(prompt_text)

        generation_latency = time.time() - generation_start

        output_path = "response.wav"

        sf.write(
            output_path,
            audio.squeeze().cpu().numpy(),
            self.sample_rate
        )

        audio_ready_time = time.time()

        playback_start = time.time()

        wave_obj = sa.WaveObject.from_wave_file(output_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        playback_latency = time.time() - playback_start

        print(
            f"🔊 Speaking ({emotion}): {clean_text}"
        )

        print(
            f"   [Full LLM output: {full_text[:150]}...]"
        )

        return {
            "text": clean_text,
            "emotion": emotion,
            "voice": "Chatterbox-Turbo",
            "sample_rate": self.sample_rate,
            "generation_latency": generation_latency,
            "audio_ready_time": audio_ready_time,
            "playback_latency": playback_latency,
            "audio_path": output_path
        }