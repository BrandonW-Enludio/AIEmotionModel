import torch
import soundfile as sf
import simpleaudio as sa
import re
import time

from qwen_tts import Qwen3TTSModel

class TTSHandler:
    def __init__(self):
        print("Loading Qwen3-TTS...")
        self.tts = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cuda",
            dtype=torch.float32
        )
        self.sample_rate = 24000
        print("✅ Qwen3-TTS loaded!")

    def speak(self, text, emotion="neu", intensity=1.0):
        start = time.time()

        emotion_styles = {
            "hap": "Speak with a cheerful, friendly tone and a slight smile.",
            "sad": "Speak softly with a subdued and gentle tone.",
            "ang": "Speak firmly with controlled frustration.",
            "fea": "Speak cautiously with slight nervousness.",
            "neu": "Speak naturally and conversationalally. Avoid exaggerated emotion."
        }

        confidence_phrase = ""
        if intensity > 0.85:
            confidence_phrase = "The emotional tone should be clear and noticeable."
        elif intensity > 0.60:
            confidence_phrase = "The emotional tone should be present but natural."
        else:
            confidence_phrase = "Keep the emotional tone subtle."

        instruct = emotion_styles.get(emotion, emotion_styles["neu"]) + " " + confidence_phrase

        # Clean text
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text).strip()

        audio_list, sr = self.tts.generate_custom_voice(
            text=text,
            speaker="vivian",
            language="chinese",
            instruct=instruct
        )

        audio = audio_list[0]

        output = "response.wav"
        sf.write(output, audio, sr)

        generation_latency = time.time() - start

        print(f"🔊 Speaking ({emotion}): {text}")
        print(f"⏱️ TTS generation: {generation_latency:.2f}s")

        playback_start = time.time()
        wave = sa.WaveObject.from_wave_file(output)
        play = wave.play()
        play.wait_done()
        playback_latency = time.time() - playback_start

        return {
            "generation_latency": generation_latency,
            "playback_latency": playback_latency
        }