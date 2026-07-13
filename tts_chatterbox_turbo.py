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


        # If LLM returned quoted text:
        quoted = re.findall(
            r'"([^"]+)"',
            clean_text
        )

        if quoted:
            clean_text = quoted[-1].strip()


        # Remove common prefixes
        prefixes = [
            "Assistant:",
            "assistant:",
            "NPC:",
            "Response:"
        ]

        for prefix in prefixes:

            if clean_text.startswith(prefix):

                clean_text = (
                    clean_text[len(prefix):]
                    .strip()
                )


        # Remove remaining quotes
        clean_text = (
            clean_text
            .strip('"')
            .strip()
        )


        if not clean_text:

            clean_text = (
                "I am here. "
                "How can I assist you?"
            )


        return clean_text



    def speak(
        self,
        text: str,
        emotion: str = "neutral",
        intensity: float = 1.0
    ):
        full_text = text
        print("\n========== TTS INPUT DEBUG ==========")
        print(text)
        print("=====================================\n")


        # -----------------------------
        # Text preparation
        # -----------------------------

        clean_text = self.clean_text(text)


        if emotion == "happy":

            prompt_text = (
                f"[laugh] {clean_text}"
            )

        elif emotion == "sad":

            prompt_text = (
                f"[sigh] {clean_text}"
            )

        elif emotion == "angry":

            prompt_text = (
                f"[shout] {clean_text}"
            )

        elif emotion == "fear":

            prompt_text = (
                f"[gasp] {clean_text}"
            )

        else:

            prompt_text = clean_text



        # -----------------------------
        # TTS Generation
        # -----------------------------

        generation_start = time.time()


        audio = self.tts.generate(
            prompt_text
        )


        generation_latency = (
            time.time()
            -
            generation_start
        )
        

        output_path = "response.wav"


        sf.write(
            output_path,
            audio.squeeze().cpu().numpy(),
            self.sample_rate
        )

        audio_ready_time = time.time()
        
        # -----------------------------
        # Playback
        # -----------------------------

        playback_start = time.time()


        wave_obj = sa.WaveObject.from_wave_file(
            output_path
        )

        play_obj = wave_obj.play()

        play_obj.wait_done()


        playback_latency = (
            time.time()
            -
            playback_start
        )


        # -----------------------------
        # Logging
        # -----------------------------

        print(
            f"🔊 Speaking ({emotion}): "
            f"{clean_text}"
        )

        print(
            f"   [Full LLM output: {full_text}]"
        )

        # -----------------------------
        # Return structured data
        # -----------------------------

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