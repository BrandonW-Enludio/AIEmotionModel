import torch
import time
import queue
import threading
import re

import sounddevice as sd
from chatterbox.tts_turbo import ChatterboxTurboTTS


class TTSHandler:
    def __init__(self):
        print("Loading Chatterbox-Turbo...")
        self.tts = ChatterboxTurboTTS.from_pretrained(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = self.tts.sr
        print("✅ Chatterbox-Turbo loaded!")

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=6)   # Bigger buffer = smoother

        self.running = True
        self.job_id = 0

        threading.Thread(target=self.tts_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()

    def clean_text(self, text):
        clean_text = text.strip()
        for marker in ["Assistant:", "assistant:", "NPC:", "Response:", "Emotion:", "User:"]:
            if marker in clean_text:
                clean_text = clean_text.split(marker)[-1].strip()
        clean_text = clean_text.strip('"').strip()
        if not clean_text or len(clean_text) < 5:
            clean_text = "I am here. How can I assist you?"
        return clean_text

    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def tts_worker(self):
        while self.running:
            item = self.text_queue.get()
            if item is None:
                break

            full_text, metadata = item
            job_id = metadata["id"]
            request_time = metadata["request_time"]
            split_sentences = metadata.get("split_sentences", True)

            if split_sentences:
                sentences = self.split_into_sentences(full_text)
            else:
                sentences = [full_text.strip()]

            print(f"🧠 TTS WORKER STARTING #{job_id} — {len(sentences)} sentences")

            for i, sentence in enumerate(sentences, 1):
                print(f"   Generating sentence {i}/{len(sentences)}: {sentence[:70]}{'...' if len(sentence)>70 else ''}")

                gen_start = time.perf_counter()
                audio = self.tts.generate(sentence)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                gen_time = time.perf_counter() - gen_start
                audio_np = audio.squeeze().cpu().numpy()
                audio_duration = len(audio_np) / self.sample_rate

                print(f"   ✅ Sentence {i} ready in {gen_time:.3f}s ({audio_duration:.2f}s audio)")

                # Pass sentence-specific info
                self.audio_queue.put((
                    audio_np,
                    {
                        "id": job_id,
                        "sentence_num": i,
                        "total_sentences": len(sentences),
                        "request_time": request_time,   # original for first sentence
                        "sentence_start_time": time.perf_counter()  # better for later sentences
                    },
                    gen_time,
                    audio_duration
                ))

            print(f"🎯 All sentences for job #{job_id} queued\n")

    def playback_worker(self):
        while self.running:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_np, meta, gen_time, audio_dur = item
            sentence_info = f"Sentence {meta['sentence_num']}/{meta['total_sentences']}"

            playback_start = time.perf_counter()

            print(f"🔊 PLAYING {sentence_info} — Job #{meta['id']}")

            sd.play(audio_np, self.sample_rate, blocking=True)

            print(f"✅ FINISHED {sentence_info} — Job #{meta['id']}")

            playback_time = time.perf_counter() - playback_start

            # Better latency calculation
            if meta['sentence_num'] == 1:
                response_latency = playback_start - meta["request_time"]
            else:
                response_latency = playback_start - meta["sentence_start_time"]

            print("\n========== LATENCY ==========")
            print(f"Response start latency: {response_latency:.3f}s")
            print(f"TTS generation:         {gen_time:.3f}s")
            print(f"Playback duration:      {playback_time:.3f}s  ({audio_dur:.2f}s audio)")
            print("=============================\n")

    def speak_async(self, text, emotion="neutral"):
        self._enqueue_text(text, emotion=emotion, split_sentences=True)

    def speak_sentence_async(self, text, emotion=None):
        """Queue a single pre-split sentence. Pass emotion only for the first sentence."""
        self._enqueue_text(text, emotion=emotion, split_sentences=False)

    def _enqueue_text(self, text, emotion=None, split_sentences=True):
        emotion_tags = {
            "hap": "[laugh]", "sad": "[sigh]", "ang": "[shout]",
            "fear": "[gasp]", "curious": "[curious]", "neutral": ""
        }

        tag = emotion_tags.get(emotion, "") if emotion else ""
        clean_text = self.clean_text(text)
        prompt_text = f"{tag} {clean_text}".strip()

        print("\n========== WHAT TTS RECEIVES ==========")
        print(prompt_text)
        print("========================================")

        self.job_id += 1
        self.text_queue.put((
            prompt_text,
            {
                "id": self.job_id,
                "request_time": time.perf_counter(),
                "text": clean_text,
                "emotion": emotion,
                "split_sentences": split_sentences,
            }
        ))

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.audio_queue.put(None)