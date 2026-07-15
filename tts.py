import torch
import time
import queue
import threading
import re

import sounddevice as sd
from chatterbox.tts_turbo import ChatterboxTurboTTS


class TTSHandler:
    def __init__(self, on_turn_complete=None):
        print("Loading Chatterbox-Turbo...")
        self.tts = ChatterboxTurboTTS.from_pretrained(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = self.tts.sr
        print("✅ Chatterbox-Turbo loaded!")

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=6)

        self.running = True
        self.job_id = 0
        self.on_turn_complete = on_turn_complete

        self._turns_lock = threading.Lock()
        self._turns = {}

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

    def begin_turn(self, turn_id, speech_end_time):
        with self._turns_lock:
            self._turns[turn_id] = {
                "speech_end_time": speech_end_time,
                "expected": None,
                "played": 0,
                "finished": False,
                "sentences": [],
                "last_playback_end": None,
                "first_playback_start": None,
            }

    def close_turn(self, turn_id, expected_sentences):
        with self._turns_lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn["expected"] = expected_sentences
            should_finish = (
                turn["expected"] is not None
                and turn["played"] >= turn["expected"]
                and not turn["finished"]
            )
            if should_finish:
                turn["finished"] = True
                result = self._build_turn_result(turn_id, turn)

        if should_finish:
            self._emit_turn_complete(turn_id, result)

    def tts_worker(self):
        while self.running:
            item = self.text_queue.get()
            if item is None:
                break

            full_text, metadata = item
            split_sentences = metadata.get("split_sentences", True)

            if split_sentences:
                sentences = self.split_into_sentences(full_text)
            else:
                sentences = [full_text.strip()]

            for i, sentence in enumerate(sentences, 1):
                gen_start = time.perf_counter()
                audio = self.tts.generate(sentence)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                gen_time = time.perf_counter() - gen_start
                audio_np = audio.squeeze().cpu().numpy()
                audio_duration = len(audio_np) / self.sample_rate

                self.audio_queue.put((
                    audio_np,
                    {
                        **metadata,
                        "sentence_num": metadata.get("sentence_index", i - 1) + 1,
                        "total_sentences": len(sentences),
                        "gen_time": gen_time,
                        "audio_duration": audio_duration,
                    },
                ))

    def playback_worker(self):
        while self.running:
            item = self.audio_queue.get()
            if item is None:
                break

            audio_np, meta = item
            turn_id = meta.get("turn_id")
            playback_start = time.time()
            gap_from_previous = None

            if turn_id is not None:
                with self._turns_lock:
                    turn = self._turns.get(turn_id)
                    if turn is not None:
                        if turn["first_playback_start"] is None:
                            turn["first_playback_start"] = playback_start
                        if turn["last_playback_end"] is not None:
                            gap_from_previous = playback_start - turn["last_playback_end"]

            sd.play(audio_np, self.sample_rate, blocking=True)

            playback_end = time.time()
            playback_time = playback_end - playback_start

            if turn_id is None:
                continue

            result = None
            with self._turns_lock:
                turn = self._turns.get(turn_id)
                if turn is None:
                    continue

                turn["sentences"].append({
                    "index": meta.get("sentence_index", meta["sentence_num"] - 1),
                    "gen_time": meta.get("gen_time", 0.0),
                    "audio_duration": meta.get("audio_duration", 0.0),
                    "playback_time": playback_time,
                    "gap_from_previous": gap_from_previous,
                })
                turn["played"] += 1
                turn["last_playback_end"] = playback_end

                should_finish = (
                    turn["expected"] is not None
                    and turn["played"] >= turn["expected"]
                    and not turn["finished"]
                )
                if should_finish:
                    turn["finished"] = True
                    result = self._build_turn_result(turn_id, turn)

            if result is not None:
                self._emit_turn_complete(turn_id, result)

    def _build_turn_result(self, turn_id, turn):
        first_start = turn["first_playback_start"]
        speech_end = turn["speech_end_time"]
        response_start = (
            (first_start - speech_end) if first_start is not None else None
        )
        return {
            "turn_id": turn_id,
            "response_start": response_start,
            "sentences": list(turn["sentences"]),
            "playback_end": turn["last_playback_end"],
        }

    def _emit_turn_complete(self, turn_id, result):
        with self._turns_lock:
            self._turns.pop(turn_id, None)

        if self.on_turn_complete is not None:
            self.on_turn_complete(result)

    def speak_async(self, text, emotion="neutral", turn_id=None, sentence_index=0):
        self._enqueue_text(
            text,
            emotion=emotion,
            split_sentences=True,
            turn_id=turn_id,
            sentence_index=sentence_index,
        )

    def speak_sentence_async(
        self,
        text,
        emotion=None,
        turn_id=None,
        sentence_index=0,
    ):
        """Queue a single pre-split sentence. Pass emotion only for the first sentence."""
        self._enqueue_text(
            text,
            emotion=emotion,
            split_sentences=False,
            turn_id=turn_id,
            sentence_index=sentence_index,
        )

    def _enqueue_text(
        self,
        text,
        emotion=None,
        split_sentences=True,
        turn_id=None,
        sentence_index=0,
    ):
        emotion_tags = {
            "hap": "[laugh]", "sad": "[sigh]", "ang": "[shout]",
            "fear": "[gasp]", "curious": "[curious]", "neutral": ""
        }

        tag = emotion_tags.get(emotion, "") if emotion else ""
        clean_text = self.clean_text(text)
        prompt_text = f"{tag} {clean_text}".strip()

        self.job_id += 1
        self.text_queue.put((
            prompt_text,
            {
                "id": self.job_id,
                "request_time": time.perf_counter(),
                "text": clean_text,
                "emotion": emotion,
                "split_sentences": split_sentences,
                "turn_id": turn_id,
                "sentence_index": sentence_index,
            }
        ))

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.audio_queue.put(None)
