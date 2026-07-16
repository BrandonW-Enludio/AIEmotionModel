import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad
import queue
import threading
import time

from pipeline_config import DEFAULT_PIPELINE, build_handlers


class VoicePipeline:
    def __init__(self, pipeline_config=None):

        self.vad_model = load_silero_vad()

        handlers = build_handlers(
            pipeline_config or DEFAULT_PIPELINE,
            on_turn_complete=self._on_turn_complete,
        )
        self.stt = handlers["stt"]
        self.ser = handlers["ser"]
        self.llm = handlers["llm"]
        self.tts = handlers["tts"]
        self.pipeline_config = handlers["config"]

        if torch.cuda.is_available():
            print("\n========== GPU MEMORY ==========")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"VRAM Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print("================================\n")
        else:
            print("CUDA not available. Running on CPU.")

        self.audio_queue = queue.Queue()

        self.is_speaking = False
        self.buffer = []

        self.speech_start_time = None
        self.speech_end_time = None

        self.sample_rate = 16000
        self.turn_id = 0
        self._pending_turns = {}
        self._pending_lock = threading.Lock()

        print(
            "🎤 Pipeline ready | "
            f"STT={self.pipeline_config['stt']} "
            f"SER={self.pipeline_config['ser']} "
            f"LLM={self.pipeline_config['llm']} "
            f"TTS={self.pipeline_config['tts']}"
        )

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata[:, 0].astype(np.float32))

    def _on_turn_complete(self, tts_result):
        turn_id = tts_result["turn_id"]
        with self._pending_lock:
            turn = self._pending_turns.pop(turn_id, None)

        if turn is None:
            return

        self._print_performance(turn, tts_result)

    def _print_performance(self, turn, tts_result):
        response_start = tts_result.get("response_start")
        llm_sentences = turn["llm_sentences"]
        tts_sentences = tts_result.get("sentences", [])

        print("\n========== PERFORMANCE ==========")
        print(f"Task #{turn['turn_id']}")
        print(
            f"Models: "
            f"{self.pipeline_config['stt']} | "
            f"{self.pipeline_config['ser']} | "
            f"{self.pipeline_config['llm']} | "
            f"{self.pipeline_config['tts']}"
        )
        print(f"STT:              {turn['stt_latency']:.2f}s")
        print(f"Emotion:          {turn['emotion_latency']:.2f}s")

        if llm_sentences:
            first = llm_sentences[0]
            print(f"LLM total:        {turn['llm_total']:.2f}s")
        else:
            print("LLM:              (no dialogue)")

        if response_start is not None:
            print(f"Response Start:   {response_start:.2f}s  (speech end → first audio)")
        else:
            print("Response Start:   n/a")

        if tts_sentences:
            first_tts = tts_sentences[0]
            print(
                f"TTS sent. 1:      gen {first_tts['gen_time']:.2f}s | "
                f"audio {first_tts['audio_duration']:.2f}s"
            )
            for item in tts_sentences[1:]:
                gap = item["gap_from_previous"]
                gap_str = f"{gap:.3f}s" if gap is not None else "n/a"
                print(
                    f"TTS sent. {item['index'] + 1}:      "
                    f"gap {gap_str} | "
                    f"gen {item['gen_time']:.2f}s | "
                    f"audio {item['audio_duration']:.2f}s"
                )

        if turn["speech_start_time"] is not None and tts_result.get("playback_end"):
            end_to_end = tts_result["playback_end"] - turn["speech_start_time"]
            print(f"End-to-End:       {end_to_end:.2f}s  (speech start → last audio)")

        print("=================================\n")

    def vad_worker(self):
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)

                speech_prob = self.vad_model(
                    torch.from_numpy(chunk),
                    self.sample_rate
                ).item()

                if speech_prob > 0.5:
                    if not self.is_speaking:
                        print("🟢 SPEECH STARTED")
                        self.speech_start_time = time.time()
                        self.is_speaking = True

                    self.buffer.extend(chunk)

                else:
                    if self.is_speaking and len(self.buffer) > 8000:
                        self.speech_end_time = time.time()
                        audio_np = np.array(self.buffer, dtype=np.float32)

                        stt_start = time.time()
                        text = self.stt.transcribe(audio_np, self.sample_rate)
                        stt_latency = time.time() - stt_start

                        print(f"📝 You said: {text}")

                        if text:
                            ser_result = self.ser.detect(audio_np, self.sample_rate)
                            voice_emotion = ser_result["emotion"]
                            voice_confidence = ser_result["confidence"]
                            emotion_latency = ser_result["latency"]

                            print(
                                f"🎙️ Voice emotion: "
                                f"{voice_emotion} ({voice_confidence:.2f})"
                            )

                            self.turn_id += 1
                            turn_id = self.turn_id
                            speech_start = self.speech_start_time
                            speech_end = self.speech_end_time

                            turn = {
                                "turn_id": turn_id,
                                "speech_start_time": speech_start,
                                "speech_end_time": speech_end,
                                "stt_latency": stt_latency,
                                "emotion_latency": emotion_latency,
                                "llm_sentences": [],
                                "llm_total": 0.0,
                            }
                            with self._pending_lock:
                                self._pending_turns[turn_id] = turn

                            self.tts.begin_turn(turn_id, speech_end)

                            llm_start = time.time()
                            sentence_count = 0

                            for chunk in self.llm.generate_response_stream(
                                user_text=text,
                                voice_emotion=voice_emotion,
                                voice_confidence=voice_confidence,
                            ):
                                sentence = chunk["sentence"]
                                sentence_index = chunk["sentence_index"]

                                turn["llm_sentences"].append({
                                    "index": sentence_index,
                                    "sentence_latency": chunk["sentence_latency"],
                                    "delta_latency": chunk["delta_latency"],
                                })

                                self.tts.speak_sentence_async(
                                    sentence,
                                    emotion=voice_emotion if sentence_index == 0 else None,
                                    turn_id=turn_id,
                                    sentence_index=sentence_index,
                                )
                                sentence_count += 1

                            turn["llm_total"] = time.time() - llm_start

                            if sentence_count == 0:
                                print("⚠️ LLM returned no dialogue. Using fallback.")
                                turn["llm_sentences"].append({
                                    "index": 0,
                                    "sentence_latency": turn["llm_total"],
                                    "delta_latency": turn["llm_total"],
                                })
                                self.tts.speak_sentence_async(
                                    "I'm here. What can I do for you?",
                                    emotion=voice_emotion,
                                    turn_id=turn_id,
                                    sentence_index=0,
                                )
                                sentence_count = 1

                            self.tts.close_turn(turn_id, sentence_count)

                        self.buffer = []
                        self.is_speaking = False

            except queue.Empty:
                continue

    def start(self):
        threading.Thread(target=self.vad_worker, daemon=True).start()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=512,
            callback=self.audio_callback
        ):
            print("Listening... Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nPipeline stopped.")


if __name__ == "__main__":
    # Swap models by editing DEFAULT_PIPELINE in pipeline_config.py
    # or overriding here, e.g.:
    #   VoicePipeline({**DEFAULT_PIPELINE, "stt": "whisper_base", "llm": "gemma2_2b"})
    pipeline = VoicePipeline()
    pipeline.start()
