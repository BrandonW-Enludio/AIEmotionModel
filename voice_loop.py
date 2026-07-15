import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad
import queue
import threading
import time

from stt import STTHandler
from tts import TTSHandler
#from emotion import EmotionDetector
from llm import LLMHandler
from ser import SERHandler


class VoicePipeline:
    def __init__(self):

        self.vad_model = load_silero_vad()

        self.stt = STTHandler(model_size="small")
        #self.emotion_detector = EmotionDetector()
        self.ser = SERHandler()
        self.llm = LLMHandler()

        self.tts = TTSHandler()

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

        # User speech state
        self.is_speaking = False
        self.buffer = []

        # Timing
        self.speech_start_time = None
        self.speech_end_time = None

        self.sample_rate = 16000

        print(
            "🎤 Full pipeline ready "
            "(VAD + STT + SER + LLM + TTS)"
        )


    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(
            indata[:, 0].astype(np.float32)
        )


    def vad_worker(self):

        while True:

            try:

                chunk = self.audio_queue.get(timeout=0.1)

                speech_prob = self.vad_model(
                    torch.from_numpy(chunk),
                    self.sample_rate
                ).item()


                # ----------------------------
                # User is speaking
                # ----------------------------
                if speech_prob > 0.5:

                    if not self.is_speaking:
                        print("🟢 SPEECH STARTED")

                        self.speech_start_time = time.time()

                        self.is_speaking = True


                    self.buffer.extend(chunk)


                # ----------------------------
                # User stopped speaking
                # ----------------------------
                else:

                    if self.is_speaking and len(self.buffer) > 8000:


                        self.speech_end_time = time.time()


                        audio_np = np.array(
                            self.buffer,
                            dtype=np.float32
                        )


                        # ----------------------------
                        # Speech to Text
                        # ----------------------------
                        stt_start = time.time()

                        text = self.stt.transcribe(
                            audio_np,
                            self.sample_rate
                        )

                        stt_latency = time.time() - stt_start


                        print(
                            f"📝 You said: {text}"
                        )


                        if text:


                            # ----------------------------
                            # Emotion Detection
                            # ----------------------------
                            ser_result = self.ser.detect(
                                audio_np,
                                self.sample_rate
                            )

                            voice_emotion = ser_result["emotion"]
                            voice_confidence = ser_result["confidence"]
                            emotion_latency = ser_result["latency"]
                            
                            print(
                                f"🎙️ Voice emotion: "
                                f"{voice_emotion} "
                                f"({voice_confidence:.2f})"
                            )

                            # ----------------------------
                            # LLM (streaming by sentence)
                            # ----------------------------
                            llm_start = time.time()
                            llm_first_sentence_latency = None
                            llm_total_latency = None
                            sentence_count = 0
                            got_response = False

                            for chunk in self.llm.generate_response_stream(
                                user_text=text,
                                voice_emotion=voice_emotion,
                                voice_confidence=voice_confidence,
                            ):
                                sentence = chunk["sentence"]
                                sentence_index = chunk["sentence_index"]
                                got_response = True

                                if llm_first_sentence_latency is None:
                                    llm_first_sentence_latency = (
                                        chunk["first_sentence_latency"]
                                    )

                                print(
                                    f"🤖 Sentence {sentence_index + 1}: "
                                    f"{sentence}"
                                )

                                self.tts.speak_sentence_async(
                                    sentence,
                                    emotion=voice_emotion if sentence_index == 0 else None,
                                )
                                sentence_count += 1

                            llm_total_latency = time.time() - llm_start

                            tts_start = time.time()

                            if not got_response:
                                print("⚠️ LLM returned no dialogue. Using fallback.")
                                self.tts.speak_sentence_async(
                                    "I'm here. What can I do for you?",
                                    emotion=voice_emotion,
                                )
                                sentence_count = 1
                                llm_first_sentence_latency = llm_total_latency

                            tts_latency = time.time() - tts_start

                            print(
                                f"📤 Queued {sentence_count} sentence(s) "
                                f"to TTS in {tts_latency:.3f}s"
                            )


                            # ----------------------------
                            # Final Metrics
                            # ----------------------------

                            response_start_latency = (
                                stt_latency
                                + emotion_latency
                                + (llm_first_sentence_latency or llm_total_latency)
                            )


                            end_to_end_latency = (
                                time.time()
                                -
                                self.speech_start_time
                            )


                            print("\n========== PERFORMANCE ==========")

                            print(
                                f"🎤 Speech Recognition: "
                                f"{stt_latency:.2f}s"
                            )

                            print(
                                f"😶 Emotion Analysis: "
                                f"{emotion_latency:.2f}s"
                            )

                            print(
                                f"🤖 LLM first sentence: "
                                f"{llm_first_sentence_latency:.2f}s"
                            )

                            print(
                                f"🤖 LLM total: "
                                f"{llm_total_latency:.2f}s"
                            )

                            print(
                                f"🗣️ TTS queued: {tts_latency:.3f}s"
                            )

                            print(
                                f"⚡ Response Start: "
                                f"{response_start_latency:.2f}s"
                            )

                            print(
                                f"🏁 End-to-End: "
                                f"{end_to_end_latency:.2f}s"
                            )
                            


                            print(
                                "=================================\n"
                            )


                        self.buffer = []
                        self.is_speaking = False


            except queue.Empty:
                continue



    def start(self):

        threading.Thread(
            target=self.vad_worker,
            daemon=True
        ).start()


        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=512,
            callback=self.audio_callback
        ):

            print(
                "Listening... Press Ctrl+C to stop."
            )

            try:

                while True:
                    time.sleep(0.1)

            except KeyboardInterrupt:

                print(
                    "\nPipeline stopped."
                )



if __name__ == "__main__":

    pipeline = VoicePipeline()
    pipeline.start()