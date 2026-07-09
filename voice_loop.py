import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad
import queue
import threading
import time
from stt import STTHandler
from tts import TTSHandler
from emotion import EmotionDetector

class VoicePipeline:
    def __init__(self):
        self.vad_model = load_silero_vad()
        self.stt = STTHandler(model_size="small")
        self.emotion_detector = EmotionDetector()
        self.tts = TTSHandler()
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.buffer = []
        self.sample_rate = 16000

        print("🎤 Full pipeline ready (VAD + STT + Emotion + TTS). Speak now...")

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata[:, 0].astype(np.float32))

    def vad_worker(self):
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                speech_prob = self.vad_model(torch.from_numpy(chunk), self.sample_rate).item()

                if speech_prob > 0.5:
                    if not self.is_speaking:
                        print("🟢 SPEECH STARTED")
                        self.is_speaking = True
                    self.buffer.extend(chunk)
                else:
                    if self.is_speaking and len(self.buffer) > 4000:  # Lowered for shorter utterances
                        start_time = time.time()
                        
                        audio_np = np.array(self.buffer, dtype=np.float32)
                        text = self.stt.transcribe(audio_np, self.sample_rate)
                        print(f"📝 You said: {text}")

                        if text:
                            emotion = self.emotion_detector.detect(text)
                            tts_start = time.time()
                            self.tts.speak(text, emotion=emotion)
                            tts_latency = time.time() - tts_start
                            total_latency = time.time() - start_time
                            print(f"⏱️ TTS latency: {tts_latency:.2f}s | Total latency: {total_latency:.2f}s")

                        self.buffer = []
                        self.is_speaking = False
            except queue.Empty:
                continue

    def start(self):
        threading.Thread(target=self.vad_worker, daemon=True).start()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                           blocksize=512, callback=self.audio_callback):
            print("Listening... Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nPipeline stopped.")

if __name__ == "__main__":
    pipeline = VoicePipeline()
    pipeline.start()