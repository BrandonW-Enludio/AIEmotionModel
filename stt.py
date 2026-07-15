from faster_whisper import WhisperModel
import torch
import numpy as np

from interfaces import STTInterface


class STTHandler(STTInterface):
    def __init__(self, model_size="small"):
        print(f"Loading faster-whisper {model_size} model... (first time may take a while)")
        self.model = WhisperModel(
            model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        print("✅ faster-whisper loaded!")

    def transcribe(self, audio_data: np.ndarray, sample_rate=16000):
        # audio_data should be float32 numpy array
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=5,
            language=None,          # Auto-detect
            vad_filter=True,        # Extra safety
            word_timestamps=False
        )
        
        text = " ".join([segment.text for segment in segments]).strip()
        return text if text else ""