import torch
import time
import numpy as np
from transformers import pipeline


class SERHandler:
    def __init__(self):
        print("Loading SER model...")

        self.device = 0 if torch.cuda.is_available() else -1

        self.classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=self.device
        )

        print(
            f"✅ SER loaded on "
            f"{'CUDA' if self.device == 0 else 'CPU'}"
        )


    def detect(self, audio_data: np.ndarray, sample_rate=16000):

        start = time.time()

        result = self.classifier(
            {
                "array": audio_data,
                "sampling_rate": sample_rate
            }
        )

        best = result[0]

        latency = time.time() - start

        return {
            "emotion": best["label"],
            "confidence": best["score"],
            "latency": latency
        }