from transformers import pipeline
import torch

class EmotionDetector:
    def __init__(self):
        print("Loading emotion classifier...")
        self.classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None
        )
        print("✅ Emotion detector ready!")

    def detect(self, text: str):
        if not text or len(text.strip()) < 5:
            return "neutral"
        
        result = self.classifier(text)[0]
        best = max(result, key=lambda x: x['score'])
        emotion = best['label'].lower().replace('_', ' ')
        
        print(f"😶 Emotion detected: {emotion} ({best['score']:.2f})")
        return emotion