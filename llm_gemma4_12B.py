import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LLMHandler:
    def __init__(self):
        print("Loading Gemma-4-12B (4-bit)...")
        model_name = "google/gemma-4-12B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True},
            trust_remote_code=True
        )
        print("✅ Gemma-4-12B loaded!")

    def generate_response(self, user_text: str, emotion: str = "neutral"):
        start_time = time.time()
        
        prompt = f"""<start_of_turn>user
You are an emotionally aware NPC in a game.
User emotion: {emotion}
User said: "{user_text}"
<end_of_turn>
<start_of_turn>model
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<start_of_turn>model")[-1].strip()
        
        latency = time.time() - start_time
        print(f"🤖 LLM latency: {latency:.2f}s")
        
        return response.strip()