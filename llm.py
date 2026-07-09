import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

class LLMHandler:
    def __init__(self):
        print("Loading Gemma-4-12B (4-bit quantized)... This may take a minute.")
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
        
        prompt = f"""You are an emotionally aware NPC.
User ({emotion}): "{user_text}"

Respond naturally and in character. Keep it concise."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new response
        response = response.split("Respond naturally")[-1].strip()
        
        latency = time.time() - start_time
        print(f"🤖 LLM response latency: {latency:.2f}s")
        
        return response.strip()