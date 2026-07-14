import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LLMHandler:
    def __init__(self):
        print("Loading Gemma 2 2B...")
        model_name = "google/gemma-2-2b-it"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True},
            trust_remote_code=True
        )
        print("✅ Gemma 2 2B loaded!")

    def generate_response(self, user_text: str, emotion: str = "neutral"):
        start_time = time.time()
        
        prompt = f"""You are a criminal holding a hostage. You are a little confused and very angry.
        User emotion: {emotion}
        User: "{user_text}"

        Respond aggressively, concisely, and in character."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode ONLY the newly generated tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()
        
        latency = time.time() - start_time
        print(f"🤖 LLM latency: {latency:.2f}s")
        
        return response.strip()