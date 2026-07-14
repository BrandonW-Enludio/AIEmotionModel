import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LLMHandler:
    def __init__(self):
        print("Loading Phi-3.5-mini...")
        model_name = "microsoft/Phi-3.5-mini-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={"load_in_4bit": True},
            trust_remote_code=True
        )
        print("✅ Phi-3.5-mini loaded!")

    def generate_response(self, user_text: str, emotion: str = "neutral"):
        start_time = time.time()
        
        prompt = f"""<|user|>
You are an emotionally aware NPC. Respond naturally and concisely in character.
User emotion: {emotion}
User: "{user_text}"
<|assistant|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        latency = time.time() - start_time
        print(f"🤖 LLM latency: {latency:.2f}s")
        
        return response.strip()