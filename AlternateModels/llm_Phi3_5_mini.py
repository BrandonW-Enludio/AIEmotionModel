import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMHandler:
    def __init__(self):
        print("Loading Phi-3.5-mini...")
        model_name = "microsoft/Phi-3.5-mini-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # trust_remote_code=False: use transformers' built-in Phi-3 code.
        # Remote hub code still references Cache.seen_tokens, which was removed.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=False,
        )
        print("✅ Phi-3.5-mini loaded!")

    def _clean_response(self, text: str) -> str:
        text = text.strip()
        # Strip common instruction / meta leakage.
        for marker in [
            "Assistant:",
            "assistant:",
            "NPC:",
            "Response:",
            "Dialogue:",
            "User emotion:",
            "User:",
            "System:",
        ]:
            if marker in text:
                text = text.split(marker)[-1].strip()

        # Drop obvious thinking / planning blocks if present.
        text = re.sub(
            r"(?is)<think>.*?</think>",
            "",
            text,
        ).strip()
        text = re.sub(r"(?im)^(thinking|instructions?|notes?):.*$", "", text).strip()

        text = text.strip().strip('"').strip()
        return text

    def generate_response(self, user_text: str, emotion: str = "neutral"):
        start_time = time.time()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, natural NPC in a game. "
                    "Respond only with dialogue the character would say aloud. "
                    "Never mention emotions, instructions, analysis, or thinking. "
                    "Keep responses short: one or two sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Player emotion: {emotion}\n"
                    f"Player said: {user_text}"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        # Decode ONLY new tokens — decoding the full sequence re-includes the prompt.
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        response = self._clean_response(response)

        latency = time.time() - start_time
        print(f"🤖 LLM latency: {latency:.2f}s")

        return response
