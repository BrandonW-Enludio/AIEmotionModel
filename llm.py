import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

class LLMHandler:
    def __init__(self):
        print("Loading Qwen3 1.7B (more human-like mode)...")
        model_name = "Qwen/Qwen3-1.7B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("✅ Qwen3 1.7B loaded!")

    def generate_response(self, user_text: str, voice_emotion="neu", voice_confidence=0.0):
        start_time = time.time()

        messages = [
            {
            "role": "system",
            "content":
            """
            You are a friendly, natural NPC in a game.

            Respond only with dialogue the character would say aloud.

            Use the player's detected voice emotion as subtle context:
            - Happy players should receive warmer, more energetic responses.
            - Sad or fearful players should receive calmer, more supportive responses.
            - Angry players should receive patient, de-escalating responses.
            - Neutral players should receive normal conversational responses.

            Emotion confidence indicates how much you should trust the emotion:
            - High confidence: let it influence your tone more.
            - Low confidence: keep your response more neutral.

            Never mention emotions, confidence scores, or analysis.
            Keep responses short: one or two sentences.
            """
            },
            {
                "role": "user",
                "content":
                f"""
            Emotion: {voice_emotion}
            Emotion confidence: {voice_confidence:.2f}

            Player said:
            {user_text}
            """
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.85,          # Higher = more human variation
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.15    # Helps avoid repetition
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean
        response = response.split("</think>")[-1].strip()

        return response