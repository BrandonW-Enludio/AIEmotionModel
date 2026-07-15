import queue
import re
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.generation.streamers import BaseStreamer


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


class TokenDecodeStreamer(BaseStreamer):
    """
    Accumulate new token ids and decode the full generated sequence each step.

    HuggingFace generate() behavior:
      1) first put() = full prompt ids
      2) later put()s = only the newly generated token(s)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_ids = []
        self.last_text = ""
        self.text_queue = queue.Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        # First call is the prompt — skip it, only accumulate new tokens.
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.generated_ids.extend(value.tolist())

        text = self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=True,
        )

        if text != self.last_text:
            self.last_text = text
            self.text_queue.put(text)

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        while True:
            item = self.text_queue.get()
            if item is None:
                break
            yield item


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

    def _build_messages(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
    ):
        return [
            {
                "role": "system",
                "content": """
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
            """,
            },
            {
                "role": "user",
                "content": f"""
            Emotion: {voice_emotion}
            Emotion confidence: {voice_confidence:.2f}

            Player said:
            {user_text}
            """,
            },
        ]

    def _prepare_inputs(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
    ):
        messages = self._build_messages(
            user_text,
            voice_emotion,
            voice_confidence,
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def _generation_kwargs(self, streamer=None):
        kwargs = {
            "max_new_tokens": 80,
            "temperature": 0.85,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    def _clean_fragment(self, text: str) -> str:
        thinking_end = "</" + "redacted_thinking>"
        text = text.split(thinking_end)[-1].strip()
        for marker in [
            "Assistant:",
            "assistant:",
            "NPC:",
            "Response:",
            "Emotion:",
            "User:",
        ]:
            if marker in text:
                text = text.split(marker)[-1].strip()
        return text.strip().strip('"').strip()

    def _split_complete_sentences(self, cleaned_buffer: str):
        parts = SENTENCE_SPLIT.split(cleaned_buffer)
        if len(parts) > 1:
            complete = [part.strip() for part in parts[:-1] if part.strip()]
            remainder = parts[-1].strip()
        else:
            complete = []
            remainder = cleaned_buffer.strip()

        # Flush final fragment when it already ends with sentence punctuation.
        if remainder and re.search(r'[.!?]$', remainder):
            complete.append(remainder)
            remainder = ""

        return complete, remainder

    def generate_response_stream(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
    ):
        """
        Stream LLM output and yield complete sentences as they arrive.

        Yields dicts:
            {"sentence": str, "sentence_index": int, "first_sentence_latency": float|None}
        """
        inputs = self._prepare_inputs(
            user_text,
            voice_emotion,
            voice_confidence,
        )

        streamer = TokenDecodeStreamer(self.tokenizer)

        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
                **self._generation_kwargs(streamer=streamer),
            },
        )

        start_time = time.time()
        first_sentence_latency = None
        sentence_index = 0
        emitted_count = 0
        remainder = ""

        generation_thread.start()

        for full_text in streamer:
            cleaned = self._clean_fragment(full_text)
            complete_sentences, remainder = self._split_complete_sentences(cleaned)

            while emitted_count < len(complete_sentences):
                sentence = complete_sentences[emitted_count]
                if first_sentence_latency is None:
                    first_sentence_latency = time.time() - start_time

                yield {
                    "sentence": sentence,
                    "sentence_index": sentence_index,
                    "first_sentence_latency": first_sentence_latency,
                }
                sentence_index += 1
                emitted_count += 1

        generation_thread.join()

        tail = remainder.strip()
        if tail and len(tail) >= 5:
            if first_sentence_latency is None:
                first_sentence_latency = time.time() - start_time

            yield {
                "sentence": tail,
                "sentence_index": sentence_index,
                "first_sentence_latency": first_sentence_latency,
            }

    def generate_response(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
    ):
        sentences = [
            chunk["sentence"]
            for chunk in self.generate_response_stream(
                user_text,
                voice_emotion,
                voice_confidence,
            )
        ]
        return " ".join(sentences).strip()