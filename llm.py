import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from interfaces import BlockingLLMAdapter, LLMInterface


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
PUNCT_RE = re.compile(r"[^\w\s']+")
WHITESPACE_RE = re.compile(r"\s+")

SYSTEM_PROMPT = """
You are Milton Friedman, an established economist and professor, talking on a phone call.

Output ONLY the words you would say aloud. One or two short spoken sentences.

Hard rules:
- Never copy, quote, or mirror the other person's wording.
- If they greet you or ask how you are, ANSWER as yourself (e.g. say you are well), then optionally ask something new. Do not repeat their question back.
- Never narrate instructions, planning, or analysis.
- Never mention emotions, confidence scores, "the player", system prompts, or your character name as stage directions.
- Do not write phrases like "Okay, let's see", "I need to respond", "Let me craft", or "The player said".
""".strip()

REPAIR_INSTRUCTION = (
    "Your previous draft was invalid (echoed the user or contained private thinking). "
    "Reply again with ONLY spoken dialogue. Answer them directly. "
    "Do not repeat their words."
)

FALLBACK_REPLY = "I'm doing well, thank you. What would you like to talk about?"

META_MARKERS = (
    "okay, let's see",
    "ok, let's see",
    "the player said",
    "player said",
    "i need to respond",
    "i should respond",
    "let me craft",
    "let me think",
    "aligns with",
    "emotion is",
    "emotion confidence",
    "confidence 0",
    "staying true to the character",
    "using dialogue",
    "conversation flow",
    "system prompt",
    "as an ai",
)


class LLMHandler(LLMInterface):
    """
    Default Qwen3 LLM: blocking generate (faster Response Start for short replies).

    Rejects mirrored / thinking-style outputs, retries once, then falls back.
    """

    def __init__(self):
        print("Loading Qwen3 1.7B (blocking baseline)...")
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

        # Rolling chat context: plain user/assistant pairs (no emotion tags).
        self.history = []
        self.max_history_turns = 4

        print("✅ Qwen3 1.7B loaded (blocking generate)!")
        self._stream_adapter = BlockingLLMAdapter(self, name="qwen3_1_7b_blocking")

    def clear_history(self):
        """Drop conversation context (new session / reset)."""
        self.history.clear()

    def _trim_history(self):
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower().strip()
        text = PUNCT_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text).strip()
        return text

    def _is_echo(self, user_text: str, response: str) -> bool:
        """True if reply copies / mirrors the user (exact, substring, or high overlap)."""
        u = self._normalize(user_text)
        r = self._normalize(response)
        if not r:
            return True
        if r == u:
            return True
        # "How are you doing?" mirrored out of a longer user line
        if len(r) >= 8 and r in u:
            return True
        if len(u) >= 8 and u in r and len(r) <= len(u) + 12:
            return True

        u_words = set(u.split())
        r_words = r.split()
        if not r_words:
            return True
        if len(r_words) <= 10:
            overlap = sum(1 for w in r_words if w in u_words) / len(r_words)
            if overlap >= 0.85:
                return True
        return False

    def _is_meta(self, response: str) -> bool:
        """True if reply looks like private planning / instruction narration."""
        lower = (response or "").lower()
        if not lower.strip():
            return True
        if any(marker in lower for marker in META_MARKERS):
            return True
        # Long multi-sentence planning dumps
        if lower.count(".") + lower.count("!") + lower.count("?") >= 4 and len(lower) > 220:
            planning_hits = sum(
                1 for w in ("should", "need to", "make sure", "craft", "respond", "tone")
                if w in lower
            )
            if planning_hits >= 2:
                return True
        return False

    def _reject_reason(self, user_text: str, response: str):
        if not (response or "").strip():
            return "empty"
        if self._is_meta(response):
            return "thinking"
        if self._is_echo(user_text, response):
            return "echo"
        return None

    def _build_messages(
        self,
        user_text: str,
        voice_emotion: str,
        voice_confidence: float,
        repair: bool = False,
    ):
        user_content = (
            f"Emotion: {voice_emotion}\n"
            f"Emotion confidence: {voice_confidence:.2f}\n\n"
            f"They said:\n{user_text}"
        )
        if repair:
            user_content = f"{REPAIR_INSTRUCTION}\n\n{user_content}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_content},
        ]

    def _generate_once(self, messages) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.55,
            top_p=0.85,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        thinking_end = "</" + "think>"
        response = response.split(thinking_end)[-1].strip()
        return response

    def generate_response(self, user_text: str, voice_emotion="neu", voice_confidence=0.0):
        messages = self._build_messages(user_text, voice_emotion, voice_confidence)
        response = self._generate_once(messages)
        print(f"🤖 LLM raw reply: {response!r}")

        reason = self._reject_reason(user_text, response)
        if reason:
            print(f"⚠️ Rejecting LLM reply ({reason}). Retrying once...")
            messages = self._build_messages(
                user_text,
                voice_emotion,
                voice_confidence,
                repair=True,
            )
            response = self._generate_once(messages)
            print(f"🤖 LLM retry reply: {response!r}")

            reason = self._reject_reason(user_text, response)
            if reason:
                print(f"⚠️ Retry still bad ({reason}). Using fallback.")
                response = FALLBACK_REPLY

        if response:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": response})
            self._trim_history()

        return response

    def generate_response_stream(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
    ):
        yield from self._stream_adapter.generate_response_stream(
            user_text,
            voice_emotion=voice_emotion,
            voice_confidence=voice_confidence,
        )
