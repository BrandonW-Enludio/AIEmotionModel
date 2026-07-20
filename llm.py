import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from interfaces import BlockingLLMAdapter, LLMInterface
from scenario import (
    format_negotiator_turn,
    get_scenario,
    llm_emotion_hint,
)


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
PUNCT_RE = re.compile(r"[^\w\s']+")
WHITESPACE_RE = re.compile(r"\s+")

META_MARKERS = (
    "okay, let's see",
    "ok, let's see",
    "the player said",
    "player said",
    "negotiator said",
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
    "as an npc",
    "hostage situation",
)


class LLMHandler(LLMInterface):
    """
    Scenario-aware blocking LLM (Qwen family by default).

    Prompt / fallback / few-shots come from scenario.py.
    Rejects mirrored / thinking-style / too-thin outputs, retries once, then falls back.
    """

    def __init__(
        self,
        scenario_id=None,
        model_name="Qwen/Qwen3-1.7B",
        display_name=None,
    ):
        label = display_name or model_name
        print(f"Loading {label} (blocking, scenario-aware)...")

        self.model_name = model_name
        self.scenario = get_scenario(scenario_id)
        self.system_prompt = self.scenario["system_prompt"]
        self.fallback_reply = self.scenario["fallback_reply"]
        self.repair_instruction = self.scenario["repair_instruction"]

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

        # Rolling chat context (negotiator / NPC spoken turns).
        self.history = []
        self.max_history_turns = 6
        self.few_shot = list(self.scenario.get("few_shot") or [])

        print(
            f"✅ {label} loaded | "
            f"scenario={self.scenario['id']}"
        )
        self._stream_adapter = BlockingLLMAdapter(
            self,
            name=label.replace("/", "_").replace(" ", "_").lower(),
        )

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

    def _is_too_thin(self, response: str) -> bool:
        """True if reply is a slogan / one-word bark, not real dialogue."""
        words = self._normalize(response).split()
        if len(words) < 8:
            return True
        if len(words) <= 2 and words[0] in {
            "exit", "no", "yes", "gun", "stop", "out", "leave", "now", "okay", "ok",
        }:
            return True
        # Stock lines that ignore almost any negotiator content
        stock = {
            "stay put no distractions",
            "stay where you are",
            "nobody comes through that door",
        }
        if self._normalize(response) in stock:
            return True
        return False

    def _reject_reason(self, user_text: str, response: str):
        if not (response or "").strip():
            return "empty"
        if self._is_meta(response):
            return "thinking"
        if self._is_too_thin(response):
            return "too_thin"
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
        hint = llm_emotion_hint(
            self.scenario,
            voice_emotion,
            voice_confidence,
        )
        user_content = format_negotiator_turn(user_text, hint)
        if repair:
            user_content = f"{self.repair_instruction}\n\n{user_content}"

        return [
            {"role": "system", "content": self.system_prompt},
            *self.few_shot,
            *self.history,
            {"role": "user", "content": user_content},
        ]

    def _generate_once(self, messages) -> str:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3 supports enable_thinking; Qwen2.5 chat template does not.
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.65,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            min_new_tokens=12,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        thinking_end = "</" + "think>"
        response = response.split(thinking_end)[-1].strip()
        if len(response) >= 2 and response[0] == response[-1] and response[0] in "\"'":
            response = response[1:-1].strip()
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
                response = self.fallback_reply

        if response:
            # Keep history in the same grounded format as live turns.
            hint = llm_emotion_hint(
                self.scenario,
                voice_emotion,
                voice_confidence,
            )
            self.history.append({
                "role": "user",
                "content": format_negotiator_turn(user_text, hint),
            })
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
