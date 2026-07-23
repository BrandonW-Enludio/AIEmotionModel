"""
OpenAI-compatible chat LLM for the voice pipeline.

Talks to any /v1/chat/completions endpoint (llama.cpp server, Ollama,
vLLM, OpenAI, Groq, OpenRouter, etc.). Scenario prompts and quality
filters match the local HuggingFace path in llm.py.

Env:
  OPENAI_BASE_URL  default http://127.0.0.1:8080/v1
  OPENAI_API_KEY   default sk-local  (ignored by many local servers)
  OPENAI_MODEL     optional — if unset, uses the first id from GET /v1/models
  OPENAI_TIMEOUT_S default 60
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from interfaces import BlockingLLMAdapter, LLMInterface
from scenario import (
    format_negotiator_turn,
    get_scenario,
    llm_emotion_hint,
)


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


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _discover_model(base_url: str, api_key: str, timeout_s: float) -> str:
    """Return the first model id from GET {base_url}/models."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Could not list models at {url} (HTTP {e.code}): {detail[:300]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"OpenAI-compat server unreachable at {url}: {e.reason}. "
            "Start llama-server (or set OPENAI_BASE_URL)."
        ) from e

    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by {url}: {data!r}")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"Model list missing id field: {models[0]!r}")
    return str(model_id)


class OpenAICompatLLMHandler(LLMInterface):
    """
    Scenario-aware blocking LLM over an OpenAI-style HTTP API.
    """

    def __init__(
        self,
        scenario_id=None,
        base_url=None,
        api_key=None,
        model=None,
        timeout_s=None,
        display_name=None,
    ):
        self.base_url = (base_url or _env("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.api_key = api_key or _env("OPENAI_API_KEY", "sk-local")
        self.timeout_s = float(
            timeout_s if timeout_s is not None else _env("OPENAI_TIMEOUT_S", "60")
        )
        self.model = model or _env("OPENAI_MODEL", "")
        if not self.model:
            print("OPENAI_MODEL unset — discovering from /v1/models ...")
            self.model = _discover_model(self.base_url, self.api_key, self.timeout_s)
            print(f"  using model id: {self.model}")

        self.scenario = get_scenario(scenario_id)
        self.system_prompt = self.scenario["system_prompt"]
        self.fallback_reply = self.scenario["fallback_reply"]
        self.repair_instruction = self.scenario["repair_instruction"]
        self.few_shot = list(self.scenario.get("few_shot") or [])

        self.history = []
        self.max_history_turns = 6

        label = display_name or f"openai_compat:{self.model}"
        print(
            f"Loading {label} (OpenAI-compat, scenario-aware)...\n"
            f"  base_url={self.base_url}\n"
            f"  model={self.model}"
        )
        print(
            f"[ok] {label} ready | scenario={self.scenario['id']}"
        )
        self._stream_adapter = BlockingLLMAdapter(
            self,
            name=label.replace("/", "_").replace(" ", "_").replace(":", "_").replace("\\", "_").lower(),
        )

    def clear_history(self):
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
        u = self._normalize(user_text)
        r = self._normalize(response)
        if not r:
            return True
        if r == u:
            return True
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
        lower = (response or "").lower()
        if not lower.strip():
            return True
        if any(marker in lower for marker in META_MARKERS):
            return True
        if lower.count(".") + lower.count("!") + lower.count("?") >= 4 and len(lower) > 220:
            planning_hits = sum(
                1 for w in ("should", "need to", "make sure", "craft", "respond", "tone")
                if w in lower
            )
            if planning_hits >= 2:
                return True
        return False

    def _is_too_thin(self, response: str) -> bool:
        words = self._normalize(response).split()
        if len(words) < 8:
            return True
        if len(words) <= 2 and words[0] in {
            "exit", "no", "yes", "gun", "stop", "out", "leave", "now", "okay", "ok",
        }:
            return True
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
        tension_snapshot=None,
    ):
        hint = llm_emotion_hint(
            self.scenario,
            voice_emotion,
            voice_confidence,
        )
        tension_block = ""
        if tension_snapshot is not None:
            tension_block = tension_snapshot.prompt_block()
        user_content = format_negotiator_turn(
            user_text,
            hint,
            tension_block=tension_block,
        )
        if repair:
            user_content = f"{self.repair_instruction}\n\n{user_content}"

        return [
            {"role": "system", "content": self.system_prompt},
            *self.few_shot,
            *self.history,
            {"role": "user", "content": user_content},
        ]

    def _chat_completions(self, messages) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.65,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compat HTTP {e.code} from {url}: {detail[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI-compat unreachable at {url}: {e.reason}"
            ) from e

        elapsed = time.time() - t0
        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected OpenAI-compat response shape: {raw[:500]}"
            ) from e

        if isinstance(content, list):
            # Some providers return content parts.
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        response = (content or "").strip()
        thinking_end = "</" + "think>"
        response = response.split(thinking_end)[-1].strip()
        if len(response) >= 2 and response[0] == response[-1] and response[0] in "\"'":
            response = response[1:-1].strip()

        print(f"⏱️ OpenAI-compat generate: {elapsed:.2f}s")
        return response

    def _generate_once(self, messages) -> str:
        return self._chat_completions(messages)

    def generate_response(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
        tension_snapshot=None,
    ):
        messages = self._build_messages(
            user_text,
            voice_emotion,
            voice_confidence,
            tension_snapshot=tension_snapshot,
        )
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
                tension_snapshot=tension_snapshot,
            )
            response = self._generate_once(messages)
            print(f"🤖 LLM retry reply: {response!r}")

            reason = self._reject_reason(user_text, response)
            if reason:
                print(f"⚠️ Retry still bad ({reason}). Using fallback.")
                response = self.fallback_reply

        if response:
            hint = llm_emotion_hint(
                self.scenario,
                voice_emotion,
                voice_confidence,
            )
            tension_block = ""
            if tension_snapshot is not None:
                tension_block = tension_snapshot.prompt_block()
            self.history.append({
                "role": "user",
                "content": format_negotiator_turn(
                    user_text,
                    hint,
                    tension_block=tension_block,
                ),
            })
            self.history.append({"role": "assistant", "content": response})
            self._trim_history()

        return response

    def generate_response_stream(
        self,
        user_text: str,
        voice_emotion="neu",
        voice_confidence=0.0,
        tension_snapshot=None,
    ):
        yield from self._stream_adapter.generate_response_stream(
            user_text,
            voice_emotion=voice_emotion,
            voice_confidence=voice_confidence,
            tension_snapshot=tension_snapshot,
        )


# Alias so pipeline loaders can use a familiar name.
LLMHandler = OpenAICompatLLMHandler
