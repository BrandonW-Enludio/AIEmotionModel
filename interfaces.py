"""Shared interfaces for swappable pipeline stages."""

from abc import ABC, abstractmethod
import re
import time
from typing import Iterator, Optional


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# Keep TTS sentence-1 short so Response Start stays in budget (gen time ∝ text).
FIRST_SENTENCE_MAX_WORDS = 10

# Prefer natural clause breaks when forcing a short opener.
_CLAUSE_BREAK = re.compile(
    r"(?:,\s+|;|\s+[—–-]\s+|\s+(?:then|but|and|so)\s+)",
    re.IGNORECASE,
)


def split_reply_for_tts(
    text: str,
    max_first_words: int = FIRST_SENTENCE_MAX_WORDS,
):
    """
    Split a full reply into spoken chunks for TTS.

    If the first sentence is long, cut it at a clause boundary (or a hard
    word cap) so Chatterbox can start audio on a short opener.
    History still stores the original full string from the LLM.
    """
    text = (text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    if not parts:
        return [text]

    first, *rest = parts
    if len(first.split()) <= max_first_words:
        return parts

    opener, leftover = _force_short_opener(first, max_first_words)
    out = [opener]
    if leftover:
        out.append(leftover)
    out.extend(rest)
    return out


def _force_short_opener(sentence: str, max_words: int):
    """Return (short_opener, remainder). Remainder may be empty."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence, ""

    best_end = None
    for match in _CLAUSE_BREAK.finditer(sentence):
        prefix = sentence[: match.start()].strip()
        if not prefix:
            continue
        if len(prefix.split()) <= max_words:
            best_end = match.end()
        else:
            break

    if best_end is not None:
        opener = sentence[:best_end].strip().rstrip(",;:—–-").strip()
        leftover = sentence[best_end:].strip()
        if opener and leftover:
            if leftover[0].islower():
                leftover = leftover[0].upper() + leftover[1:]
            if opener[-1] not in ".!?":
                opener = opener + "."
            return opener, leftover

    opener = " ".join(words[:max_words]).rstrip(",;:—–-")
    leftover = " ".join(words[max_words:]).strip()
    if leftover and leftover[0].islower():
        leftover = leftover[0].upper() + leftover[1:]
    if opener and opener[-1] not in ".!?":
        opener = opener + "."
    return opener, leftover


class STTInterface(ABC):
    @abstractmethod
    def transcribe(self, audio_data, sample_rate: int = 16000) -> str:
        """Return transcript text for the utterance."""


class SERInterface(ABC):
    @abstractmethod
    def detect(self, audio_data, sample_rate: int = 16000) -> dict:
        """
        Return:
            emotion: str
            confidence: float
            latency: float
        """


class LLMInterface(ABC):
    @abstractmethod
    def generate_response_stream(
        self,
        user_text: str,
        voice_emotion: str = "neu",
        voice_confidence: float = 0.0,
    ) -> Iterator[dict]:
        """
        Yield sentence chunks:
            sentence, sentence_index, first_sentence_latency,
            sentence_latency, delta_latency
        """


class TTSInterface(ABC):
    @abstractmethod
    def begin_turn(self, turn_id: int, speech_end_time: float) -> None:
        ...

    @abstractmethod
    def speak_sentence_async(
        self,
        text: str,
        emotion: Optional[str] = None,
        turn_id: Optional[int] = None,
        sentence_index: int = 0,
    ) -> None:
        ...

    @abstractmethod
    def close_turn(self, turn_id: int, expected_sentences: int) -> None:
        ...


class BlockingLLMAdapter(LLMInterface):
    """
    Wrap a legacy LLM that only implements generate_response(...).

    Stream lead will be ~0 because the full reply is produced before any
    sentence is yielded — useful as a baseline vs true streaming models.
    """

    def __init__(self, blocking_llm, name: str = "blocking_llm"):
        self.llm = blocking_llm
        self.name = name

    def generate_response_stream(
        self,
        user_text: str,
        voice_emotion: str = "neu",
        voice_confidence: float = 0.0,
    ):
        start = time.time()

        # Support both current and older AlternateModels signatures.
        try:
            text = self.llm.generate_response(
                user_text,
                voice_emotion=voice_emotion,
                voice_confidence=voice_confidence,
            )
        except TypeError:
            try:
                text = self.llm.generate_response(
                    user_text,
                    emotion=voice_emotion,
                )
            except TypeError:
                text = self.llm.generate_response(user_text, voice_emotion)

        total = time.time() - start
        text = (text or "").strip()
        if not text:
            return

        parts = split_reply_for_tts(text)
        if not parts:
            return

        for index, sentence in enumerate(parts):
            yield {
                "sentence": sentence,
                "sentence_index": index,
                "first_sentence_latency": total,
                "sentence_latency": total,
                "delta_latency": total if index == 0 else 0.0,
            }
