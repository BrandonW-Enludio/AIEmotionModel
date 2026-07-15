"""Shared interfaces for swappable pipeline stages."""

from abc import ABC, abstractmethod
import re
import time
from typing import Iterator, Optional


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


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
                text = self.llm.generate_response(user_text, voice_emotion)
            except TypeError:
                text = self.llm.generate_response(user_text, voice_emotion)

        total = time.time() - start
        text = (text or "").strip()
        if not text:
            return

        parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
        if not parts:
            parts = [text]

        for index, sentence in enumerate(parts):
            yield {
                "sentence": sentence,
                "sentence_index": index,
                "first_sentence_latency": total,
                "sentence_latency": total,
                "delta_latency": total if index == 0 else 0.0,
            }
