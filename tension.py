"""
SER-driven negotiation tension (no transcript intent / META keyword bags).

The LLM still hears the negotiator's words and decides how to answer.
This module only tracks a coarse tension level from voice emotion and
exposes prompt stance + ending flags for the scenario loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from scenario import normalize_emotion_label


DEFAULT_SER_DELTAS = {
    "ang": 1,
    "hap": 0,  # ambiguous; hold
    "neu": -1,
    "sad": -1,
    "fea": 0,
    "fear": 0,
}


@dataclass
class TensionSnapshot:
    level: int
    max_level: int
    phase: str
    stance: str
    may_bargain: bool
    emotion: str
    confidence: float
    delta: int
    low_streak: int
    high_streak: int
    outcome: Optional[str] = None  # None | "success" | "failed"

    def prompt_block(self) -> str:
        lines = [
            f"Tension level: {self.level}/{self.max_level} (phase: {self.phase}).",
            f"Stance: {self.stance}",
        ]
        if self.may_bargain:
            lines.append(
                "BARGAINING UNLOCKED: You may negotiate concretely "
                "(conditions, proof, a path to stand down). "
                "You may move toward an exit deal if they sound genuine. "
                "Stay in character — still cautious, not a pushover."
            )
        else:
            lines.append(
                "Tension is still too high to resolve: do not free the hostage "
                "or drop the weapon. Test them, demand proof, or hold position."
            )
        return "\n".join(lines)


@dataclass
class TensionMachine:
    """Simple level SM: SER nudges tension up/down; endings from streaks."""

    max_level: int = 4
    level: int = 3
    bargain_at_or_below: int = 1
    success_at_or_below: int = 1
    success_streak_needed: int = 2
    fail_at_or_above: int = 4
    fail_streak_needed: int = 2
    confidence_floor: float = 0.45
    ser_deltas: dict = field(default_factory=lambda: dict(DEFAULT_SER_DELTAS))
    bands: dict = field(default_factory=dict)
    low_streak: int = 0
    high_streak: int = 0
    outcome: Optional[str] = None
    turn_count: int = 0

    @classmethod
    def from_scenario(cls, scenario: dict) -> "TensionMachine":
        cfg = scenario.get("tension") or {}
        bands = cfg.get("bands") or _default_bands(int(cfg.get("max_level", 4)))
        norm_bands = {int(k): v for k, v in bands.items()}
        return cls(
            max_level=int(cfg.get("max_level", 4)),
            level=int(cfg.get("start_level", 3)),
            bargain_at_or_below=int(cfg.get("bargain_at_or_below", 1)),
            success_at_or_below=int(cfg.get("success_at_or_below", 1)),
            success_streak_needed=int(cfg.get("success_streak_needed", 2)),
            fail_at_or_above=int(cfg.get("fail_at_or_above", 4)),
            fail_streak_needed=int(cfg.get("fail_streak_needed", 2)),
            confidence_floor=float(
                cfg.get(
                    "confidence_floor",
                    scenario.get("emotion_confidence_floor", 0.45),
                )
            ),
            ser_deltas={
                **DEFAULT_SER_DELTAS,
                **(cfg.get("ser_deltas") or {}),
            },
            bands=norm_bands,
        )

    def _band(self, level: int) -> dict:
        if level in self.bands:
            return self.bands[level]
        return {
            "phase": "unknown",
            "stance": "Stay wary and firm.",
            "may_bargain": level <= self.bargain_at_or_below,
        }

    def update(
        self,
        voice_emotion: str,
        voice_confidence: float,
    ) -> TensionSnapshot:
        """Apply one SER observation. Ending sticks once set."""
        self.turn_count += 1
        key = normalize_emotion_label(voice_emotion)
        delta = 0
        if voice_confidence >= self.confidence_floor:
            delta = int(self.ser_deltas.get(key, 0))

        if self.outcome is None:
            self.level = max(0, min(self.max_level, self.level + delta))

            if self.level <= self.success_at_or_below:
                self.low_streak += 1
            else:
                self.low_streak = 0

            if self.level >= self.fail_at_or_above:
                self.high_streak += 1
            else:
                self.high_streak = 0

            if self.low_streak >= self.success_streak_needed:
                self.outcome = "success"
            elif self.high_streak >= self.fail_streak_needed:
                self.outcome = "failed"

        band = self._band(self.level)
        return TensionSnapshot(
            level=self.level,
            max_level=self.max_level,
            phase=str(band.get("phase", "unknown")),
            stance=str(band.get("stance", "Stay wary and firm.")),
            may_bargain=bool(
                band.get("may_bargain", self.level <= self.bargain_at_or_below)
            ),
            emotion=key,
            confidence=float(voice_confidence),
            delta=delta,
            low_streak=self.low_streak,
            high_streak=self.high_streak,
            outcome=self.outcome,
        )


def _default_bands(max_level: int) -> dict[int, dict[str, Any]]:
    return {
        4: {
            "phase": "crisis",
            "stance": "Highly agitated — clipped, distrustful, ready to dig in.",
            "may_bargain": False,
        },
        3: {
            "phase": "hostile",
            "stance": "Wary and firm — test them; do not give ground easily.",
            "may_bargain": False,
        },
        2: {
            "phase": "guarded",
            "stance": "Still armed and suspicious, but listening harder.",
            "may_bargain": False,
        },
        1: {
            "phase": "bargaining",
            "stance": "Tension is down — you can name conditions and bargain.",
            "may_bargain": True,
        },
        0: {
            "phase": "resolution",
            "stance": (
                "Lowest tension — if they keep offering a clear safe path, "
                "you may accept a concrete exit deal in character."
            ),
            "may_bargain": True,
        },
    }
