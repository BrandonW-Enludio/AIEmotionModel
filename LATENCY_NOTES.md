# Latency notes

## LLM streaming vs blocking (2026-07-15)

A/B on the same pipeline (VAD → STT → SER → LLM → async TTS), Qwen3-1.7B + Chatterbox-Turbo.

| Mode | Typical Response Start (speech end → first audio) |
|------|-----------------------------------------------------|
| Blocking LLM (full reply, then TTS) | ~1.3–1.8s |
| Streaming LLM (sentence handoff) | ~1.8–2.1s |

Streaming added about **~0.5s** to Response Start for short 1–2 sentence NPC replies.

### Why (likely)

- Short replies → little “stream lead”; first sentence ≈ full reply.
- Streamer/thread overhead vs a single `generate()`.
- Same-GPU contention: TTS starting while LLM still generates can slow first TTS audio.

### Takeaway

For short de-escalation dialogue, **prefer blocking LLM**. Streaming remains useful to re-test for long replies or if LLM/TTS are split across devices.

Streaming implementation reference branch: `llm-streaming-tokens`.
