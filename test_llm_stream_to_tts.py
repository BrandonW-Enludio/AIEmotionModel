import time

from llm import LLMHandler
from tts import TTSHandler


USER_TEXT = "Put the gun down. Nobody needs to get hurt here."
VOICE_EMOTION = "ang"
VOICE_CONFIDENCE = 0.82


def main():
    llm = LLMHandler()
    tts = TTSHandler()

    print("\n========== STREAMING LLM -> TTS BENCHMARK ==========")
    print(f"User: {USER_TEXT}")
    print(f"Emotion: {VOICE_EMOTION} ({VOICE_CONFIDENCE:.2f})\n")

    pipeline_start = time.perf_counter()
    llm_start = time.perf_counter()
    first_sentence_latency = None
    first_tts_queued = None
    sentence_count = 0

    for chunk in llm.generate_response_stream(
        user_text=USER_TEXT,
        voice_emotion=VOICE_EMOTION,
        voice_confidence=VOICE_CONFIDENCE,
    ):
        sentence = chunk["sentence"]
        sentence_index = chunk["sentence_index"]

        if first_sentence_latency is None:
            first_sentence_latency = chunk["first_sentence_latency"]
            print(
                f"⚡ First sentence ready in "
                f"{first_sentence_latency:.3f}s: {sentence}"
            )
        else:
            print(f"🤖 Sentence {sentence_index + 1}: {sentence}")

        tts.speak_sentence_async(
            sentence,
            emotion=VOICE_EMOTION if sentence_index == 0 else None,
        )
        sentence_count += 1

        if first_tts_queued is None:
            first_tts_queued = time.perf_counter() - pipeline_start

    llm_total = time.perf_counter() - llm_start

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Sentences streamed:     {sentence_count}")
    print(f"LLM first sentence:     {first_sentence_latency:.3f}s")
    print(f"LLM total:              {llm_total:.3f}s")
    print(f"First TTS queued at:    {first_tts_queued:.3f}s")
    print("=========================================")
    print("Listening for TTS playback... (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBenchmark stopped.")


if __name__ == "__main__":
    main()
