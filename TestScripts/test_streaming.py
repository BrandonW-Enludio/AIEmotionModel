import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = (
    "Welcome to the world of streaming text to speech. "
    "This audio should be generated in real time chunks."
)

print("Warming up...")

# Warm-up pass (not measured)
for audio_chunk, metrics in model.generate_stream(text):
    pass

print("Warm-up complete")

# Actual measurement
audio_chunks = []

torch.cuda.synchronize()
start = time.perf_counter()

first_chunk_time = None

for audio_chunk, metrics in model.generate_stream(text):

    if first_chunk_time is None:
        torch.cuda.synchronize()
        first_chunk_time = time.perf_counter() - start
        print(f"First chunk latency: {first_chunk_time:.3f}s")

    print(
        f"Chunk {metrics.chunk_count} | "
        f"RTF: {metrics.rtf:.3f}" if metrics.rtf else
        f"Chunk {metrics.chunk_count}"
    )

    audio_chunks.append(audio_chunk)

torch.cuda.synchronize()
total_time = time.perf_counter() - start

print(f"Total generation time: {total_time:.3f}s")

final_audio = torch.cat(audio_chunks, dim=-1)

ta.save(
    "streaming_test.wav",
    final_audio.cpu(),
    model.sr
)

print("Saved streaming_test.wav")