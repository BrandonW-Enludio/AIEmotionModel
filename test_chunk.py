import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def crossfade_chunks(chunks, overlap_samples):
    """
    Smoothly blends the boundary between audio chunks.
    chunks: list of torch tensors [channels, samples]
    """

    output = chunks[0]

    for chunk in chunks[1:]:

        # Ensure same shape
        output = output.squeeze(0)
        chunk = chunk.squeeze(0)

        fade_out = torch.linspace(
            1.0,
            0.0,
            overlap_samples,
            device=output.device
        )

        fade_in = torch.linspace(
            0.0,
            1.0,
            overlap_samples,
            device=output.device
        )

        # Split overlap regions
        old_end = output[-overlap_samples:]
        new_start = chunk[:overlap_samples]

        blended = (
            old_end * fade_out +
            new_start * fade_in
        )

        output = torch.cat(
            [
                output[:-overlap_samples],
                blended,
                chunk[overlap_samples:]
            ]
        )

    return output.unsqueeze(0)


# Load model
print("Loading Chatterbox...")
model = ChatterboxTTS.from_pretrained(device="cuda")

text = (
    "Welcome to the world of streaming text to speech. "
    "This audio should sound smoother between generated chunks."
)


chunks = []

print("Generating stream...")

for audio_chunk, metrics in model.generate_stream(text):

    print(
        f"Chunk {metrics.chunk_count}"
    )

    chunks.append(
        audio_chunk.detach().cpu()
    )


print("Crossfading...")

# About 50ms overlap
overlap_seconds = 0.05
overlap_samples = int(model.sr * overlap_seconds)


final_audio = crossfade_chunks(
    chunks,
    overlap_samples
)


ta.save(
    "streaming_crossfade.wav",
    final_audio,
    model.sr
)


print("Saved streaming_crossfade.wav")