import torch
import sounddevice as sd

from chatterbox.tts import ChatterboxTTS


def crossfade_pair(previous, current, overlap_samples):

    fade_out = torch.linspace(
        1,
        0,
        overlap_samples
    )

    fade_in = torch.linspace(
        0,
        1,
        overlap_samples
    )


    blended = (
        previous[-overlap_samples:] * fade_out
        +
        current[:overlap_samples] * fade_in
    )


    return torch.cat(
        [
            previous[:-overlap_samples],
            blended,
            current[overlap_samples:]
        ]
    )


model = ChatterboxTTS.from_pretrained(device="cuda")


text = (
    "Welcome to the world of streaming text to speech. "
    "This should now play more smoothly."
)


overlap_samples = int(model.sr * 0.05)


with sd.OutputStream(
    samplerate=model.sr,
    channels=1,
    dtype="float32"
) as stream:


    previous_chunk = None


    for audio_chunk, metrics in model.generate_stream(text):

        print(
            f"Playing chunk {metrics.chunk_count}"
        )


        chunk = audio_chunk.squeeze().cpu()


        if previous_chunk is None:

            previous_chunk = chunk

            continue


        output = crossfade_pair(
            previous_chunk,
            chunk,
            overlap_samples
        )


        stream.write(
            output.numpy()
        )


        previous_chunk = chunk


    # Play final chunk
    if previous_chunk is not None:
        stream.write(
            previous_chunk.numpy()
        )