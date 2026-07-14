import time
import torch
import torchaudio as ta
import sounddevice as sd
from chatterbox.tts_turbo import ChatterboxTurboTTS


# -------------------------
# Sentence buffer
# -------------------------

class SentenceBuffer:
    def __init__(self):
        self.buffer = ""

    def add_text(self, text):
        """
        Add new incoming text.
        Returns completed sentences.
        """

        self.buffer += text

        sentences = []

        while True:
            split_index = -1

            for char in [".", "!", "?"]:
                index = self.buffer.find(char)

                if index != -1:
                    if split_index == -1 or index < split_index:
                        split_index = index

            if split_index == -1:
                break

            sentence = self.buffer[:split_index + 1].strip()

            if sentence:
                sentences.append(sentence)

            self.buffer = self.buffer[split_index + 1:].lstrip()

        return sentences

    def flush(self):
        """
        Return remaining text.
        """
        if self.buffer.strip():
            remaining = self.buffer.strip()
            self.buffer = ""
            return remaining

        return None


# -------------------------
# Load Turbo
# -------------------------

print("Loading Chatterbox Turbo...")

model = ChatterboxTurboTTS.from_pretrained(
    device="cuda"
)

print("Model loaded.")


# -------------------------
# Fake streamed LLM output
# -------------------------

tokens = [
    "Welcome ",
    "back ",
    "Brandon. ",
    "I ",
    "have ",
    "finished ",
    "testing ",
    "the ",
    "Chatterbox ",
    "Turbo ",
    "sentence ",
    "buffer. ",
    "This ",
    "should ",
    "generate ",
    "audio ",
    "while ",
    "the ",
    "LLM ",
    "continues ",
    "thinking!"
]


buffer = SentenceBuffer()

audio_segments = []


# -------------------------
# Generate as sentences arrive
# -------------------------

for token in tokens:

    sentences = buffer.add_text(token)

    for sentence in sentences:

        print("\nGenerating:")
        print(sentence)

        start = time.perf_counter()

        audio = model.generate(sentence)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        print(
            f"Generation: {elapsed:.3f}s"
        )


        print("Playing...")

        sd.play(
            audio.squeeze().cpu().numpy(),
            model.sr,
            blocking=True
        )


# Flush leftover text

remaining = buffer.flush()

if remaining:

    print("\nGenerating final:")
    print(remaining)

    audio = model.generate(remaining)

    #audio_segments.append(
    #    audio.cpu()
    #)
    print("Playing...")

    audio_np = (
        audio
        .squeeze()
        .cpu()
        .numpy()
    )

    sd.play(
        audio_np,
        samplerate=model.sr,
        blocking=True
    )

# -------------------------
# Combine output
# -------------------------

#final_audio = torch.cat(
#    audio_segments,
#    dim=-1
#)


#ta.save(
#    "turbo_sentence_buffer.wav",
#    final_audio,
#    model.sr
#)


#print("\nSaved turbo_sentence_buffer.wav")
print("Generation Completed")