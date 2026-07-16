import time
import queue
import threading

import torch
import sounddevice as sd

from chatterbox.tts_turbo import ChatterboxTurboTTS


# -----------------------
# Queues
# -----------------------

sentence_queue = queue.Queue()
audio_queue = queue.Queue()


# -----------------------
# Load model
# -----------------------

print("Loading Turbo...")

model = ChatterboxTurboTTS.from_pretrained(
    device="cuda"
)

print("Loaded.")


# -----------------------
# TTS worker
# -----------------------

def tts_worker():

    while True:

        sentence = sentence_queue.get()

        if sentence is None:
            break

        print(
            f"TTS generating: {sentence}"
        )

        start = time.perf_counter()

        audio = model.generate(sentence)

        torch.cuda.synchronize()

        print(
            f"TTS finished in "
            f"{time.perf_counter()-start:.3f}s"
        )

        audio_queue.put(
            audio.cpu()
        )


# -----------------------
# Playback worker
# -----------------------

def playback_worker():

    while True:

        audio = audio_queue.get()

        if audio is None:
            break

        print("Playing")

        sd.play(
            audio.squeeze().numpy(),
            model.sr,
            blocking=True
        )


# -----------------------
# Start threads
# -----------------------

threading.Thread(
    target=tts_worker,
    daemon=True
).start()


threading.Thread(
    target=playback_worker,
    daemon=True
).start()



# -----------------------
# Simulated Qwen output
# -----------------------

sentences = [
    "Welcome back Brandon.",
    "I have finished testing the Chatterbox Turbo sentence buffer.",
    "This should now play without pauses between sentences."
]


for sentence in sentences:

    sentence_queue.put(sentence)

    # simulate LLM delay
    time.sleep(0.2)



# wait for everything

time.sleep(15)