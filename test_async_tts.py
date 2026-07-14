import time

from tts import TTSHandler


tts = TTSHandler()


time.sleep(1)


sentences = [
    "Welcome back Brandon.",
    "I have finished testing the asynchronous Chatterbox Turbo pipeline.",
    "This should now generate ahead of playback."
]


for sentence in sentences:

    tts.speak_async(
        sentence
    )

    # simulate Qwen streaming delay
    time.sleep(0.2)



while True:
    time.sleep(1)