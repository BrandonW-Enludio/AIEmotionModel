import sounddevice as sd
from ser import SERHandler


ser = SERHandler()

print("Speak for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.squeeze()

result = ser.detect(audio)

print(result)