import torchaudio as ta
import torch
import sounddevice as sd
import re
import queue
import threading
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Welcome to the world of streaming text-to-speech! This audio will be generated and played in real-time chunks. " \
       "Each sentence will now start and end cleanly. This feels much more natural, doesn't it? " \
       "We can add more sentences here for testing. " \
       "The playback should now flow smoothly between sentences."

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

sentences = split_into_sentences(text)

# Queue for audio chunks (sentence-level)
audio_queue = queue.Queue(maxsize=3)  # Small buffer for smoother flow
stop_event = threading.Event()

def playback_worker():
    """Background thread that plays audio from the queue"""
    while not stop_event.is_set():
        try:
            audio_np = audio_queue.get(timeout=0.5)
            if audio_np is None:  # Sentinel value to stop
                break
            sd.play(audio_np, samplerate=model.sr)
            sd.wait()  # Wait for this sentence to finish playing
            audio_queue.task_done()
        except queue.Empty:
            continue
    print("Playback thread stopped.")

# Start playback thread
playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

print("Starting smooth sentence-aligned TTS with background playback...\n")

audio_chunks = []  # Optional: for final save

try:
    for i, sentence in enumerate(sentences):
        print(f"Generating sentence {i+1}/{len(sentences)}: {sentence}")
        
        # Generate full sentence (clean boundaries)
        audio = model.generate(sentence)
        audio_np = audio.squeeze().cpu().numpy()
        
        # Put in queue for playback thread (non-blocking for generation)
        audio_queue.put(audio_np)
        audio_chunks.append(audio)
        
        print(f"Queued sentence {i+1} for playback\n")

    # Wait for queue to empty
    audio_queue.join()

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    # Stop playback thread cleanly
    audio_queue.put(None)  # Sentinel
    stop_event.set()
    playback_thread.join(timeout=2.0)

# Optional: Save complete audio
if audio_chunks:
    final_audio = torch.cat(audio_chunks, dim=-1)
    ta.save("smooth_sentence_output.wav", final_audio, model.sr)
    print("Full audio saved to smooth_sentence_output.wav")