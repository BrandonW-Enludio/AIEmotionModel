from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

print(type(model))
print(hasattr(model, "generate_stream"))