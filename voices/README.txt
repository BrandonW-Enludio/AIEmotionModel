Put your Chatterbox Turbo reference voice here as:

  reference.wav

Requirements:
  - WAV format
  - At least 5 seconds (6-15s recommended)
  - One clean English speaker
  - No music / heavy noise / reverb

TTSHandler loads this once at startup via prepare_conditionals(),
then reuses it for every generate() call so latency stays low.

Or pass a custom path:
  TTSHandler(voice_prompt_path=r"voices\my_npc.wav")
