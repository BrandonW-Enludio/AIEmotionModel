Default Chatterbox Turbo voice clip (tracked in git):

  reference.wav

Requirements for replacements:
  - WAV format
  - At least 5 seconds (6-15s recommended)
  - One clean English speaker
  - No music / heavy noise / reverb

TTSHandler loads this once at startup via prepare_conditionals(),
then reuses it for every generate() call so latency stays low.

Or pass a custom path:
  TTSHandler(voice_prompt_path=r"voices\my_npc.wav")
