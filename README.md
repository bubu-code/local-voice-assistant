# Local Voice Assistant

### Setup

Create a virtual environment with Python 3.11.2

Install Whipser from OpenAI:
`pip3 install -U openai-whisper`

Install `ffmpeg` via Homebrew:
`brew install ffmpeg`

### Applications

`stt_app.py`is a speech-to-text application: it transcribes text from user's microphone input.
`stt_llm_app`extends the previous application by adding a LLM: the transcribed text is fed into the LLM which give a text answer.

### Problems

The speech-to-text application fails to recognize some music-related specific French words such as "élégie" or "requiem". The project is to build an application that is able to recognize these words (and other related) without increasing model complexity and latency.

The first thing to investigate is whether or not it is poissible to "finetune" whisper, more precisely to add new term to his vocabulary to make them more easily recognisable.


