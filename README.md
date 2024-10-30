# Local Voice Assistant

### Description of the project

This projects aims at building a speech-to-text application in order to communicate with an avatar of the french compositor Gabriel Fauré. The constraints concerning the app are the following
- it must run fully in local
- it must be lighweight and minimize latency
- it should recognize efficiently vocabulary linked to the world of Fauré, that is: transcribe correctly the names, the music-related vocabulary etc

### Setup

Create a virtual environment with Python 3.11.2

Install Whipser from OpenAI:
`pip3 install -U openai-whisper`

Install `ffmpeg` via Homebrew:
`brew install ffmpeg`

### Description of the applciation

`main.py` combines a speech-to-text application and a LLM: it transcribes text from user's microphone input and gives the text as input to the LLM. Then, the answer of the LLM is printed as output. There is no text-to-speech functionality implemented yet.

### Problems

The speech-to-text application fails to recognize some music-related specific French words such as "élégie" or "requiem". The project is to build an application that is able to recognize these words (and other related) without increasing model complexity and latency.

The first thing to investigate is whether or not it is poissible to "finetune" whisper, more precisely to add new term to his vocabulary to make them more easily recognisable.

The model currently has no memory and a very limited knwoledge. It may hallucinate if we ask question that are not clearly in his knowledge.

- Lack of knowledge
- High risk for hallucinations
- No memory

### Directions of work

**Speech-to-Text component**
- Better recognition of specific vocabulary

**LLM component**
- More prompt engineering (system prompt, prompt template, etc)
- Retrieval Augmented Generation (RAG) (to add external knowledge)
- Finetuning

### Ongoing work

I choose to work on the speech-to-text component. The first step is to make a review of the litterature aiming to answer the following question
- Is it possible to "finetune" an existing model (such as Whisper) to improve recognition of specific words
- How to build my own little stt specialized in Gabriel Fauré, specialized in keywords, and make it work in conjonction with Whisper.

The last point is the most interesting: build a model that can recognise specific keywords and use it in conjunction with the main model, ie Whisper.


### References
- Whisper repository
- voice assistant repo




