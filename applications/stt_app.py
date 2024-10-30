import warnings
from argparse import ArgumentParser
import torch
from rich.console import Console
import whisper
import time
from queue import Queue
import threading
import sounddevice as sd
import numpy as np

# Ignore warnings
warnings.simplefilter("ignore", FutureWarning)

# Build an argument parser object
parser = ArgumentParser()
parser.add_argument("--model_size", default="base", choices=("tiny", "base", "small"),
                        help="Size of the Whisper TTS model ('base' by default).")
args = parser.parse_args()

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load speech-to-text model
stt_model = whisper.load_model(args.model_size, device=device)

# Build a console object
console = Console()

# Define a helper function to record audio from user's microphone
def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.
    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


# Speech-to-text application
if __name__ == "__main__":
    console.print("[cyan]À votre écoute! Appuyez sur Ctrl+C pour quitter l'application.")
    
    try:
        while True:
            console.input(
                "Appuyez sur Entrée pour démarrer l'enregistrement, puis appuyez à nouveau sur Entrée pour y mettre fin."
            )

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue)
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            # Store audio data as a numpy vector
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                # Transcribe speech
                with console.status("Transcription en cours...", spinner="earth"): 
                    result = stt_model.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
                # Extract text from the result
                text = result["text"].strip() 
                # Print transcription   
                console.print(f"[yellow]Vous : {text}")
            else:
                console.print(
                    "[red]Aucun audio enregistré. Pouvez-vous vous assurer du bon fonctionnement de votre microphone ?."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Fin de la session...")

    console.print("[blue]Session terminée. Au plaisir de vous revoir !")


