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
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Ignore warnings
warnings.simplefilter("ignore", FutureWarning)

# Build an argument parser object
parser = ArgumentParser()
# Speech-to-text arguments
parser.add_argument('--stt_name', type=str, default='small', choices=('tiny', 'base', 'small'),
                        help="Size of the TTS model ('base' by default).")
# Language model arguments
parser.add_argument('--l_name', type=str, default='llama3.2', choices=('gemma:2b', 'llama3.2'),
                    help="Language model ('llama3.2' by default).")
parser.add_argument('--temperature', type=float, default=0.0, help="Temperature parameter for the LLM (0.0 by default).")
# Parse arguments
args = parser.parse_args()
stt_name = args.stt_name
l_name = args.l_name
temperature = args.temperature

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load speech-to-text model
stt_model = whisper.load_model(name=stt_name, device=device)

# Load language model
llm = OllamaLLM(
    model=l_name,
    temperature=temperature
)

string_template = """Vous êtes le célèbre compositeur français Garbriel Fauré, né le 12 mai 1845. 
Votre rôle est de répondre poliment et de manière concise à la qestion {text} en utilisant les informations délimitées par les balises <context> et </context>.
<context>
Années 1845-1870 
Origines et famille :
- Naissance et famille : Gabriel Fauré est né le 12 mai 1845 à Pamiers, une petite ville de l'Ariège. Bien que sa famille soit bien enracinée dans la région méridionale, ses œuvres ne reflètent pas explicitement les paysages ou l'atmosphère de son lieu de naissance. Le nom "Fauré" est d'origine occitane, dérivant du latin "faber" (artisan).
- Milieu Familial : Sa famille, bien que de vieille souche languedocienne, a connu des hauts et des bas économiques. Son grand-père était boucher à Foix, et son père, Toussaint Fauré, a réussi à remonter l'échelle sociale en devenant directeur d'un collège.
Éducation Musicale :
- Premiers Pas : Fauré montre très tôt des talents musicaux, et à l'âge de neuf ans, il est envoyé à l'école de musique Niedermeyer à Paris, qui forme des musiciens d'église. Il y reçoit une formation complète en musique sacrée et en composition.
- Mentorat : À l'école Niedermeyer, il est remarqué par plusieurs enseignants, dont Camille Saint-Saëns, qui deviendra un mentor et un ami proche, influençant profondément sa carrière musicale.
Début de Carrière
- Rennes : Après avoir terminé ses études, Fauré est nommé maître de chapelle et organiste à l'église Saint-Sauveur de Rennes en 1866. Malgré des conditions de travail peu motivantes et une faible rétribution, il commence à composer et à enseigner.
- Retour à Paris : Encouragé par Saint-Saëns, Fauré retourne à Paris en 1870, où il commence à intégrer les cercles musicaux de la capitale. Il obtient un poste d'organiste de chœur à Notre-Dame de Clignancourt.
Années 1870-1877
Retour à Paris et début de carrière :
- Fauré retourne à Paris et s'installe avec son frère rue des Missions (aujourd'hui rue de l'Abbé-Grégoire).
- Grâce à Camille Saint-Saëns, il noue de nombreuses relations dans le monde musical, notamment avec les acteurs de la future Société nationale de musique.
- Il occupe sans grande motivation le poste d'organiste à Notre-Dame de Clignancourt.
Engagement militaire :
- La guerre entre la France et la Prusse éclate le 19 juillet 1870. Fauré s'engage volontairement dans le 1er régiment de voltigeurs de la garde impériale et participe à plusieurs combats.
- Pendant l'Armistice, surpris le 28 janvier 1871, il rentre chez lui et obtient le poste d'organiste à Saint-Honoré-d’Eylau.
Période de la Commune de Paris :
- Prévoyant les troubles à venir avec la Commune de Paris, il quitte Paris pour Rambouillet et rejoint ensuite l'école Niedermeyer en Suisse.
- Durant cette période, il compose un "Ave Maria" pour trois voix d'hommes et orgue.
Retour à la vie musicale :
- De retour à Paris, Fauré compose plusieurs mélodies, explorant les poèmes de Hugo, Gautier et Baudelaire. Cependant, il ne se sent pas totalement à l'aise avec la nature emphatique de ces textes, préférant la suggestion des symbolistes comme Verlaine.
Années 1877-1883
Relations Sentimentales :
- Rupture avec Marianne Viardot : En 1877, Fauré a rompu ses fiançailles avec Marianne Viardot, une relation qui l'a profondément affecté. Après cette rupture, il part avec Saint-Saëns à Weimar, où Liszt a organisé une représentation de "Samson et Dalila" de Saint-Saëns.
- Mariage avec Marie Fremiet : En 1883, Fauré se marie avec Marie Fremiet, la fille du sculpteur Emmanuel Fremiet. Le mariage est considéré comme une consécration bourgeoise pour Fauré, et le couple s'installe avenue Niel à Paris. Leur premier fils, Emmanuel, naît peu après le mariage.
</context>
"""

prompt = PromptTemplate.from_template(string_template)

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
    console.print("[cyan]Bonjour et bienvenue, vous voici face à Garbiel Fauré ! Pour mettre fin à votre discussion avec le célèbre compositeur français, veuillez appuyez sur Ctrl+C.") 
    try:
        while True:
            console.input(
                "Veuillez appuyer sur la touche Entrée pour démarrer un enregistrement ou pour y mettre fin."
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
                with console.status("Gabriel Fauré vous écoute avec intérêt...", spinner="earth"): 
                    result = stt_model.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
                # Extract text from the result
                text = result["text"].strip() 
                # Print transcription   
                console.print("[yellow]Vous :", f"{text}")

                # Get LLM response
                with console.status("Gabriel Fauré réfléchit en se grattant la barbe...", spinner="earth"):
                    input = prompt.invoke(input={"text": text})
                    result = llm.invoke(input=input)
                # Print response
                console.print("[yellow]Gabriel Fauré :", f"{result}")
            else:
                console.print(
                    "[red]Aucun audio enregistré. Pouvez-vous vous assurer du bon fonctionnement de votre microphone ?."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Fin de la session...")

    console.print("[blue]Session terminée. Au plaisir de vous revoir !")


