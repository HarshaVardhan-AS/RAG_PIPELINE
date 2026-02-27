from gtts import gTTS
import uuid
import os
import re

def clean_text_for_tts(text: str) -> str:
    # Remove markdown symbols
    text = re.sub(r'\*+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\n+', '. ', text)

    return text.strip()
def text_to_speech(text: str, lang: str = "en") -> str:

    text = clean_text_for_tts(text)

    os.makedirs("audio", exist_ok=True)

    filename = f"audio/{uuid.uuid4()}.mp3"

    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    return filename