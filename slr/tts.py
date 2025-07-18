import pyttsx3

# Initialize the TTS engine only once
_tts_engine = pyttsx3.init()

def speak_word(word: str):
    """
    Speak the given word aloud using the system's TTS engine.
    Only speaks if the word is non-empty and not just whitespace.
    """
    if word and word.strip():
        _tts_engine.say(word)
        _tts_engine.runAndWait()
