import cv2 as cv
import os
import time
import speech_recognition as sr

def speech_to_images(window_name, images_folder):
    """
    Captures speech input, converts to text, and displays alphabet images
    for each character with timing rules:
    - 0.5s between characters in a word
    - 1s pause between words
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Please speak now...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return

    # Prepare list of images and timing
    chars = []
    for word in text.split():
        for c in word:
            if c.isalpha():
                chars.append((c.upper(), 0.5))
        chars.append((" ", 1.0))  # Word break

    # Remove last space pause
    if chars and chars[-1][0] == " ":
        chars = chars[:-1]

    # Display each character image
    for char, pause in chars:
        if char == " ":
            time.sleep(pause)
            continue
        img_path = os.path.join(images_folder, f"{char}.png")
        if not os.path.exists(img_path):
            print(f"Image for '{char}' not found at {img_path}")
            continue
        img = cv.imread(img_path)
        if img is None:
            print(f"Failed to load image for '{char}'")
            continue
        cv.imshow(window_name, img)
        cv.waitKey(int(pause * 1000))
    # After completion, close the image window
    cv.destroyWindow(window_name)

import speech_recognition as sr

def recognize_speech():
    """
    Listens to the user's speech via microphone and returns the recognized text as a string.
    Uses Google's free speech recognition API.
    Returns an empty string if nothing is recognized or on error.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Please speak now...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""