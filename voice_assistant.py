import speech_recognition as sr
import pyttsx3
from datetime import datetime

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

speak("Hello! I am your AI Vision Assistant. How can I help you?")

while True:

    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print("You:", command)

        if "hello" in command or "hi" in command:
            speak("Hello! Nice to meet you.")

        elif "time" in command:
            time = datetime.now().strftime("%I:%M %p")
            speak("The time is " + time)

        elif "what can you do" in command:
            speak(
                "I can detect objects, recognize hand gestures, "
                "read text and help with safety detection."
            )

        elif "stop" in command or "exit" in command:
            speak("Goodbye!")
            break

        else:
            speak("I heard you, but I don't know that command yet.")

    except sr.UnknownValueError:
        print("Could not understand.")

    except sr.RequestError:
        speak("Speech recognition service is unavailable.")