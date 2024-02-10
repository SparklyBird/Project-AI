import copy
import pyttsx3
from llama_cpp import Llama


def choose_language():
    print("Choose input language:")
    print("1 for English")
    print("2 for Russian")

    choice = input("Enter the number: ")

    if choice == "1":
        return "english"
    elif choice == "2":
        return "russian"
    else:
        print("Invalid choice. Defaulting to English.")
        return "english"


def get_text_input():
    return input("Enter your question: ")


print("Converting text to speech...")
def convert_text_to_speech(text, language):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        if language == "english":
            engine.setProperty('voice', voices[1].id)  # Change the index to select a different English voice
        elif language == "russian":
            engine.setProperty('voice', voices[2].id)  # Change the index to select a different Russian voice

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error converting text-to-speech:", str(e))


print("Running model...")
def run_model_with_prompt(llm, text_prompt):
    try:
        stream = llm(
            f"Question: {text_prompt} Answer:",
            max_tokens=100,
            stop=["\n", "Question:", "Q:"],
            stream=True
        )

        # Accumulate the output text
        output_text = ""
        for output in stream:
            completionFragment = copy.deepcopy(output)
            output_text += completionFragment["choices"][0]["text"]

        # Print the full output text
        print(output_text)

        # Convert text to speech
        convert_text_to_speech(output_text, selected_language)
    except Exception as e:
        print("Error executing model:", str(e))


# load the model
try:
    print("Loading model...")
    llm = Llama(model_path="./models/mistral-7b-v0.1.Q4_0.gguf")
    print("Model loaded!")
except Exception as e:
    print("Error loading model:", str(e))

# Choose language for TTS
selected_language = choose_language()

# Get text input for the prompt
text_prompt = get_text_input()

# Convert text to speech with the selected language
if text_prompt:
    print("Converting text to speech...")
    convert_text_to_speech(text_prompt, selected_language)
else:
    print("No input question provided.")

# Run the model with the text prompt
if text_prompt:
    run_model_with_prompt(llm, text_prompt)
else:
    print("No input question provided.")
