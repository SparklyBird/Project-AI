import json
import pyttsx3
import copy
from difflib import get_close_matches
from llama_cpp import Llama


def load_memory(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data


def save_memory(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def find_best_match(user_question: str, questions: list[str]) -> str or None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=1)
    return matches[0] if matches else None


def get_answer(question: str, knowledge: dict) -> str or None:
    for q in knowledge['questions']:
        if q['question'] == question:
            return q['answer']


def convert_text_to_speech(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error converting text-to-speech:", str(e))


def run_model_with_prompt(llm, text_prompt):
    try:
        stream = llm(
            f"Question: {text_prompt} Answer:",
            max_tokens=100,
            stop=["\n", "Question:", "Q:"],
            stream=True
        )

        output_text = ""
        for output in stream:
            completionFragment = copy.deepcopy(output)
            output_text += completionFragment["choices"][0]["text"]

        return output_text

    except Exception as e:
        print("Error executing model:", str(e))


def bot(llm):
    memory: dict = load_memory('memory.json')
    while True:
        user_input: str = input('User: ')
        if user_input.lower() == 'quit':
            break
        best_match: str or None = find_best_match(user_input, [q['question'] for q in memory['questions']])

        if best_match:
            answer = get_answer(best_match, memory)
            print(f'Bot: {answer}')
            convert_text_to_speech(answer)
        else:
            llm_answer = run_model_with_prompt(llm, user_input)
            print(f'Bot: {llm_answer}')
            convert_text_to_speech(llm_answer)
            memory['questions'].append({'question': user_input, 'answer': llm_answer})
            save_memory('memory.json', memory)


if __name__ == '__main__':
    # load the LLM model
    try:
        print("Loading LLM model...")
        llm = Llama(model_path="./models/mistral-7b-v0.1.Q4_0.gguf")
        print("LLM Model loaded!")
    except Exception as e:
        print("Error loading LLM model:", str(e))
    bot(llm)
