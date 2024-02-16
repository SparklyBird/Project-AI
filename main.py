import json
import pyttsx3
import copy
from difflib import get_close_matches
from llama_cpp import Llama
import textwrap
from contextlib import ContextDecorator
import os
import sys
import requests
from bs4 import BeautifulSoup

def search_wikipedia(query):
    url = f"https://en.wikipedia.org/wiki/{query}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract the main content of the Wikipedia page
        main_content = soup.find('div', id='mw-content-text')
        if main_content:
            # Get the first paragraph as the summary
            summary = main_content.find('p').get_text()
            return summary
    return None

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
            print(f'Bot:')
            for line in textwrap.wrap(answer, width=80):
                print(line)
            convert_text_to_speech(answer)
        else:
            wikipedia_summary = search_wikipedia(user_input)
            if wikipedia_summary:
                print(f'Bot:')
                for line in textwrap.wrap(wikipedia_summary, width=80):
                    print(line)
                convert_text_to_speech(wikipedia_summary)
                memory['questions'].append({'question': user_input, 'answer': wikipedia_summary})
                save_memory('memory.json', memory)
            else:
                llm_answer = run_model_with_prompt(llm, user_input)
                print(f'Bot:')
                for line in textwrap.wrap(llm_answer, width=80):
                    print(line)
                convert_text_to_speech(llm_answer)
                memory['questions'].append({'question': user_input, 'answer': llm_answer})
                save_memory('memory.json', memory)


class suppress_stdout_stderr(ContextDecorator):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()

    def suppress(self):
        return self


if __name__ == '__main__':
    with suppress_stdout_stderr():
        try:
            print("Loading LLM model...")
            llm = Llama(model_path="./models/mistral-7b-v0.1.Q4_0.gguf")
            llm.verbose = False
            print("LLM Model loaded!")
        except Exception as e:
            print("Error loading LLM model:", str(e))
    bot(llm)