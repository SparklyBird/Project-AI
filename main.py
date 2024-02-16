import json
import pyttsx3
import copy
from difflib import get_close_matches, SequenceMatcher
from llama_cpp import Llama
import textwrap
from contextlib import ContextDecorator
import os
import sys
import requests
from bs4 import BeautifulSoup
from spellchecker import SpellChecker


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


def get_similar_questions(question: str, questions: list[str]) -> list[str]:
    similar_questions = []
    for q in questions:
        if SequenceMatcher(None, question.lower(), q.lower()).ratio() > 0.8:
            similar_questions.append(q)
    return similar_questions


def get_answer_for_similar_questions(similar_questions: list[str], knowledge: dict) -> str or None:
    for q in similar_questions:
        for entry in knowledge['questions']:
            if entry['question'].lower() == q.lower():
                return entry['answer']
    return None


def bot(llm):
    memory: dict = load_memory('memory.json')
    spell = SpellChecker()
    while True:
        user_input: str = input('User: ')
        if user_input.lower() == 'quit':
            break

        # Spell check and correct user input
        corrected_input = ' '.join([spell.correction(word) for word in user_input.split()])

        best_match: str or None = find_best_match(corrected_input, [q['question'] for q in memory['questions']])

        if best_match:
            answer = get_answer(best_match, memory)
            print(f'Bot:')
            # Increase the width parameter to ensure complete response display
            for line in textwrap.wrap(answer, width=150):
                print(line)
            convert_text_to_speech(answer)
        else:
            wikipedia_summary = search_wikipedia(corrected_input)
            if wikipedia_summary:
                print(f'Bot:')
                # Increase the width parameter to ensure complete response display
                for line in textwrap.wrap(wikipedia_summary, width=150):
                    print(line)
                convert_text_to_speech(wikipedia_summary)
                memory['questions'].append({'question': corrected_input, 'answer': wikipedia_summary})
                save_memory('memory.json', memory)
            else:
                llm_answer = run_model_with_prompt(llm, corrected_input)
                print(f'Bot:')
                # Increase the width parameter to ensure complete response display
                for line in textwrap.wrap(llm_answer, width=150):
                    print(line)
                convert_text_to_speech(llm_answer)
                memory['questions'].append({'question': corrected_input, 'answer': llm_answer})
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