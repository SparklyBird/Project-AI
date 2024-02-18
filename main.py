import json
import pyttsx3
import copy
from difflib import get_close_matches
from llama_cpp import Llama
import textwrap
from contextlib import ContextDecorator
import os
import sys
from spellchecker import SpellChecker
import sqlite3

DATABASE_FILE = 'memory.db'


def create_table():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
                        id INTEGER PRIMARY KEY,
                        question TEXT,
                        answer TEXT
                    )''')
    conn.commit()
    conn.close()


def load_memory() -> list:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memory")
    rows = cursor.fetchall()
    conn.close()
    return [{'question': row[1], 'answer': row[2]} for row in rows]


def save_memory(data: list):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory")
    for item in data:
        # Capitalize the first word and ensure the question ends with a question mark
        question = item['question'].strip().capitalize()
        if not question.endswith('?'):
            question += '?'
        cursor.execute("INSERT INTO memory (question, answer) VALUES (?, ?)", (question, item['answer']))
    conn.commit()
    conn.close()


def find_best_match(user_question: str, questions: list[str]) -> str or None:
    user_question_lower = user_question.lower()
    questions_lower = [q.lower() for q in questions]
    matches: list = get_close_matches(user_question_lower, questions_lower, n=1, cutoff=0.8)
    if matches:
        original_question = questions[questions_lower.index(matches[0])]
        return original_question
    return None


def get_answer(question: str, knowledge: list) -> str or None:
    for q in knowledge:
        if q['question'] == question:
            return q['answer']
    return None


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
    create_table()
    memory = load_memory()
    spell = SpellChecker()
    while True:
        user_input = input('User: ')
        if user_input.lower() == 'quit':
            break
        corrected_input = ' '.join([spell.correction(word) for word in user_input.split()])
        best_match = find_best_match(corrected_input, [q['question'] for q in memory])
        if best_match:
            answer = get_answer(best_match, memory)
            print(f'Bot:')
            for line in textwrap.wrap(answer, width=120):
                print(line)
            convert_text_to_speech(answer)
        else:
            llm_answer = run_model_with_prompt(llm, corrected_input)
            print(f'Bot:')
            for line in textwrap.wrap(llm_answer, width=120):
                print(line)
            convert_text_to_speech(llm_answer)
            memory.append({'question': corrected_input, 'answer': llm_answer})
            save_memory(memory)


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
