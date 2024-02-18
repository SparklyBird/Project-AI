import pyttsx3
import copy
from difflib import get_close_matches
from llama_cpp import Llama
import textwrap
from contextlib import ContextDecorator
import os
import sys
import sqlite3
import spacy

nlp = spacy.load('en_core_web_lg')

DATABASE_FILE = 'memory.db'


def create_table():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
                        id INTEGER PRIMARY KEY,
                        question TEXT,
                        answer TEXT,
                        processed_info TEXT
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
        question = item['question'].strip().capitalize()
        if not question.endswith('?'):
            question += '?'
        # Extract parts of speech and numbers from the question
        doc = nlp(question)
        processed_info = {
            'verbs': ' '.join([token.lemma_ for token in doc if token.pos_ == 'VERB']),
            'adjectives': ' '.join([token.lemma_ for token in doc if token.pos_ == 'ADJ']),
            'nouns': ' '.join([token.lemma_ for token in doc if token.pos_ == 'NOUN']),
            'numbers': ' '.join([token.lemma_ for token in doc if token.pos_ == 'NUM'])
        }
        processed_info = ' '.join(processed_info.values())
        cursor.execute("INSERT INTO memory (question, answer, processed_info) VALUES (?, ?, ?)",
                       (question, item['answer'], processed_info))
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


def get_answer_based_on_similarity(question: str, database_file: str) -> str or None:
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    question_processed = preprocess_text(question.lower().strip('?'))
    question_doc = nlp(question_processed)
    input_info = ' '.join([token.lemma_ for token in question_doc if token.pos_ in {'VERB', 'ADJ', 'NOUN', 'NUM'}])
    cursor.execute("SELECT question, answer, processed_info FROM memory WHERE processed_info LIKE ?", (f"%{input_info}%",))
    rows = cursor.fetchall()
    conn.close()
    similar_questions = []
    max_similarity = 0.85  # Minimum similarity threshold
    for row in rows:
        db_question = row[0]
        db_answer = row[1]
        db_info = row[2]
        db_doc = nlp(db_info)
        # Check if both input question and database question have non-empty vectors
        if question_doc.vector_norm and db_doc.vector_norm:
            # Calculate similarity score based on processed information
            similarity_score = question_doc.similarity(db_doc)
            # Add the question, answer, and similarity score to the list if similarity score is above threshold
            if similarity_score >= max_similarity:
                similar_questions.append({'question': db_question, 'answer': db_answer, 'similarity': similarity_score})
    if not similar_questions:
        return None
    # Sort similar questions based on similarity score in descending order
    similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
    # Return the answer with the highest similarity score
    return similar_questions[0]['answer']


def bot(llm):
    create_table()
    memory = load_memory()
    while True:
        user_input = input('User: ')
        if user_input.lower() == 'quit':
            break
        corrected_input = user_input
        answer = get_answer_based_on_similarity(corrected_input, DATABASE_FILE)
        if answer:
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
