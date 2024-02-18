# Intelligent Chatbot with Database Integration
This Python script implements a conversational bot powered by a language model. The bot interacts with users, provides answers based on pre-existing knowledge, and learns from llm model responses to enhance its knowledge base over time.

## Features
* Memory: The bot maintains a memory of previously asked questions and their corresponding answers in a SQLite database file named memory.db.
* Natural Language Processing: Employs spaCy for natural language processing tasks such as part-of-speech tagging and similarity scoring.
* Text-to-Speech: Utilizes the pyttsx3 library to convert text responses into speech for enhanced user interaction.
* Contextual Learning: Incorporates the LLM model to provide answers to questions not found in its memory.
* Interactive Interface: The bot continuously interacts with users until instructed to quit.