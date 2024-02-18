# Conversational Bot with LLM
This Python script implements a conversational bot powered by a language model. The bot interacts with users, provides answers based on pre-existing knowledge, and learns llm model responses to enhance its knowledge base over time.

## Features
* Memory: The bot maintains a memory of previously asked questions and their corresponding answers in a SQLite database named memory.db.
* Text-to-Speech: Utilizes the pyttsx3 library to convert text responses into speech for enhanced user interaction.
* Contextual Learning: Incorporates LLM model to provide answers to questions not found in its memory.
* Interactive Interface: The bot continuously interacts with users until instructed to quit.

## Issues
* Doesn't always find question because it needs more complex method that uses Natural language processing (NLP).