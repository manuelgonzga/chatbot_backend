import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext

# Inicialización
lemmatizer = WordNetLemmatizer()
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Limpieza del input del usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertir frase a vector (bag of words)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecir intención
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    return classes[max_index]

# Seleccionar una respuesta aleatoria de la intención
def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Función principal para responder
def respuesta(message):
    tag = predict_class(message)
    return get_response(tag, intents)

# ===== INTERFAZ TKINTER =====

# Función cuando el usuario hace clic en "Enviar"
def enviar():
    user_input = entry.get()
    chat_log.insert(tk.END, "Tú: " + user_input + "\n")
    entry.delete(0, tk.END)

    bot_response = respuesta(user_input)
    chat_log.insert(tk.END, "Bot: " + bot_response + "\n\n")
    chat_log.see(tk.END)

# Crear ventana
ventana = tk.Tk()
ventana.title("Chatbot - MaiNU")

# Área de conversación
chat_log = scrolledtext.ScrolledText(ventana, width=60, height=20)
chat_log.pack(padx=10, pady=10)

# Campo de texto y botón
entry = tk.Entry(ventana, width=45)
entry.pack(padx=10, pady=5, side=tk.LEFT)

boton = tk.Button(ventana, text="Enviar", command=enviar)
boton.pack(padx=5, pady=5, side=tk.LEFT)

# Iniciar aplicación
ventana.mainloop()
