from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Habilita CORS

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

# Cargar archivos del chatbot
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Funciones del chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    return classes[max_index]

def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Ruta principal que carga el HTML
@app.route("/")
def home():
    return send_from_directory('static', 'index.html')

# Ruta API del bot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("Mensaje recibido:", data)  # Debug
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"response": "Please write something!"})
    tag = predict_class(message)
    response = get_response(tag, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
