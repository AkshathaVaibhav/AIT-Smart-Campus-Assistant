import os
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Helper function to get absolute paths for files in serverless environment
def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# Load resources (Ensure these files are in the same /api folder as this script)
with open(get_path('chatbot_dataset.json'), 'r') as f:
    intents = json.load(f)

words = pickle.load(open(get_path('words.pkl'), 'rb'))
classes = pickle.load(open(get_path('classes.pkl'), 'rb'))
model = load_model(get_path('chatbot_model.h5'))

def clean_up_sentence(sentence):
    # Ensure NLTK data is downloaded if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')
    
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

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
    
    ERROR_THRESHOLD = 0.50 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't fully understand that question regarding Atria. Could you please rephrase?"
    
    tag = intents_list[0]['intent']
    for i in intents_json:
        if i['intent'] == tag:
            return i['answer']
    return "I am experiencing trouble fetching that specific detail right now."

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "Please type something!"})
        
    ints = predict_class(user_message)
    reply = get_response(ints, intents)
    return jsonify({"response": reply})

@app.route('/api/download-timetable', methods=['GET'])
def download_timetable():
    try:
        # File must be inside the api/ folder
        return send_file(get_path('Draft TT UG - July 2026.pdf'))
    except Exception as e:
        return jsonify({"error": "Timetable file not found on server."}), 404

# Note: Do not include app.run() here; Vercel handles the execution.