import os
import json
import pickle
import numpy as np
import traceback
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for all routes
from flask_cors import CORS
CORS(app, origins=["https://ait-smart-campus-assistant.onrender.com", "*"])

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# Load resources
with open(get_path('chatbot_dataset.json'), 'r') as f:
    intents = json.load(f)

words = pickle.load(open(get_path('words.pkl'), 'rb'))
classes = pickle.load(open(get_path('classes.pkl'), 'rb'))
model = load_model(get_path('chatbot_model.h5'))

# --- NEW: Serve the Frontend ---
@app.route('/')
def index():
    # Looks for index.html in the root folder (one level up from /api)
    return send_from_directory('../', 'index.html')

# Serve other files (like css/js) from the root folder
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../', path)

# --- Chatbot Logic ---
def clean_up_sentence(sentence):
    # Ensure NLTK data is downloaded if not present
    try:
        # Check for the new punkt_tab resource
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # Download all required NLTK resources
        nltk.download('punkt')
        nltk.download('punkt_tab')  # <--- THIS IS THE MAGIC FIX
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
        return "I'm sorry, I don't fully understand. Could you please rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json:
        if i['intent'] == tag:
            return i['answer']
    return "I am experiencing trouble fetching that detail right now."

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "Please type something!"})
            
        ints = predict_class(user_message)
        reply = get_response(ints, intents)
        return jsonify({"response": reply})
        
    except Exception as e:
        # This will catch ANY crash and send the exact error back to your screen!
        error_details = traceback.format_exc()
        print(error_details) # Prints to Render logs
        return jsonify({"response": f"SERVER CRASHED: {str(e)}"}), 200