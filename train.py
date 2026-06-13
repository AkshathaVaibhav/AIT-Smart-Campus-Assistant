import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

lemmatizer = WordNetLemmatizer()

with open('chatbot_dataset.json', 'r') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents:
    tag = intent['intent']
    if tag not in classes:
        classes.append(tag)
    for question in intent['questions']:
        # Clean text instantly before tokenization
        word_list = nltk.word_tokenize(question.lower())
        words.extend(word_list)
        documents.append((word_list, tag))

words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word) for word in doc[0]]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

np.random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Optimize model layers for smaller text inputs
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.3)) # Lower dropout prevents underfitting small data
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Using Adam optimizer works significantly better for small intent maps
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

print("🔄 Retraining model with optimized weights...")
model.fit(train_x, train_y, epochs=300, batch_size=4, verbose=1)
model.save('chatbot_model.h5')
print("✅ Done retraining!")