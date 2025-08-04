import os
import pandas as pd
from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def extract_text_from_file(path):
    if path.lower().endswith('.pdf'):
        return extract_pdf(path)
    elif path.lower().endswith('.docx'):
        doc = Document(path)
        return '\n'.join([para.text for para in doc.paragraphs])
    return ""

def build_dataset(folder):
    data = []
    for label in ['resumes', 'not_resumes']:
        label_folder = os.path.join(folder, label)
        for file in os.listdir(label_folder):
            path = os.path.join(label_folder, file)
            text = extract_text_from_file(path)
            if text.strip():
                data.append((text, 1 if label == 'resumes' else 0))
    return pd.DataFrame(data, columns=['text', 'label'])

# 🔽 INSERTED: Create folders if they don't exist
training_data_path = r"C:/Users/susha/Desktop/Resume Classifier/Training Data"
os.makedirs(os.path.join(training_data_path, "resumes"), exist_ok=True)
os.makedirs(os.path.join(training_data_path, "not_resumes"), exist_ok=True)

# Load dataset
df = build_dataset(training_data_path)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=500)
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Embedding(10000, 64))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)

# Folder where model files will be saved
save_folder = r"D:/Resume/Model Data"
os.makedirs(save_folder, exist_ok=True)

model_path = os.path.join(save_folder, "resume_classifier_model.keras")
tokenizer_path = os.path.join(save_folder, "resume_tokenizer.pkl")

# Save model and tokenizer
model.save(model_path)
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Tokenizer saved to: {tokenizer_path}")