import tensorflow as tf

# Tắt GPU, chỉ dùng CPU
tf.config.set_visible_devices([], 'GPU')

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
# Tải tokenizer
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Tải mô hình
model = load_model('data/sentiment_model.keras')

# Hàm dự đoán
def predict_sentiment(texts):
    # Làm sạch văn bản
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text
    
    cleaned_texts = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    # Dự đoán
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes, predictions

# Ví dụ
sample_texts = [
    "I love this product so muchhhhh!",
    "This is the worst experience ever.",
    "It's okay, nothing special.",
    "I'm so happy with this product!",
    "I'm so sad with this product!",
    "I'm so angry with this product!",
    "I'm so excited with this product!",
    "I'm so nervous with this product!",
    "I hate this product!",
]
labels, probs = predict_sentiment(sample_texts)

# In kết quả
for text, label, prob in zip(sample_texts, labels, probs):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment} (Prob: {prob})")
    print()