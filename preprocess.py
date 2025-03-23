import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def preprocess_data(input_file, output_file, max_words=5000, maxlen=100):
    # Đọc file nhỏ
    data = pd.read_csv(input_file)
    
    # Làm sạch văn bản
    data['text'] = data['text'].apply(clean_text)
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    
    # Padding
    X = pad_sequences(sequences, maxlen=maxlen)
    
    # Lưu tokenizer
    with open('data/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Chuyển nhãn thành one-hot encoding và ép kiểu thành số
    y = pd.get_dummies(data['polarity']).values.astype(np.int32)  # Ép thành int32
    
    # Lưu dữ liệu đã xử lý
    data_processed = {'X': X, 'y': y}
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(data_processed, f)
    
    print("Sample X:", X[:5])
    print("Sample y:", y[:5])
    print("y shape:", y.shape)

if __name__ == "__main__":
    input_file = 'data/sentiment_small.csv'
    output_file = 'data/sentiment_small.csv'
    preprocess_data(input_file, output_file)