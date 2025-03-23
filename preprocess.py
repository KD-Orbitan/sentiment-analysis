import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Xóa ký tự đặc biệt và số
    text = text.lower()  # Chuyển thành chữ thường
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
    
    # Lưu dữ liệu đã xử lý (X và y)
    y = pd.get_dummies(data['polarity']).values  # Chuyển nhãn thành one-hot
    data_processed = {'X': X, 'y': y}
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(data_processed, f)
    
    print("Sample X:", X[:5])
    print("Sample y:", y[:5])

if __name__ == "__main__":
    input_file = 'data/sentiment_small.csv'
    output_file = 'data/sentiment_small.csv'  # Không cần lưu lại nếu không muốn
    preprocess_data(input_file, output_file)