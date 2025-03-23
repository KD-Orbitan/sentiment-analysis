import pandas as pd
import re

# Đọc file nhỏ
data = pd.read_csv('sentiment_small.csv')

# Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Xóa ký tự đặc biệt và số
    text = text.lower()  # Chuyển thành chữ thường
    return text

# Áp dụng làm sạch
data['text'] = data['text'].apply(clean_text)

# Lưu lại file đã làm sạch
data.to_csv('sentiment_small.csv', index=False)
# print(data['text'].head())

# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Đọc file đã làm sạch
# data = pd.read_csv('sentiment_small.csv')

# Tokenization
tokenizer = Tokenizer(num_words=5000)  # Giới hạn 5000 từ phổ biến nhất
tokenizer.fit_on_texts(data['text'])  # Học từ vựng từ dữ liệu
sequences = tokenizer.texts_to_sequences(data['text'])  # Chuyển văn bản thành chuỗi số

# Padding
X = pad_sequences(sequences, maxlen=100)  # Đảm bảo mỗi chuỗi dài 100

# Lưu tokenizer nếu cần tái sử dụng
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(X[:5])  # Kiểm tra 5 chuỗi đầu