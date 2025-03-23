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
print(data['text'].head())