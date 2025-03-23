import pandas as pd

# Đọc file gốc (toàn bộ hoặc giới hạn tùy máy bạn)
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, nrows=10000)  # Lấy 10,000 dòng
data.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']  # Đặt tên cột
data = data[['text', 'polarity']]  # Giữ text và nhãn

# Chuyển polarity: 0 (tiêu cực), 1 (trung lập), 2 (tích cực)
data['polarity'] = data['polarity'].replace({0: 0, 2: 1, 4: 2})

# Lưu thành file nhỏ
data.to_csv('sentiment_small.csv', index=False)
print(data.head())