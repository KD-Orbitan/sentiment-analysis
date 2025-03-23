# Sentiment Analysis Project

## Overview
Dự án phân loại cảm xúc (tiêu cực/tích cực) trên dữ liệu Twitter bằng mô hình LSTM.

## Dataset
- Nguồn: Sentiment140 (1.6M dòng).
- Dữ liệu dùng: 10,000 mẫu (5000 tiêu cực, 5000 tích cực).

## Pipeline
1. `get_data.py`: Lấy và shuffle dữ liệu.
2. `preprocess.py`: Làm sạch, tokenization, padding.
3. `train.py`: Huấn luyện LSTM (64 units, dropout 0.2, early stopping).
4. `predict.py`: Dự đoán trên dữ liệu mới.

## Results
- Test Accuracy: 72.47%.
- Precision/Recall:
  - Negative: 0.69 / 0.80.
  - Positive: 0.77 / 0.65.
- Val Loss tốt nhất: 0.5316 (epoch 2).
- Ví dụ dự đoán:
  - "I love this product so muchhhhh!" → Positive (70.3%).
  - "This is the worst experience ever." → Negative (98.9%).
  - "It's okay, nothing special." → Positive (69.9%, hơi sai vì trung tính).

## Usage
- Cài đặt: `pip install -r requirements.txt`.
- Chạy: `python3 predict.py` với câu mới.
