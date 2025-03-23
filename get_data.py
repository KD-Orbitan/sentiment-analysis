import pandas as pd

def get_small_dataset(input_file, output_file, nrows=10000):
    # Đọc file gốc
    data = pd.read_csv(input_file, encoding='latin-1', header=None, nrows=nrows)
    data.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    data = data[['text', 'polarity']]
    
    # Chuyển polarity: 0 (tiêu cực), 1 (trung lập), 2 (tích cực)
    data['polarity'] = data['polarity'].replace({0: 0, 2: 1, 4: 2})
    
    # Lưu file nhỏ
    data.to_csv(output_file, index=False)
    print(data.head())

if __name__ == "__main__":
    input_file = 'training.1600000.processed.noemoticon.csv'  # File gốc lớn
    output_file = 'data/sentiment_small.csv'  # File nhỏ
    get_small_dataset(input_file, output_file)