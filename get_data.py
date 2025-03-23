import pandas as pd

def get_small_dataset(input_file, output_file, nrows_per_class=5000):
    # Đọc file gốc
    data = pd.read_csv(input_file, encoding='latin-1', header=None)
    data.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    data = data[['text', 'polarity']]
    
    # Lấy 5000 dòng polarity = 0 và 5000 dòng polarity = 4
    data_neg = data[data['polarity'] == 0].head(nrows_per_class)
    data_pos = data[data['polarity'] == 4].head(nrows_per_class)
    data_balanced = pd.concat([data_neg, data_pos])
    
    # Chuyển polarity: 0 giữ nguyên, 4 thành 1
    data_balanced['polarity'] = data_balanced['polarity'].replace({0: 0, 4: 1})
    
    # Xáo trộn dữ liệu
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Lưu file nhỏ
    data_balanced.to_csv(output_file, index=False)
    print(data_balanced['polarity'].value_counts())
    print(data_balanced.head())
    print(data_balanced.tail())

if __name__ == "__main__":
    input_file = 'training.1600000.processed.noemoticon.csv'
    output_file = 'data/sentiment_small.csv'
    get_small_dataset(input_file, output_file)