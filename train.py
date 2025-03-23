import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf

# Tắt GPU, chỉ dùng CPU
tf.config.set_visible_devices([], 'GPU')

# Đọc dữ liệu đã xử lý
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X, y = data['X'], data['y']

print("X shape:", X.shape)
print("y shape:", y.shape)

# Chia dữ liệu
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Lưu dữ liệu đã chia
split_data = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test
}
with open('data/split_data.pkl', 'wb') as f:
    pickle.dump(split_data, f)

# Xây dựng mô hình (2 lớp)
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))

# Lưu mô hình
model.save('data/sentiment_model.h5')
print("Model saved to data/sentiment_model.h5")

# Đánh giá trên tập test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Lưu kết quả huấn luyện
with open('data/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)