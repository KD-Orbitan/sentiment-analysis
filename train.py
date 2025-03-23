import tensorflow as tf

# Tắt GPU, chỉ dùng CPU
tf.config.set_visible_devices([], 'GPU')

import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu đã xử lý
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X, y = data['X'], data['y']

print("X shape:", X.shape)
print("y shape:", y.shape)

# Chia dữ liệu (đảm bảo shuffle)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=True)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Kiểm tra phân bố nhãn
print("Train label distribution:", np.sum(y_train, axis=0))
print("Val label distribution:", np.sum(y_val, axis=0))
print("Test label distribution:", np.sum(y_test, axis=0))

# Lưu dữ liệu đã chia
split_data = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test
}
with open('data/split_data.pkl', 'wb') as f:
    pickle.dump(split_data, f)

# Xây dựng mô hình
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=1)

# Huấn luyện
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Lưu mô hình
model.save('data/sentiment_model.keras')
print("Model saved to data/sentiment_model.keras")

# Đánh giá trên tập test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Dự đoán trên tập test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Tính precision, recall
print(classification_report(y_test_classes, y_pred_classes, target_names=['Negative', 'Positive']))

# Vẽ biểu đồ loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig('data/loss_plot.png')
# plt.show()

# Lưu kết quả huấn luyện
with open('data/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)