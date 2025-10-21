import os
import time
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, classification_report

# 1. Đọc file labels

def load_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Đọc danh sách nhãn từ file labels.txt
labels_vi = load_labels("labels.txt")
classes = len(labels_vi)  # Số lớp (12)
print(f"Đã tải {classes} nhãn từ labels.txt")

# 2. Hàm load dữ liệu

def load_data(folder, classes, img_size=(30, 30)):
    data, labels = [], []
    for i in range(classes):
        path = os.path.join(folder, str(i))
        if not os.path.exists(path):
            print(f"Bỏ qua class {i} (không có thư mục).")
            continue

        images = os.listdir(path)
        for a in images:
            file_path = os.path.join(path, a)
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize(img_size)
                img = np.array(img)
                data.append(img)
                labels.append(i)
            except Exception as e:
                print(f"Lỗi load ảnh: {file_path} | {e}")
                continue

    return np.array(data), np.array(labels)

# 3. Load dataset

cur_path = os.getcwd()

X_train, y_train = load_data(os.path.join(cur_path, "data_split", "train"), classes)
X_val, y_val     = load_data(os.path.join(cur_path, "data_split", "val"), classes)
X_test, y_test   = load_data(os.path.join(cur_path, "data_split", "test"), classes)

# Chuẩn hóa dữ liệu
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# One-hot encoding
y_train = to_categorical(y_train, classes)
y_val   = to_categorical(y_val, classes)
y_test  = to_categorical(y_test, classes)

# 4. Xây dựng mô hình CNN

model = Sequential([
    Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

# 5. Compile & Train

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
start = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1
)
end = time.time()

train_time = end - start
print(f"Thời gian huấn luyện: {train_time:.2f} giây ({train_time/60:.2f} phút)")

# 6. Đánh giá mô hình

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Độ chính xác trên tập test:", accuracy_score(y_true, y_pred_classes))
print("\nBáo cáo chi tiết:\n", classification_report(y_true, y_pred_classes, target_names=labels_vi))

# 7. Lưu mô hình

model.save("cnn_model.h5")
print("Đã lưu mô hình vào cnn_model.h5")
