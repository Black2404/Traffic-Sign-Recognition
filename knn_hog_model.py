import os
import cv2
import time
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Đọc file labels

def load_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Đọc danh sách nhãn từ file labels.txt
labels_vi = load_labels("labels.txt")
classes = len(labels_vi)
print(f"Đã tải {classes} nhãn từ labels.txt:")

# 2. Hàm trích xuất HOG

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

# 3. Hàm load dữ liệu

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
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                features = extract_hog_features(img)
                data.append(features)
                labels.append(i)
            except Exception as e:
                print(f"Lỗi load ảnh: {file_path} | {e}")
                continue

    return np.array(data), np.array(labels)

# 4. Load dataset

cur_path = os.getcwd()

print("Đang load train...")
X_train, y_train = load_data(os.path.join(cur_path, "data_split", "train"), classes)

print("Đang load val...")
X_val, y_val = load_data(os.path.join(cur_path, "data_split", "val"), classes)

print("Đang load test...")
X_test, y_test = load_data(os.path.join(cur_path, "data_split", "test"), classes)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# 5. Huấn luyện hoặc Load mô hình KNN

model_path = "knn_hog_model.pkl"

if os.path.exists(model_path):
    print("\nĐang load mô hình KNN đã lưu...")
    knn = joblib.load(model_path)
else:
    print("\nTraining KNN + HOG...")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    end = time.time()
    print(f"Thời gian train: {end - start:.2f}s")

    # Lưu mô hình sau khi train
    joblib.dump(knn, model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

# 6. Đánh giá mô hình

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nĐộ chính xác trên tập test:", acc)

# In báo cáo chi tiết theo tên nhãn
print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=labels_vi))
