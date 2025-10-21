import os
import cv2
import numpy as np
import time
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Hàm trích xuất HOG

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

# 2. Hàm load dữ liệu

def load_data(folder, classes, img_size=(30, 30)):
    data, labels = [], []
    for i in range(classes):
        path = os.path.join(folder, str(i))
        if not os.path.exists(path):
            print(f"Bỏ qua class {i} (không có thư mục).")
            continue

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
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

# 3. Đọc nhãn từ file labels.txt

def load_labels(label_file="labels.txt"):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Đã tải {len(labels)} nhãn từ labels.txt")
    return labels

labels_vi = load_labels()
classes = len(labels_vi)

# 4. Load dataset

cur_path = os.getcwd()

print("\nĐang load dữ liệu train...")
X_train, y_train = load_data(os.path.join(cur_path, "data_split", "train"), classes)

print("Đang load dữ liệu val...")
X_val, y_val = load_data(os.path.join(cur_path, "data_split", "val"), classes)

print("Đang load dữ liệu test...")
X_test, y_test = load_data(os.path.join(cur_path, "data_split", "test"), classes)

print(f"\nKích thước dữ liệu:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 5. Huấn luyện hoặc Load mô hình

model_path = "randomforest_hog_model.pkl"

if os.path.exists(model_path):
    print("\nĐang load mô hình đã lưu...")
    rf = joblib.load(model_path)
else:
    print("\nĐang huấn luyện RandomForest + HOG...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    end = time.time()
    print(f"Thời gian train: {end - start:.2f}s")

    joblib.dump(rf, model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

# 6. Đánh giá mô hình

print("\nĐang đánh giá mô hình...")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác trên tập test: {acc:.4f}")

print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=labels_vi))
