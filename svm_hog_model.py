import os
import numpy as np
import time
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

# 1. Đọc file labels

def load_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels_vi = load_labels("labels.txt")
classes = len(labels_vi)
print(f"Đã tải {classes} nhãn từ labels.txt:")

# 2. Hàm load dữ liệu

def load_dataset(folder, classes, image_size=(64, 64)):
    data, labels = [], []
    for i in range(classes):
        class_dir = os.path.join(folder, str(i))
        if not os.path.isdir(class_dir):
            print(f"Bỏ qua class {i} (không có thư mục).")
            continue
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            try:
                img = imread(file_path, as_gray=True)
                img_resized = resize(img, image_size, anti_aliasing=True)
                features = hog(img_resized,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L2-Hys')
                data.append(features)
                labels.append(i)
            except Exception as e:
                print("Lỗi đọc ảnh:", file_path, e)
    return np.array(data), np.array(labels)

# 3. Load dataset

cur_path = os.getcwd()

print("Đang load dữ liệu train...")
X_train, y_train = load_dataset(os.path.join(cur_path, "data_split", "train"), classes)

print("Đang load dữ liệu test...")
X_test, y_test = load_dataset(os.path.join(cur_path, "data_split", "test"), classes)

print("Kích thước train:", X_train.shape, " | Test:", X_test.shape)

# 4. Train hoặc Load SVM

model_path = "svm_hog_model.joblib"

if os.path.exists(model_path):
    print("\nĐang load mô hình SVM đã lưu...")
    clf = load(model_path)
else:
    print("\nBắt đầu train SVM + HOG...")
    start = time.time()
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)
    end = time.time()
    print(f"Thời gian train: {end - start:.2f}s")

    dump(clf, model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

# 5. Đánh giá mô hình

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nĐộ chính xác trên tập test:", acc)
print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=labels_vi))
