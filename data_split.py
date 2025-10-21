import os
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = "dataset"
output_dir = "data_split"

# Tạo thư mục đầu ra
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Duyệt qua từng class
classes = sorted(os.listdir(dataset_dir))

for class_name in classes:
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)

    # Chia train (70%), val (15%), test (15%)
    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # Copy file sang thư mục mới
    for f in train_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_dir, "train", class_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_dir, "val", class_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for f in test_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(output_dir, "test", class_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

print("Đã chia xong dataset vào thư mục", output_dir)
