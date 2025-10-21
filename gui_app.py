from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import base64

app = Flask(__name__)

# Load model và labels
model = tf.keras.models.load_model("cnn_model.h5")
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Hàm dự đoán
def predict_image_pil(img: Image.Image):
    img = img.resize((30, 30)).convert("RGB")
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[class_id], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = ""
    img_data = ""
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                img = Image.open(file.stream)
                label, conf = predict_image_pil(img)

                # Chuyển ảnh sang base64 để hiển thị
                buffered = io.BytesIO()
                img.thumbnail((250, 250))
                img.save(buffered, format="PNG")
                img_data = base64.b64encode(buffered.getvalue()).decode()

                result_text = f"Biển báo: {label} | Độ tin cậy: {conf*100:.2f}%"

    return render_template("index.html", result=result_text, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)
