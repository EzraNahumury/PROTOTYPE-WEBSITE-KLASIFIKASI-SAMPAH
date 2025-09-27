import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image
import time

# === Flask App ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Pastikan folder upload ada
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# === Load TFLite Model ===
interpreter = tf.lite.Interpreter(model_path="model_fix.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # contoh: (1, 224, 224, 3)

# === Daftar nama kelas (urut sesuai train_dataset.class_names di Colab) ===
class_names = ["kaca", "kardus", "kertas", "logam", "plastik", "residu"]


# === Preprocess Gambar ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", predicted_class=None)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", predicted_class=None)

        # Simpan file upload (pakai timestamp biar nama unik)
        filename = f"{int(time.time())}_{file.filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess
        input_data = preprocess_image(file_path)

        # Prediksi pakai TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(output_data))

        return render_template(
            "index.html",
            filename=filename,  # hanya nama file
            predicted_class=predicted_class,
            confidence=confidence
        )

    return render_template("index.html", predicted_class=None)

# === Run Flask ===
if __name__ == "__main__":
    app.run(debug=True)
