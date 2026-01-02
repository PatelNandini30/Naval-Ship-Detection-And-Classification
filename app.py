import os
import cv2
import torch
from flask import Flask, render_template, request, send_from_directory, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv
from collections import Counter

from config import UPLOAD_FOLDER, MODEL_PATH, CONF_THRESHOLD, ALLOWED_EXTENSIONS
from chatbot import ask_ai

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

print(f"[INFO] YOLO loaded on {device}")

last_detections = []
last_confidences = []

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads():
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    global last_detections, last_confidences
    last_detections = []
    last_confidences = []

    output_image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="No image selected")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type")

        clear_uploads()

        input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
        output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
        file.save(input_path)

        results = model.predict(source=input_path, conf=CONF_THRESHOLD, save=False)
        image = cv2.imread(input_path)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]
            last_detections.append(name)
            last_confidences.append(conf)

            label = f"{name} {conf*100:.1f}%"
            cv2.rectangle(image, (x1, y1), (x2, y2), (79, 179, 255), 2)
            cv2.putText(
                image, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (79, 179, 255), 2
            )

        cv2.imwrite(output_path, image)
        output_image_url = "/uploads/output.jpg"

    return render_template("index.html", image_url=output_image_url)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    context = ""
    if last_detections:
        counts = Counter(last_detections)
        avg_conf = sum(last_confidences) / len(last_confidences)

        confidence_level = (
            "High" if avg_conf > 0.75 else
            "Medium" if avg_conf > 0.5 else
            "Low"
        )

        context = (
            f"YOLO detected classes: {dict(counts)}. "
            f"Average confidence: {avg_conf:.2f} ({confidence_level}). "
        )

        if len(counts) > 1:
            context += (
                "Multiple ship classes detected, potential classification ambiguity."
            )

    reply = ask_ai(user_msg, context)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
