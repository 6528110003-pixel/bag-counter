from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    count = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # โหลดโมเดลตอนใช้งาน
        model = YOLO("best.pt")

        results = model(path)

        count = len(results[0].boxes)

        image_path = path

    return render_template("index.html", count=count, image=image_path)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
