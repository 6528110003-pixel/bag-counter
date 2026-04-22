from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# โหลดโมเดล
model = YOLO("best.pt")


@app.route("/")
def home():
    return {"status": "running", "service": "Bag Counter AI"}


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return {"error": "no image"}, 400

    file = request.files["image"]

    temp = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp.name)

    results = model(temp.name)

    count = len(results[0].boxes)

    return jsonify({"count": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
