from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# ===== Roboflow Client =====
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_ROBOFLOW_API_KEY"   # ใส่ API KEY ของคุณ
)

# ===== หน้า Home =====
@app.route("/")
def home():
    return {
        "status": "running",
        "project": "Bag Counter AI"
    }

# ===== นับกระสอบ =====
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_url = data["image"]

    try:
        result = CLIENT.infer(
            image_url,
            model_id="sack-c8cro/2"   # ใส่ model id ของคุณ
        )

        predictions = result.get("predictions", [])
        count = len(predictions)

        return jsonify({
            "count": count,
            "detections": predictions
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ===== Run Server =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
