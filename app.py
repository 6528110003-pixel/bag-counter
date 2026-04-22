from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import base64

app = Flask(__name__)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ใส่_API_KEY_ของคุณ"
)

@app.route("/")
def home():
    return "AI Sack Counter Running"

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    image_bytes = file.read()

    result = client.run_workflow(
        workspace_name="s-workspace-bkeck",
        workflow_name="detect-count-and-visualize",
        images={
            "image": image_bytes
        }
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run()
