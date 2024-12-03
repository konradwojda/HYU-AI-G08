from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model.predict import predict_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route("/")
def home():
    return {"message": "Deepfake detection API is running."}


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles image upload and runs prediction without saving the file to disk.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        result = predict_image(
            image_file=file.stream, model_path="deepfake_detector.pth"
        )
        return jsonify({"message": result.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("./uploads", exist_ok=True)
    app.run(debug=True)
