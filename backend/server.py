import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})



@app.route("/")
def home():
    return {"message": "Deepfake detection API is runnings"}

@app.route("/upload", methods=['POST'])
def upload():
    """
    Handles image upload and runs prediction.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    #save uploaded file --> WORKING
    img_path = os.path.join("./uploads", file.filename)
    file.save(img_path)

    try:
        # Perform prediction via subprocess
        result = subprocess.check_output(
            ["python", "./model/predict.py", img_path],
            stderr=subprocess.STDOUT,  # Capture errors as part of the output
            text=True,  # Automatically decode to string
        )
        # Cleanup the saved file
        os.remove(img_path)        
        return jsonify({"message": result.strip()})
    
    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess execution
        error_message = e.output.strip()
        return jsonify({"error": f"Subprocess failed: {error_message}"}), 500

        # return jsonify({"message": f"The image is: {result}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    


if __name__ == "__main__":
    # Ensure the uploads folder exists
    os.makedirs("./uploads", exist_ok=True)

    # Start the Flask app
    app.run(debug=True)
