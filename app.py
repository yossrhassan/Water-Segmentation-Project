from io import BytesIO

import numpy as np
import tifffile
from flask import Flask, jsonify, request, send_file
from PIL import Image

from inference import predict_array


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Display a simple upload page for 12-channel TIFF images."""

    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Water Segmentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 720px;
                margin: 60px auto;
                padding: 24px;
                background: #f3f6fa;
                color: #172033;
            }
            .card {
                background: white;
                padding: 28px;
                border-radius: 14px;
                box-shadow: 0 4px 18px #00000012;
            }
            h1 { color: #176b87; }
            input, button {
                margin-top: 16px;
                padding: 12px;
                max-width: 100%;
            }
            button {
                display: block;
                background: #176b87;
                color: white;
                border: 0;
                border-radius: 7px;
                cursor: pointer;
            }
            iframe {
                width: 100%;
                height: 360px;
                margin-top: 20px;
                border: 1px solid #ddd;
                background: white;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Water Segmentation</h1>
            <p>
                Upload a 12-channel TIFF image of size 128 × 128.
                The model will return a binary water mask.
            </p>

            <form action="/predict" method="post"
                  enctype="multipart/form-data" target="result_frame">
                <input type="file" name="image"
                       accept=".tif,.tiff" required>
                <button type="submit">Predict Water Mask</button>
            </form>

            <h2>Prediction Result</h2>
            <iframe name="result_frame" title="Segmentation result"></iframe>
        </div>
    </body>
    </html>
    """


@app.route("/health", methods=["GET"])
def health():
    """Report that the API is running."""

    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a multispectral TIFF upload and return a binary mask PNG."""

    if "image" not in request.files:
        return jsonify({
            "error": "Upload a TIFF file using the 'image' field."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No file was selected."}), 400

    if not uploaded_file.filename.lower().endswith((".tif", ".tiff")):
        return jsonify({
            "error": "Only .tif and .tiff files are supported."
        }), 400

    try:
        file_bytes = uploaded_file.read()

        if not file_bytes:
            return jsonify({"error": "The uploaded file is empty."}), 400

        image = tifffile.imread(BytesIO(file_bytes))

        # Run segmentation using the saved 12-channel U-Net.
        mask = predict_array(image)

        if mask.shape != (128, 128):
            return jsonify({
                "error": f"Unexpected prediction shape: {mask.shape}"
            }), 500

        # Convert binary values 0 and 1 to black and white pixels.
        mask_image = Image.fromarray(
            (mask.astype(np.uint8) * 255),
            mode="L"
        )

        output_buffer = BytesIO()
        mask_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return send_file(
            output_buffer,
            mimetype="image/png",
            as_attachment=False,
            download_name="water_mask.png"
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        app.logger.exception("Prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)