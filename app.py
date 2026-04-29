import io
import base64
import os

import numpy as np
import pydicom
from flask import Flask, jsonify, render_template, request
from PIL import Image

from quality_model import load_quality_model, predict_quality
from segmentation_guidance import load_segmentation_model, segment_lv, compute_guidance

app = Flask(__name__)

MODEL_PATH = "quality_model_weights.pth"
SEG_MODEL_PATH = "Project1_Model_4CH.pth"

quality_model = load_quality_model(MODEL_PATH)
segmentation_model = load_segmentation_model(SEG_MODEL_PATH)


def get_first_frame(ds) -> np.ndarray:
    """Get first frame safely for grayscale or color, single- or multi-frame."""
    if "PixelData" not in ds:
        raise ValueError("DICOM has no PixelData")

    pixels = ds.pixel_array

    if pixels.ndim == 4 and pixels.shape[-1] == 3:
        return pixels[0]

    if pixels.ndim == 3 and pixels.shape[-1] != 3:
        return pixels[0]

    return pixels


def to_grayscale_8bit(pixels: np.ndarray) -> np.ndarray:
    """Convert pixels to 2D grayscale uint8."""
    if pixels.ndim == 3:
        if pixels.shape[-1] == 3:
            pixels = pixels.mean(axis=-1)
        elif pixels.shape[-1] == 4:
            pixels = pixels[..., :3].mean(axis=-1)
        else:
            raise ValueError(f"Unsupported pixel shape: {pixels.shape}")

    if pixels.ndim != 2:
        raise ValueError(f"Unsupported pixel shape: {pixels.shape}")

    pixels = pixels.astype(np.float32)
    p_min = float(pixels.min())
    p_max = float(pixels.max())

    if p_max == p_min:
        raise ValueError("Image has no intensity variation")

    norm = (pixels - p_min) / (p_max - p_min) * 255.0
    return norm.astype(np.uint8)


def load_uploaded_image(file_storage) -> np.ndarray:
    """
    Accept DICOM, PNG, JPG, JPEG.
    Returns one image/frame as a numpy array.
    """
    filename = file_storage.filename.lower()
    ext = os.path.splitext(filename)[1]

    if ext == ".dcm":
        ds = pydicom.dcmread(file_storage, force=True)
        return get_first_frame(ds)

    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file_storage.stream)
        return np.array(img)

    raise ValueError("Unsupported file type. Please upload a .dcm, .png, .jpg, or .jpeg file.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        pixels = load_uploaded_image(f)
        img8 = to_grayscale_8bit(pixels)

        pil_img = Image.fromarray(img8, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        raw_score, _ = predict_quality(img8, quality_model)
        score = round(raw_score * 100, 1)

        response = {
            "file_name": f.filename,
            "score": score,
            "png_base64": "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8"),
        }

        if score >= 95:
            response.update({
                "assessment": "Good",
                "segmentation_status": "Not required",
                "guidance": "No further adjustment needed.",
                "done": True,
                "current_com": None,
                "target_com": None,
                "angle_deg": None,
                "overlay_base64": None,
            })
        else:
            mask = segment_lv(img8, segmentation_model)
            guidance_result = compute_guidance(img8, mask)

            response.update({
                "assessment": "Needs Improvement",
                "segmentation_status": guidance_result["segmentation_status"],
                "guidance": guidance_result["guidance"],
                "done": False,
                "current_com": guidance_result["current_com"],
                "target_com": guidance_result["target_com"],
                "angle_deg": guidance_result["angle_deg"],
                "overlay_base64": guidance_result["overlay_base64"],
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True)