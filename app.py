"""
vehicleDetector — number-plate detection API.

Upload a vehicle photo, get the plate number back as JSON.

    GET  /            -> single-page UI
    POST /api/detect  -> multipart form, field "image"; returns JSON
    GET  /health      -> {"status": "ok", "reader_ready": bool}

Pipeline: contour-based plate localization (OpenCV) + EasyOCR.
The OCR model is initialized once, lazily, on first use — never per request.
"""

import logging
import os
import re
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------ config

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
# Downscale images wider than this for detection; boxes are mapped back
# to original coordinates before responding.
MAX_DETECT_DIM = 1600
EASYOCR_MODEL_DIR = os.environ.get(
    "EASYOCR_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".EasyOCR", "model"),
)

log = logging.getLogger("vehicle_detector")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# Sessions are unused by the JSON API; keep a key configured anyway.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-not-for-production")


# ---------------------------------------------------------- OCR singleton

_reader = None
_reader_lock = threading.Lock()


def get_reader():
    """Return the shared EasyOCR reader, initializing it once (thread-safe).

    First call downloads/loads ~100MB of weights and can take up to a minute;
    every later call is instant. Never call this per request without caching.
    """
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr  # heavy import, deferred until actually needed

                log.info("Initializing EasyOCR reader (first call is slow)...")
                _reader = easyocr.Reader(
                    ["en"], gpu=False, model_storage_directory=EASYOCR_MODEL_DIR
                )
                log.info("EasyOCR reader ready")
    return _reader


# --------------------------------------------------------------- detection

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _find_plate_boxes(gray):
    """Contour-based plate localization.

    Returns [(x, y, w, h)] in image coordinates, largest plausible first.
    Plates are wide, quadrilateral, mid-sized contours.
    """
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)  # denoise, keep edges
    edged = cv2.Canny(blurred, 30, 200)
    found = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]  # OpenCV 4.x vs 3.x

    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        area = cv2.contourArea(contour)
        if area < 0.0015 * img_area or area > 0.6 * img_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / float(h)
        if 1.8 <= aspect <= 7.0:
            candidates.append((x, y, w, h, area))

    # De-dup: drop boxes fully contained in a larger candidate.
    kept = []
    for box in sorted(candidates, key=lambda b: b[4], reverse=True):
        x, y, w, h, _ = box
        if any(
            x >= kx and y >= ky and x + w <= kx + kw and y + h <= ky + kh
            for kx, ky, kw, kh, _ in kept
        ):
            continue
        kept.append(box)
    return [(x, y, w, h) for x, y, w, h, _ in kept[:6]]


_PLATE_CHARS = re.compile(r"[^A-Z0-9]")


def _ocr_boxes(reader, gray, boxes):
    """OCR the candidate boxes; return (text, confidence, box) or None."""
    best = None
    for (x, y, w, h) in boxes[:3]:
        pad = max(2, int(0.06 * w))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if crop.shape[1] < 220:  # upscale tiny crops for the OCR model
            scale = 220.0 / crop.shape[1]
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        for _bbox, text, conf in reader.readtext(crop):
            clean = _PLATE_CHARS.sub("", text.upper())
            if len(clean) >= 3 and conf > 0.25 and (best is None or conf > best[1]):
                best = (clean, float(conf), (x, y, w, h))
    return best


def detect_plate(image_bytes: bytes) -> dict:
    """Run the full pipeline on raw image bytes. No disk I/O."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    orig_h, orig_w = img.shape[:2]

    scale = 1.0
    if max(orig_h, orig_w) > MAX_DETECT_DIM:
        scale = MAX_DETECT_DIM / max(orig_h, orig_w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = _find_plate_boxes(gray)
    log.info("plate candidates: %d", len(boxes))

    reader = get_reader()  # slow only on the very first call
    result = _ocr_boxes(reader, gray, boxes)

    def _unscale(b):
        x, y, w, h = b
        return {"x": int(x / scale), "y": int(y / scale),
                "w": int(w / scale), "h": int(h / scale)}

    response = {
        "plate_text": None,
        "confidence": 0.0,
        "boxes": [_unscale(b) for b in boxes],
        "match": None,
        "image_width": orig_w,
        "image_height": orig_h,
    }
    if result:
        text, conf, box = result
        response["plate_text"] = text
        response["confidence"] = round(conf, 3)
        response["match"] = _unscale(box)
    return response


# ------------------------------------------------------------------ routes

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    # reader_ready lets clients/UIs show a warmup state on cold starts.
    return jsonify({"status": "ok", "reader_ready": _reader is not None})


@app.post("/api/detect")
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image part in request (field name 'image')"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        return jsonify({"error": "Allowed image types: png, jpg, jpeg, webp"}), 400
    try:
        data = file.read()
        if not data:
            return jsonify({"error": "Empty file"}), 400
        result = detect_plate(data)
        log.info("detect ok plate=%s confidence=%.3f",
                 result["plate_text"], result["confidence"])
        return jsonify(result)
    except ValueError as exc:
        log.warning("detect rejected input: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception:  # never leak tracebacks to clients
        log.exception("detect failed")
        return jsonify({"error": "Detection failed"}), 500


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "File too large (max 16 MB)"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info("starting on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
