import os
import json
import time
import uuid
import html
import subprocess
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader

import google.generativeai as genai

# ======================================================
# CONFIG
# ======================================================
UPLOAD_FOLDER = "uploads"
POPPLER_PATH = r"C:\poppler\Library\bin"
LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

GEMINI_API_KEY = "YOUR_API_KEY"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ======================================================
# FLASK
# ======================================================
app = Flask(__name__)
CORS(app)

# ======================================================
# GEMINI
# ======================================================
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0,
        "max_output_tokens": 4096
    }
)

# ======================================================
# HELPERS
# ======================================================
def rate_limit(delay=0.8):
    time.sleep(delay)

def safe_json_load(text):
    text = html.unescape(text.strip())
    if text.startswith("```"):
        text = text.split("```")[1]
    try:
        return json.loads(text)
    except:
        return None

def is_complete_json(text):
    return text.count("{") == text.count("}") and text.count('"') % 2 == 0

# ======================================================
# FILE HANDLING
# ======================================================
def docx_to_pdf(path):
    subprocess.run([
        LIBREOFFICE_PATH,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", UPLOAD_FOLDER,
        path
    ], timeout=90)
    return path.replace(".docx", ".pdf")

def pdf_to_text(pdf):
    try:
        reader = PdfReader(pdf)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return text.strip()
    except:
        return ""

def pdf_to_images(pdf):
    return convert_from_path(pdf, poppler_path=POPPLER_PATH)

def ocr_images(images):
    return "\n".join(pytesseract.image_to_string(img) for img in images)

# ======================================================
# GEMINI PIPELINE
# ======================================================
STRUCTURING_PROMPT = """
Convert the following resume text into VALID JSON ONLY.

Rules:
- No markdown
- No explanation
- Preserve all information
- Use null if missing
- skills: array
- education: array
- experience: array
"""

def extract_structured_data(resume_text):
    rate_limit()

    response = model.generate_content(
        STRUCTURING_PROMPT + "\n\n" + resume_text
    )

    if not response or not response.candidates:
        return {}

    raw = response.candidates[0].content.parts[0].text

    if is_complete_json(raw):
        parsed = safe_json_load(raw)
        if parsed:
            return parsed

    # 🔁 Retry continuation if truncated
    rate_limit()
    response2 = model.generate_content(
        "Continue and complete the JSON below. ONLY JSON.\n\n" + raw
    )

    if response2 and response2.candidates:
        raw2 = raw + response2.candidates[0].content.parts[0].text
        parsed = safe_json_load(raw2)
        if parsed:
            return parsed

    return {}

# ======================================================
# API
# ======================================================
@app.route("/cv", methods=["POST"])
def extract_cv():
    file = request.files.get("pdf_file")
    if not file:
        return jsonify({"error": "No file"}), 400

    name = secure_filename(file.filename)
    uid = str(uuid.uuid4())
    path = os.path.join(UPLOAD_FOLDER, uid + name)
    file.save(path)

    if path.endswith(".docx"):
        path = docx_to_pdf(path)

    # ---- Try digital text
    text = pdf_to_text(path)

    # ---- If empty → OCR
    if len(text.strip()) < 100:
        images = pdf_to_images(path)
        text = ocr_images(images)

    if not text.strip():
        return jsonify({"error": "Unable to extract text"}), 400

    data = extract_structured_data(text)

    # ✅ GUARANTEED RESPONSE
    return jsonify({
        "extracted_data": data,
        "status": "success",
        "confidence": "high" if len(text) > 500 else "medium"
    }), 200

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
