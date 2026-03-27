import logging
import os
import json
import time
import uuid
import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

import google.generativeai as genai

from CV_Data.Data import cv_system_prompt, cv_user_prompt

# ======================================================
# CONFIG
# ======================================================
UPLOAD_FOLDER = "uploads"
POPPLER_PATH = r"C:\Users\Administrator\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ======================================================
# FLASK APP
# ======================================================
app = Flask(__name__)
CORS(app)

# ======================================================
# GEMINI CONFIG
# ======================================================
GEMINI_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.08,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
)

# ======================================================
# RATE LIMIT
# ======================================================
_LAST_CALL = 0

def rate_limit(min_interval=0.7):
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < min_interval:
        time.sleep(min_interval - (now - _LAST_CALL))
    _LAST_CALL = time.time()

# ======================================================
# SAFE JSON PARSER
# ======================================================
def safe_json_load(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
    try:
        return json.loads(text)
    except Exception:
        logging.error("❌ Invalid JSON returned by Gemini")
        return {}

# ======================================================
# DOCX → PDF
# ======================================================
def safe_docx_to_pdf(input_docx, output_dir):
    result = subprocess.run(
        [
            LIBREOFFICE_PATH,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            input_docx,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError("DOCX → PDF failed")

    pdf_path = os.path.splitext(input_docx)[0] + ".pdf"

    for _ in range(10):
        if os.path.exists(pdf_path):
            return pdf_path
        time.sleep(0.3)

    raise RuntimeError("PDF not generated")

# ======================================================
# PDF → IMAGES
# ======================================================
def pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{i}.png")
            img.save(p, "PNG")
            paths.append(p)
        return paths
    except Exception as e:
        logging.error(f"PDF → image failed: {e}")
        return []

# ======================================================
# GEMINI MULTI‑PAGE + 2‑STEP EXTRACTION ✅
# ======================================================
JSON_REFORMAT_PROMPT = """
Convert the following resume text into a VALID JSON OBJECT ONLY.

Rules:
- Output ONLY JSON
- No markdown
- No explanations
- Use null if missing
- skills must be array
- educationalQualification must be array
- Past Work Experience must be:
  "Past Work Experience 1" to "Past Work Experience 4"

Resume text:
"""

def gemini_extract_multi(image_paths):
    rate_limit()

    images = []
    for path in image_paths:
        with open(path, "rb") as f:
            images.append({
                "mime_type": "image/png",
                "data": f.read()
            })

    # -------- STEP 1: RAW TEXT EXTRACTION --------
    response = model.generate_content(
        [cv_system_prompt, *images, cv_user_prompt]
    )

    if not response or not response.candidates:
        return {}

    raw_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text"):
            raw_text += part.text

    raw_text = raw_text.strip()
    if not raw_text:
        return {}

    # -------- STEP 2: CONVERT TO JSON --------
    response_json = model.generate_content(
        [JSON_REFORMAT_PROMPT + raw_text]
    )

    if not response_json or not response_json.candidates:
        return {}

    json_text = ""
    for part in response_json.candidates[0].content.parts:
        if hasattr(part, "text"):
            json_text += part.text

    return safe_json_load(json_text)

# ======================================================
# CV ENDPOINT
# ======================================================
@app.route("/cv", methods=["POST"])
def extract_cv():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf_file"]
    filename = secure_filename(file.filename)

    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    uid = str(uuid.uuid4())[:8]
    saved_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{filename}")
    file.save(saved_path)

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".docx":
            pdf_path = safe_docx_to_pdf(saved_path, UPLOAD_FOLDER)
        elif ext == ".pdf":
            pdf_path = saved_path
        else:
            return jsonify({"error": "Only PDF or DOCX supported"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    image_paths = pdf_to_images(pdf_path)
    if not image_paths:
        return jsonify({"error": "Unreadable PDF"}), 400

    logging.info(f"Processing {filename} | pages={len(image_paths)}")

    json_data = gemini_extract_multi(image_paths)
    if not json_data:
        return jsonify({"error": "Extraction failed"}), 500

    combined_data = {
        "name": json_data.get("name"),
        "phoneNumber": json_data.get("phoneNumber"),
        "email": json_data.get("email"),
        "location": json_data.get("location"),
        "maritalStatus": json_data.get("maritalStatus"),
        "presentEmployer": json_data.get("presentEmployer"),
        "presentJobTitle": json_data.get("presentJobTitle"),
        "presentJobDescription": json_data.get("presentJobDescription"),
        "presentJobStartingDate": json_data.get("presentJobStartingDate"),
        "totalExperience": json_data.get("totalExperience"),
        "dateOfBirth": json_data.get("dateOfBirth"),
        "yearGap": json_data.get("yearGap"),
        "skills": list(set(json_data.get("skills", []))),
        "pastWorkExperience": [],
        "educationalQualification": [],
    }

    for i in range(1, 5):
        exp = json_data.get(f"Past Work Experience {i}")
        if isinstance(exp, dict) and any(exp.values()):
            combined_data["pastWorkExperience"].append(exp)

    for edu in json_data.get("educationalQualification", []):
        if isinstance(edu, dict) and any(edu.values()):
            combined_data["educationalQualification"].append(edu)

    return jsonify({"extracted_data": [combined_data]}), 200

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="157.20.51.172", port=5000, debug=False)
