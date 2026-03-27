import logging
import os
import json
import time
import uuid
import subprocess
import re

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
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
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
# UTILS
# ======================================================
_LAST_CALL = 0


def rate_limit(min_interval=0.7):
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < min_interval:
        time.sleep(min_interval - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def normalize_json(text: str) -> dict:
    """
    Strict JSON parsing only
    """
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


def safe_docx_to_pdf(input_docx, output_dir):
    if not os.path.exists(input_docx):
        raise FileNotFoundError("DOCX not found")

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
        raise RuntimeError("LibreOffice DOCX → PDF failed")

    pdf_path = os.path.splitext(input_docx)[0] + ".pdf"

    # wait until PDF is actually written
    for _ in range(10):
        if os.path.exists(pdf_path):
            return pdf_path
        time.sleep(0.3)

    raise RuntimeError("PDF not generated")


def pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_p{i}.png")
            img.save(path, "PNG")
            paths.append(path)
        return paths
    except Exception as e:
        logging.error(f"❌ PDF → Image failed: {e}")
        return []


# ======================================================
# GEMINI MULTI-PAGE EXTRACTION (FIXED)
# ======================================================
def gemini_extract_multi(image_paths, system_prompt, user_prompt):
    rate_limit()

    images_payload = []
    for p in image_paths:
        with open(p, "rb") as f:
            images_payload.append({
                "mime_type": "image/png",
                "data": f.read()
            })

    response = model.generate_content(
        [system_prompt, *images_payload, user_prompt]
    )

    if not response or not response.candidates:
        logging.warning("⚠ Gemini returned no candidates")
        return {}

    content = response.candidates[0].content
    if not content or not content.parts:
        return {}

    raw_text = "".join(
        p.text for p in content.parts if hasattr(p, "text") and p.text
    ).strip()

    if not raw_text:
        return {}

    return normalize_json(raw_text)


# ======================================================
# CV API
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
        return jsonify({"error": f"File conversion failed: {e}"}), 400

    image_paths = pdf_to_images(pdf_path)
    if not image_paths:
        return jsonify({"error": "Unable to read PDF pages"}), 400

    logging.info(f"Processing {filename} | pages={len(image_paths)}")

    json_data = gemini_extract_multi(
        image_paths, cv_system_prompt, cv_user_prompt
    )

    if not isinstance(json_data, dict):
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
            combined_data["pastWorkExperience"].append({
                "jobTitle": exp.get("jobTitle"),
                "company": exp.get("company"),
                "duration": exp.get("duration"),
                "description": exp.get("description"),
            })

    if isinstance(json_data.get("educationalQualification"), list):
        for edu in json_data["educationalQualification"]:
            if isinstance(edu, dict) and any(edu.values()):
                combined_data["educationalQualification"].append({
                    "level": edu.get("level"),
                    "institution": edu.get("institution"),
                    "year": edu.get("year"),
                    "details": edu.get("details"),
                })

    return jsonify({"extracted_data": [combined_data]}), 200


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="157.20.51.172", port=5000, debug=False)
