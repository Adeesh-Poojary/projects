import os
import imaplib
import email
from pdfminer.high_level import extract_text as extract_pdf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import shutil
import pytesseract
from pdf2image import convert_from_path
import re
import logging
from docx2pdf import convert as docx2pdf_convert

# Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Email config ---
EMAIL = 'shrirao005@gmail.com'
PASSWORD = 'kqih tbqf hhla zpsx'
IMAP_SERVER = 'imap.gmail.com'

# --- Folder config ---
DOWNLOAD_FOLDER = r"C:/Users/susha/Desktop/Resume Classifier/All Files"
RESUME_FOLDER = r"C:/Users/susha/Desktop/Resume Classifier/Classified Resumes"
NOT_RESUME_FOLDER = r"C:/Users/susha/Desktop/Resume Classifier/Rejected Resumes"
MODEL_PATH = r"D:/Resume/Model Data/resume_classifier_model.keras"
TOKENIZER_PATH = r"D:/Resume/Model Data/resume_tokenizer.pkl"

# --- Clear download folder safely ---
def clear_download_folder():
    for file in os.listdir(DOWNLOAD_FOLDER):
        file_path = os.path.join(DOWNLOAD_FOLDER, file)
        try:
            os.remove(file_path)
            logging.info(f"🗑 Deleted: {file}")
        except PermissionError:
            logging.warning(f"🔒 Skipped (in use): {file}")
        except Exception as e:
            logging.error(f"⚠ Failed to delete {file}: {e}")
    logging.info("🧹 Cleared download folder.")

# --- Create folders if missing ---
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(RESUME_FOLDER, exist_ok=True)
os.makedirs(NOT_RESUME_FOLDER, exist_ok=True)

# --- Load model and tokenizer ---
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
MAXLEN = 1000

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Text extraction with OCR fallback ---
def extract_text_from_file(path):
    try:
        if path.lower().endswith('.pdf'):
            text = extract_pdf(path).strip()
            if text:
                logging.info(f"📄 Extracted text from {os.path.basename(path)}")
                return preprocess_text(text)

            logging.info(f"🔍 Using OCR for {os.path.basename(path)}")
            images = convert_from_path(path)
            ocr_text = ''.join(pytesseract.image_to_string(img) for img in images)
            return preprocess_text(ocr_text)
    except Exception as e:
        logging.error(f"❌ Extraction failed for {path}: {e}")
    return ""

# --- Avoid overwriting files with the same name ---
def safe_copy(src_path, dest_folder):
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    counter = 1
    dest_path = os.path.join(dest_folder, base)
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_folder, f"{name} ({counter}){ext}")
        counter += 1
    shutil.copy(src_path, dest_path)
    return dest_path

# --- Connect to Gmail and download resumes ---
def connect_and_download():
    logging.info("📥 Connecting to Gmail...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')

    typ, data = mail.search(None, 'ALL')
    for num in data[0].split():
        typ, msg_data = mail.fetch(num, '(RFC822)')
        raw_email = msg_data[0][1]
        message = email.message_from_bytes(raw_email)

        for part in message.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue

            filename = part.get_filename()
            if filename and filename.lower().endswith(('.pdf', '.docx')):
                filepath = os.path.join(DOWNLOAD_FOLDER, filename)
                if os.path.exists(filepath):
                    logging.info(f"⚠ Skipped (already exists): {filename}")
                else:
                    content = part.get_payload(decode=True)
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    logging.info(f"✅ Downloaded: {filename}")

# --- Classify all files in download folder ---
def classify_files():
    logging.info("🧠 Starting classification...")

    for file in os.listdir(DOWNLOAD_FOLDER):
        original_path = os.path.join(DOWNLOAD_FOLDER, file)
        filename_lower = file.lower()

        # Convert DOCX to PDF
        if filename_lower.endswith('.docx'):
            try:
                pdf_name = os.path.splitext(file)[0] + ".pdf"
                pdf_path = os.path.join(DOWNLOAD_FOLDER, pdf_name)
                docx2pdf_convert(original_path, pdf_path)
                logging.info(f"📄 Converted DOCX to PDF: {pdf_name}")
                file_path = pdf_path
            except Exception as e:
                logging.error(f"❌ Failed to convert {file} to PDF: {e}")
                continue
        elif filename_lower.endswith('.pdf'):
            file_path = original_path
        else:
            logging.warning(f"⏭ Skipped unsupported file: {file}")
            continue

        logging.info(f"\n📄 Processing: {os.path.basename(file_path)}")
        text = extract_text_from_file(file_path)

        if not text.strip():
            logging.warning(f"❌ Skipped (no content): {file}")
            continue

        seq = tokenizer.texts_to_sequences([text])
        if not any(seq[0]):
            logging.warning(f"⚠ Skipped (no valid tokens): {file}")
            continue

        padded = pad_sequences(seq, maxlen=MAXLEN)
        pred = model.predict(padded, verbose=0)[0][0]

        logging.info(f"🔍 Prediction: {pred:.4f} → {'RESUME' if pred >= 0.5 else 'NOT RESUME'}")

        if pred >= 0.5:
            saved_path = safe_copy(file_path, RESUME_FOLDER)
            logging.info(f"✅ Classified as RESUME → {os.path.basename(saved_path)}")
        else:
            saved_path = safe_copy(file_path, NOT_RESUME_FOLDER)
            logging.info(f"❌ Classified as NOT RESUME → {os.path.basename(saved_path)}")

# --- Run Everything ---
if _name_ == "_main_":
    clear_download_folder()
    connect_and_download()
    classify_files()
    logging.info("🏁 All files classified.")