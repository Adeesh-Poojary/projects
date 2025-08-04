import fitz  # PyMuPDF
import spacy

# Load trained model
nlp = spacy.load("resume_ner_model")

# Load PDF
pdf_path = "C:/Users/Lokesh/Desktop/ISOLDE MERCER.txt"  # <-- change this to your actual file
doc = fitz.open(pdf_path)

# Extract text from all pages
text = ""
for page in doc:
    text += page.get_text()

print("📄 Resume Text Extracted:")
print(text[:300], "\n...")  # show preview

# Run through NER model
spacy_doc = nlp(text)

print("\n🧠 Extracted Entities:")
for ent in spacy_doc.ents:
    print(f"{ent.text} → {ent.label_}")
