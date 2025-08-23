# database_builder_final.py

import os
import fitz  # PyMuPDF
import re
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# =========================
# CONFIG
# =========================
PDF_PATH = "Bharatiya_Nyaya_Sanhita_2023.pdf"
CHROMA_DIR = "chroma_db_Database2"
COLLECTION_NAME = "bns_sections_definitive"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# =========================
# INITIALIZATION
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")
print(f"🧠 Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME, device=device)

# =========================
# FUNCTIONS
# =========================
def load_pdf_text(pdf_path):
    """Loads the entire PDF into a single string."""
    print(f"📄 Loading PDF text from: {pdf_path}")
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in tqdm(doc, desc="Extracting pages"):
            full_text += page.get_text("text", sort=True) + "\n"
    print("✅ PDF loaded.")
    return full_text

def split_into_sections(text):
    """
    This is the definitive section splitter. It finds the start of all main sections
    and then extracts the text between each start point.
    """
    print("✂️ Splitting document into discrete legal sections (Definitive Method)...")
    
    # This regex finds the character position of a section number (e.g., "42.", "43.").
    # It looks for a number (1-3 digits), a period, and a space.
    pattern = re.compile(r'\s(\d{1,3}[A-Z]?\.\s)')
    matches = list(pattern.finditer(text))
    
    if not matches:
        print("❌ FATAL: Could not find any section markers. The PDF text might be structured unexpectedly.")
        return []

    sections = []
    # Iterate through matches to slice the text from one section start to the next
    for i in range(len(matches) - 1):
        start_index = matches[i].start()
        end_index = matches[i+1].start()
        # Clean up whitespace and remove the messy title fragments on the left
        section_text = re.sub(r'^\s*[a-zA-Z\s,]+\s*(?=\d{1,3})', '', text[start_index:end_index]).strip()
        section_text = re.sub(r'\s+', ' ', section_text).strip()
        sections.append(section_text)
    
    # Add the very last section
    last_section_text = re.sub(r'^\s*[a-zA-Z\s,]+\s*(?=\d{1,3})', '', text[matches[-1].start():]).strip()
    last_section_text = re.sub(r'\s+', ' ', last_section_text).strip()
    sections.append(last_section_text)
    
    # A final filter to ensure quality
    valid_sections = [s for s in sections if len(s) > 50 and re.match(r'^\d{1,3}', s)]
    print(f"✅ Found {len(valid_sections)} valid sections.")
    return valid_sections


def extract_metadata_from_section(section_text):
    """Extracts the main section number, which is now reliably at the start."""
    section_match = re.match(r'(\d{1,3}[A-Z]?)', section_text)
    section_id = f"Section {section_match.group(1)}" if section_match else "Unknown"
    return {"section": section_id, "source_doc": PDF_PATH}

def build_chroma_db(sections):
    if not sections:
        print("❌ No sections found. Halting build.")
        return

    print(f"🗄 Initializing ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🧹 Old collection '{COLLECTION_NAME}' deleted for a fresh start.")
    except Exception:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    print("🔍 Generating embeddings...")
    texts_to_embed = ["Represent the law document for retrieval: " + section for section in sections]
    embeddings = embedding_model.encode(texts_to_embed, batch_size=16, show_progress_bar=True)
    
    print("➕ Adding sections to the database...")
    all_metadata = [extract_metadata_from_section(text) for text in sections]
    all_ids = [f"doc_{i}" for i in range(len(sections))] # Guarantees unique IDs
    
    collection.add(
        ids=all_ids,
        embeddings=embeddings.tolist(),
        documents=sections,
        metadatas=all_metadata
    )

    print(f"✅ Database built successfully with {collection.count()} sections.")

if __name__ == "__main__":
    full_text = load_pdf_text(PDF_PATH)
    sections = split_into_sections(full_text)
    build_chroma_db(sections)