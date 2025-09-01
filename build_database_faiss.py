# build_database_faiss.py

import os
import fitz  # PyMuPDF
import re
from tqdm import tqdm
import torch

# --- THE DEFINITIVE FIX: Import the 'Document' class ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# =========================
# CONFIG
# =========================
PDF_PATH = "Bharatiya_Nyaya_Sanhita_2023.pdf"
FAISS_INDEX_DIR = "faiss_index" # The new database folder name
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# =========================
# INITIALIZATION
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")
embedding_model = SentenceTransformerEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': device})

# =========================
# FUNCTIONS
# =========================
def load_pdf_text(pdf_path):
    print(f"📄 Loading PDF text from: {pdf_path}")
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in tqdm(doc, desc="Extracting pages"):
            full_text += page.get_text("text", sort=True) + "\n"
    print("✅ PDF loaded.")
    return full_text

def split_text_into_sections(text):
    print("✂️ Splitting document into discrete legal sections...")
    pattern = re.compile(r'^\d{1,3}[A-Z]?\.\s', re.MULTILINE)
    matches = list(pattern.finditer(text))
    
    if not matches:
        print("❌ FATAL: Could not find any sections.")
        return []

    sections = []
    for i in range(len(matches) - 1):
        start_index = matches[i].start()
        end_index = matches[i+1].start()
        section_text = re.sub(r'\s+', ' ', text[start_index:end_index]).strip()
        sections.append(section_text)
    sections.append(re.sub(r'\s+', ' ', text[matches[-1].start():]).strip())
    
    valid_sections = [s for s in sections if len(s) > 50 and re.match(r'^\d{1,3}', s)]
    print(f"✅ Found {len(valid_sections)} valid sections.")
    return valid_sections

def create_documents_with_metadata(sections):
    """Creates a list of LangChain Document objects with metadata."""
    documents = []
    for section_text in sections:
        section_match = re.match(r'(\d{1,3}[A-Z]?)', section_text)
        section_id = f"Section {section_match.group(1)}" if section_match else "Unknown"
        metadata = {"section": section_id, "source_doc": PDF_PATH}
        documents.append(Document(page_content=section_text, metadata=metadata))
    return documents

def build_faiss_index(documents):
    if not documents:
        print("❌ No documents to process. Halting build.")
        return

    print("🔍 Generating embeddings and building FAISS index...")
    # Add instruction prefix for better retrieval performance
    texts_to_embed = ["Represent the law document for retrieval: " + doc.page_content for doc in documents]
    
    vector_store = FAISS.from_texts(
        texts=texts_to_embed, 
        embedding=embedding_model, 
        metadatas=[doc.metadata for doc in documents]
    )
    
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"✅ FAISS index built successfully with {len(documents)} sections and saved to '{FAISS_INDEX_DIR}'.")

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":
    full_text = load_pdf_text(PDF_PATH)
    sections = split_text_into_sections(full_text)
    documents = create_documents_with_metadata(sections)
    build_faiss_index(documents)