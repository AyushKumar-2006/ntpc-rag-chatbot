import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import chromadb
import fitz
from embedder import embed_texts

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "reports")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHROMA_PATH = "/Users/ayushkumar/ntpc-rag-chatbot/backend/chroma"

CHUNK_SIZE = 400
OVERLAP = 50  # ✅ Overlap added

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
        i += size - overlap  # ✅ Slide window with overlap
    return chunks

def extract_pdf_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    filename = os.path.basename(pdf_path)
    total_text = 0

    for page_num, page in enumerate(doc):
        text = page.get_text()
        total_text += len(text)
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            chunks.append({
                "text": chunk,
                "meta": {
                    "source": filename,
                    "title": filename,
                    "url": filename,
                    "section": f"page_{page_num + 1}",
                    "page": str(page_num + 1)
                }
            })

    # ✅ Warn if PDF gave no text (scanned PDF)
    if total_text < 100:
        print(f"  ⚠️ WARNING: {filename} has no extractable text — may be scanned!")

    return chunks
       

def main():
    all_chunks, all_ids, all_metas = [], [], []
    idx = 0

    # ✅ Index website pages (pages.json)
    pages_file = os.path.join(RAW_DIR, "pages.json")
    if os.path.exists(pages_file):
        with open(pages_file) as f:
            pages = json.load(f)
        for page in pages:
            texts = page.get("paragraphs", []) + page.get("list_items", [])
            full_text = " ".join(texts)
            for chunk in chunk_text(full_text):
                all_chunks.append(chunk)
                all_ids.append(f"doc_{idx}")
                all_metas.append({
                    "source": page.get("url", "website"),
                    "title": page.get("title", "NTPC Website"),
                    "url": page.get("url", ""),
                    "section": page.get("section", "General"),
                    "page": "0"
                })
                idx += 1
        print(f"✅ Website pages indexed: {idx} chunks")
    else:
        print("⚠️ pages.json not found, skipping website data")

    # ✅ Index all PDFs
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"\n📄 Found {len(pdf_files)} PDFs to index...")

    for pdf_file in sorted(pdf_files):
        print(f"Indexing PDF: {pdf_file}")
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        chunks = extract_pdf_chunks(pdf_path)
        print(f"  → {len(chunks)} chunks extracted")
        for item in chunks:
            all_chunks.append(item["text"])
            all_ids.append(f"doc_{idx}")
            all_metas.append(item["meta"])
            idx += 1

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Creating embeddings...")
    embeddings = embed_texts(all_chunks)

    # ✅ Save to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("ntpc_content")
        print("Old collection deleted.")
    except:
        pass

    collection = client.create_collection("ntpc_content")

    BATCH_SIZE = 5000
    for i in range(0, len(all_chunks), BATCH_SIZE):
        collection.add(
            documents=all_chunks[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            ids=all_ids[i:i+BATCH_SIZE],
            metadatas=all_metas[i:i+BATCH_SIZE]
        )
        print(f"Added batch {i//BATCH_SIZE + 1}")

    print(f"\n✅ Done! {collection.count()} chunks saved to ChromaDB.")

if __name__ == "__main__":
    main()