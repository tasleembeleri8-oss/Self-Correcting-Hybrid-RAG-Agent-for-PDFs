import pickle
from pathlib import Path

import chromadb
from pypdf import PdfReader

client = chromadb.PersistentClient(path="./chroma_db")


def chunk(text, size=800, overlap=150):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def ingest(pdf_dir="data/pdfs"):
    # start fresh so re-running never creates duplicate IDs
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.get_or_create_collection("docs")

    all_chunks = []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        for page_no, page in enumerate(PdfReader(pdf).pages):
            for c in chunk(page.extract_text() or ""):
                if len(c.strip()) > 50:
                    all_chunks.append((c, {"source": pdf.name, "page": page_no + 1}))

    if not all_chunks:
        raise SystemExit("No text found. Check that data/pdfs has text-based PDFs.")

    # add in batches to stay under Chroma's batch limit
    for i in range(0, len(all_chunks), 500):
        batch = all_chunks[i:i + 500]
        col.add(
            ids=[f"c{i + j}" for j in range(len(batch))],
            documents=[c for c, _ in batch],
            metadatas=[m for _, m in batch],
        )

    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Ingested {len(all_chunks)} chunks")


if __name__ == "__main__":
    ingest()