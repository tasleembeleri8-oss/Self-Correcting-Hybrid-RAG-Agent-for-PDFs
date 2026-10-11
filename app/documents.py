"""Per-document indexing.

Pipeline for every uploaded PDF:
    bytes -> validate -> extract text per page -> chunk -> embed (Chroma) + BM25

KEY DESIGN RULE: every document gets its OWN Chroma collection and its OWN BM25 index.
There is no global index. Retrieval always starts from a document_id, so a question about
document A can never see chunks from document B.

Storage is in memory (temporary) on purpose: simple, safe for a portfolio demo, and wiped on
restart. Old documents are evicted automatically (see MAX_DOCUMENTS and DOCUMENT_TTL_SECONDS).
"""
import io
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# ---- limits (protect the server from huge or abusive uploads) ----
MAX_UPLOAD_BYTES = 10 * 1024 * 1024      # 10 MB
MAX_PAGES = 200
MAX_CHUNKS = 2000
MAX_DOCUMENTS = 20                        # keep at most this many documents in memory
DOCUMENT_TTL_SECONDS = 2 * 60 * 60        # forget documents after 2 hours

CHUNK_SIZE = 800                          # characters per chunk
CHUNK_OVERLAP = 150                       # characters shared between neighbouring chunks

# None = Chroma's default embedding model (all-MiniLM-L6-v2, runs locally via ONNX).
# Kept as a variable so the embedding model can be swapped (or faked in tests) in one place.
EMBEDDING_FUNCTION: Any = None


class PdfError(Exception):
    """A problem with the uploaded file that is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class IndexedDocument:
    id: str
    filename: str
    pages: int
    chunks: list          # list of (text, metadata) - position in this list == chunk number
    bm25: BM25Okapi       # keyword index for THIS document only
    collection: Any       # Chroma collection (vector index) for THIS document only
    created_at: float


# In-memory, in-process store. One Chroma client, but one *collection* per document.
_client = chromadb.EphemeralClient()
_documents: dict = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ text helpers
def tokenize(text: str) -> list:
    """Lowercase word tokens. BM25 compares these exact words between query and chunk."""
    return re.findall(r"\w+", text.lower())


def safe_display_name(filename: str) -> str:
    """Name shown in the UI only. It is NEVER used to build a file path (we never write
    uploads to disk), so path traversal like '../../x.pdf' has nothing to attack."""
    base = (filename or "document.pdf").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:80].strip()
    return base or "document.pdf"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks.

    Why chunk? Embedding models and LLM prompts work on small pieces, and retrieval needs to
    return just the relevant piece, not a whole page.
    Why overlap? A fact that sits on a chunk boundary would otherwise be cut in half.
    Improvement over a fixed cut: we end a chunk at the last whitespace so words are not split.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:      # only snap back if it doesn't make the chunk tiny
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # step forward, keeping `overlap` characters
    return chunks


# ------------------------------------------------------------------ PDF parsing
def extract_pages(data: bytes) -> list:
    """Return the text of each page. The PDF is only PARSED, never executed."""
    # Real PDFs start with '%PDF-'. Checking the bytes (not just the file extension) stops
    # someone renaming a .exe or .txt to .pdf.
    if b"%PDF-" not in data[:1024]:
        raise PdfError("This file is not a valid PDF.", 400)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError("This PDF is password-protected. Remove the password and try again.", 422)
        if len(reader.pages) > MAX_PAGES:
            raise PdfError(f"This PDF has too many pages (limit is {MAX_PAGES}).", 413)
        return [(page.extract_text() or "") for page in reader.pages]
    except PdfError:
        raise
    except Exception:
        raise PdfError("This PDF could not be read. It may be corrupted.", 422)


def build_chunks(pages: list) -> list:
    """Chunk page by page so every chunk remembers which page it came from."""
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        for piece in chunk_text(text):
            if len(piece) > 50:              # skip near-empty fragments (page numbers, headers)
                chunks.append((piece, {"page": page_no}))
    return chunks


# ------------------------------------------------------------------ indexing
def index_chunks(filename: str, page_count: int, chunks: list) -> IndexedDocument:
    """Create this document's private vector index (Chroma) and keyword index (BM25)."""
    doc_id = uuid.uuid4().hex                # 128-bit random id: cannot be guessed
    kwargs = {"embedding_function": EMBEDDING_FUNCTION} if EMBEDDING_FUNCTION else {}
    collection = _client.create_collection(f"doc_{doc_id}", **kwargs)
    try:
        # Chroma turns each chunk into an embedding (a vector of numbers capturing meaning)
        # when we add it. Batches keep memory use and request size small.
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            collection.add(
                ids=[f"c{i + j}" for j in range(len(batch))],
                documents=[text for text, _ in batch],
                metadatas=[meta for _, meta in batch],
            )
        # BM25 = classic keyword ranking. Built once from the same chunks, kept in memory.
        bm25 = BM25Okapi([tokenize(text) for text, _ in chunks])
    except Exception:
        _drop_collection(collection.name)
        raise PdfError("Processing failed while indexing the document. Please try again.", 500)

    doc = IndexedDocument(doc_id, filename, page_count, chunks, bm25, collection, time.time())
    with _lock:
        _documents[doc_id] = doc
        _evict_locked()
    return doc


def add_pdf(data: bytes, filename: str) -> IndexedDocument:
    """Full upload pipeline: validate -> extract -> chunk -> index."""
    pages = extract_pages(data)
    chunks = build_chunks(pages)
    if not chunks:
        raise PdfError("No readable text found. The PDF may be a scan or contain only images.", 422)
    if len(chunks) > MAX_CHUNKS:
        raise PdfError("This PDF is too large to process. Try a shorter one.", 413)
    return index_chunks(safe_display_name(filename), len(pages), chunks)


# ------------------------------------------------------------------ lookup + cleanup
def get_document(doc_id: str):
    """Return the document, or None if the id is unknown or expired."""
    with _lock:
        doc = _documents.get(doc_id)
        if doc and time.time() - doc.created_at > DOCUMENT_TTL_SECONDS:
            _drop_locked(doc_id)
            return None
        return doc


def _drop_collection(name: str) -> None:
    try:
        _client.delete_collection(name)
    except Exception:
        pass


def _drop_locked(doc_id: str) -> None:
    doc = _documents.pop(doc_id, None)
    if doc:
        _drop_collection(doc.collection.name)


def _evict_locked() -> None:
    """Remove expired documents, then the oldest ones if there are too many."""
    now = time.time()
    for doc in [d for d in _documents.values() if now - d.created_at > DOCUMENT_TTL_SECONDS]:
        _drop_locked(doc.id)
    while len(_documents) > MAX_DOCUMENTS:
        oldest = min(_documents.values(), key=lambda d: d.created_at)
        _drop_locked(oldest.id)
