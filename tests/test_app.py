import io
import os

os.environ.setdefault("GROQ_API_KEY", "test-key")  # tests never call the real LLM

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import graph
from app.documents import build_chunks, chunk_text, index_chunks
from app.main import app
from app.retriever import hybrid_search
from app.schemas import Grade

client = TestClient(app)


def _blank_pdf() -> bytes:
    """A valid PDF with no text (like a scanned document)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _doc(text_blocks):
    """Index a document directly from text, skipping PDF parsing."""
    chunks = [(t, {"page": 1}) for t in text_blocks]
    return index_chunks("test.pdf", 1, chunks)


# ---------------------------------------------------------------- basic endpoints
def test_health():
    assert client.get("/health").status_code == 200


def test_home_and_docs_load():
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------- upload validation
def test_upload_rejects_non_pdf_extension():
    r = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_rejects_fake_pdf():
    r = client.post("/upload", files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 400


def test_upload_rejects_empty_file():
    r = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_upload_rejects_pdf_without_text():
    r = client.post("/upload", files={"file": ("blank.pdf", _blank_pdf(), "application/pdf")})
    assert r.status_code == 422


def test_upload_rejects_corrupted_pdf():
    r = client.post("/upload", files={"file": ("bad.pdf", b"%PDF-1.4 garbage garbage", "application/pdf")})
    assert r.status_code == 422


def test_path_traversal_filename_is_harmless():
    # Not a real PDF, so it is rejected, but the point is it never touches the file system.
    r = client.post("/upload", files={"file": ("../../etc/passwd.pdf", b"x", "application/pdf")})
    assert r.status_code == 400


# ---------------------------------------------------------------- ask validation
def test_ask_rejects_bad_document_id():
    r = client.post("/ask", json={"document_id": "../../x", "question": "What is this?"})
    assert r.status_code == 422


def test_ask_unknown_document_is_404():
    r = client.post("/ask", json={"document_id": "0" * 32, "question": "What is this?"})
    assert r.status_code == 404


def test_ask_requires_document_id():
    assert client.post("/ask", json={"question": "What is this?"}).status_code == 422


# ---------------------------------------------------------------- chunking
def test_chunking_overlaps_and_does_not_split_words():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    assert all(not c.endswith("wor") for c in chunks)       # no mid-word cuts
    assert chunks[0][-10:] in chunks[1]                      # overlap carried forward


def test_build_chunks_records_page_numbers():
    chunks = build_chunks(["a " * 100, "b " * 100])
    assert {meta["page"] for _, meta in chunks} == {1, 2}


# ---------------------------------------------------------------- isolation (the key rule)
def test_search_only_sees_its_own_document():
    a = _doc(["Apples are red fruit grown in orchards and eaten fresh by many people."] * 3)
    b = _doc(["Zebras are striped animals living on the African savanna in large herds."] * 3)
    from_a = hybrid_search(a.id, "zebra savanna", k=5)
    from_b = hybrid_search(b.id, "apple orchard", k=5)
    assert all("Apples" in t for t in from_a)
    assert all("Zebras" in t for t in from_b)


# ---------------------------------------------------------------- full flow with a fake LLM
class _FakeReply:
    def __init__(self, content):
        self.content = content


def test_ask_flow_with_retry(monkeypatch):
    doc = _doc(["The study used xylitol as the sweetening agent in every formulation tested."] * 2)
    verdicts = iter([False, True])                            # first grade weak, second good

    class FakeGrader:
        def invoke(self, _prompt):
            return Grade(relevant=next(verdicts))

    class FakeLLM:
        def invoke(self, prompt):
            return _FakeReply("xylitol sweetener" if "Rewrite" in prompt else "It used xylitol.")

    monkeypatch.setattr(graph, "grader", FakeGrader())
    monkeypatch.setattr(graph, "llm", FakeLLM())

    r = client.post("/ask", json={"document_id": doc.id, "question": "Which sweetener was used?"})
    assert r.status_code == 200
    body = r.json()
    assert body["retries"] == 1 and "xylitol" in body["answer"] and body["passages"]
