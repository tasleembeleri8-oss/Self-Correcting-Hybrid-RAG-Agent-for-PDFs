import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .documents import MAX_UPLOAD_BYTES, PdfError, add_pdf, get_document, safe_display_name
from .graph import agent
from .retriever import DocumentNotFound
from .schemas import AskRequest, AskResponse, UploadResponse

logger = logging.getLogger("rag")
app = FastAPI(title="Self-Correcting PDF RAG Assistant")
STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Receive a PDF, index it privately, and return a document_id for later questions."""
    name = safe_display_name(file.filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Read at most limit+1 bytes: enough to detect "too big" without loading a huge file.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is too large (limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    try:
        # Parsing + embedding is slow, CPU-bound work. Running it in a thread keeps the
        # server responsive for other requests while it works.
        doc = await run_in_threadpool(add_pdf, data, name)
    except PdfError as e:
        raise HTTPException(e.status_code, e.message)
    except Exception:
        logger.exception("Unexpected upload failure")
        raise HTTPException(500, "Something went wrong while processing the PDF.")
    return UploadResponse(document_id=doc.id, filename=doc.filename, pages=doc.pages, chunks=len(doc.chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer a question using ONLY the document identified by document_id."""
    if get_document(req.document_id) is None:
        raise HTTPException(404, "Document not found or expired. Please upload the PDF again.")
    try:
        result = await agent.ainvoke(
            {"doc_id": req.document_id, "question": req.question, "query": req.question, "retries": 0}
        )
    except DocumentNotFound:
        raise HTTPException(404, "Document not found or expired. Please upload the PDF again.")
    except Exception:
        logger.exception("Agent failure")
        raise HTTPException(503, "The AI service failed or is rate-limited. Please try again in a moment.")
    return AskResponse(
        answer=result["answer"], retries=result["retries"], passages=result.get("docs", [])
    )
