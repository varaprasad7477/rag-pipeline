from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models import DocumentInfo, QueryRequest, QueryResponse
from .service import RagService

app = FastAPI(title="RAG Pipeline", version="1.0.0", description="Local-first hybrid RAG API")
service = RagService(settings)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(service.store.documents()),
        "mode": settings.llm_provider,
    }


@app.get("/api/documents", response_model=list[DocumentInfo])
def documents() -> list[dict]:
    return service.store.documents()


@app.post("/api/documents", status_code=201)
async def upload(file: Annotated[UploadFile, File()]) -> dict:
    raw = await file.read(settings.max_file_bytes + 1)
    if len(raw) > settings.max_file_bytes:
        raise HTTPException(413, "file is larger than the configured limit")
    suffix = Path(file.filename or "document.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".json", ".pdf"}:
        raise HTTPException(415, "supported formats: txt, md, csv, json, pdf")
    try:
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise HTTPException(501, "install the pdf extra: pip install -e .[pdf]") from exc
            import io

            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        else:
            text = raw.decode("utf-8")
        doc_id, chunks, created = service.ingest(file.filename or "document", text)
    except UnicodeDecodeError as exc:
        raise HTTPException(400, "text files must use UTF-8 encoding") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": doc_id, "chunks": chunks, "created": created}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    if not service.store.delete_document(doc_id):
        raise HTTPException(404, "document not found")


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return service.ask(request.question, request.top_k, request.session_id)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"model provider failed: {exc}") from exc
