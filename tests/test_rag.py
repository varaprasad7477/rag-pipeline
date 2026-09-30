from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.chunking import chunk_text
from app.config import Settings
from app.main import app
from app.service import RagService
from app.store import Store


def service(tmp_path: Path) -> RagService:
    config = Settings(data_dir=tmp_path, chunk_size=120, chunk_overlap=20)
    return RagService(config, Store(tmp_path / "test.db"))


def test_chunking_keeps_overlap_and_limits_size() -> None:
    chunks = chunk_text("First sentence. " * 30, size=100, overlap=15)
    assert len(chunks) > 2
    assert all(len(chunk.text) <= 115 for chunk in chunks)


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    rag = service(tmp_path)
    first = rag.ingest("guide.md", "Python uses indentation. FastAPI builds web APIs.")
    second = rag.ingest("copy.md", "Python uses indentation. FastAPI builds web APIs.")
    assert first[2] is True
    assert second[2] is False
    assert len(rag.store.documents()) == 1


def test_hybrid_retrieval_returns_grounded_source(tmp_path: Path) -> None:
    rag = service(tmp_path)
    rag.ingest("facts.txt", "The Atlas launch window opens on Friday. The backup day is Sunday.")
    result = rag.ask("When does the Atlas launch window open?")
    assert "Friday" in result.answer
    assert result.sources[0].document == "facts.txt"
    assert "[1]" in result.answer


def test_api_health() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pdf_upload_is_available_in_standard_install(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf_path.open("wb") as output:
        writer.write(output)

    monkeypatch.setattr("app.main.service.ingest", lambda name, text: ("pdf-doc", 1, True))
    with pdf_path.open("rb") as uploaded:
        response = TestClient(app).post(
            "/api/documents", files={"file": ("sample.pdf", uploaded, "application/pdf")}
        )

    assert response.status_code == 201
    assert response.json() == {"id": "pdf-doc", "chunks": 1, "created": True}
