import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, content_hash TEXT UNIQUE NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL, position INTEGER NOT NULL,
  text TEXT NOT NULL, tokens TEXT NOT NULL, vector TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            try:
                yield db
                db.commit()
            finally:
                db.close()

    def add_document(self, doc_id: str, name: str, content_hash: str, chunks: list[dict]) -> bool:
        with self.connect() as db:
            if db.execute(
                "SELECT 1 FROM documents WHERE content_hash=?", (content_hash,)
            ).fetchone():
                return False
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (doc_id, name, content_hash, datetime.now(UTC).isoformat()),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        c["id"],
                        doc_id,
                        c["position"],
                        c["text"],
                        json.dumps(c["tokens"]),
                        json.dumps(c["vector"]),
                    )
                    for c in chunks
                ],
            )
        return True

    def all_chunks(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT c.*, d.name document_name FROM chunks c JOIN documents d ON d.id=c.document_id"
            ).fetchall()
        return [
            {**dict(r), "tokens": json.loads(r["tokens"]), "vector": json.loads(r["vector"])}
            for r in rows
        ]

    def documents(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT d.id,d.name,d.created_at,COUNT(c.id) chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations(session_id,role,content,created_at) VALUES(?,?,?,?)",
                (session_id, role, content, datetime.now(UTC).isoformat()),
            )

    def history(self, session_id: str, limit: int = 6) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT role,content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]
