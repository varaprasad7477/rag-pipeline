import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    position: int


def normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_text(text: str, size: int = 900, overlap: int = 140) -> list[Chunk]:
    """Split on paragraph/sentence boundaries while retaining a small overlap."""
    text = normalize(text)
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""
    for piece in pieces:
        while len(piece) > size:
            if buffer:
                chunks.append(Chunk(buffer, len(chunks)))
                buffer = buffer[-overlap:]
            cut = piece.rfind(" ", 0, size - len(buffer))
            cut = cut if cut > 0 else size - len(buffer)
            candidate = f"{buffer} {piece[:cut]}".strip()
            chunks.append(Chunk(candidate, len(chunks)))
            buffer, piece = candidate[-overlap:], piece[cut:].strip()
        candidate = f"{buffer} {piece}".strip()
        if len(candidate) > size and buffer:
            chunks.append(Chunk(buffer, len(chunks)))
            buffer = f"{buffer[-overlap:]} {piece}".strip()
        else:
            buffer = candidate
    if buffer:
        chunks.append(Chunk(buffer, len(chunks)))
    return chunks
