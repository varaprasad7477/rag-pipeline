from pydantic import BaseModel, Field


class Source(BaseModel):
    chunk_id: str
    document: str
    excerpt: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str = Field(default="default", max_length=100)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    mode: str


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    created_at: str
