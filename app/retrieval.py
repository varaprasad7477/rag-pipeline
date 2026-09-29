import hashlib
import math
import re
from collections import Counter

DIMENSIONS = 256
WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]+")


def tokenize(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]


def embed(tokens: list[str]) -> list[float]:
    """Deterministic feature-hashing embedding; local, fast and API-key free."""
    vector = [0.0] * DIMENSIONS
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % DIMENSIONS
        sign = 1 if digest[0] & 1 else -1
        vector[index] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class HybridRetriever:
    """BM25 lexical search fused with deterministic semantic feature hashing."""

    def search(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        lengths = [len(chunk["tokens"]) for chunk in chunks]
        average_length = sum(lengths) / len(lengths)
        document_frequency = Counter(token for chunk in chunks for token in set(chunk["tokens"]))
        query_vector = embed(query_tokens)
        lexical: list[float] = []
        semantic: list[float] = []
        for chunk, length in zip(chunks, lengths, strict=True):
            counts = Counter(chunk["tokens"])
            score = 0.0
            for token, query_weight in query_counts.items():
                frequency = counts[token]
                if not frequency:
                    continue
                inverse = math.log(
                    1
                    + (len(chunks) - document_frequency[token] + 0.5)
                    / (document_frequency[token] + 0.5)
                )
                score += (
                    query_weight
                    * inverse
                    * frequency
                    * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
                )
            lexical.append(score)
            semantic.append(_cosine(query_vector, chunk["vector"]))

        def ranks(values: list[float]) -> dict[int, int]:
            return {
                index: rank
                for rank, index in enumerate(
                    sorted(range(len(values)), key=values.__getitem__, reverse=True), 1
                )
            }

        lexical_rank, semantic_rank = ranks(lexical), ranks(semantic)
        ranked = []
        for index, chunk in enumerate(chunks):
            rrf = 1 / (60 + lexical_rank[index]) + 1 / (60 + semantic_rank[index])
            ranked.append({**chunk, "score": round(rrf * 30.5, 4)})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
