"""Tiny, dependency-free retrieval evaluation. Run with: python -m evals.run"""

import tempfile
from pathlib import Path

from app.config import Settings
from app.service import RagService

CASES = [
    (
        "handbook.md",
        "Employees receive 24 days of annual leave. Expense reports are due by Friday.",
        "How much annual leave?",
        "24 days",
    ),
    (
        "ops.md",
        "Production deploys require two reviewers. Rollbacks use the previous container tag.",
        "How many reviewers for production?",
        "two reviewers",
    ),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        service = RagService(Settings(data_dir=Path(directory)))
        for name, text, _, _ in CASES:
            service.ingest(name, text)
        passed = 0
        for _, _, question, expected in CASES:
            result = service.ask(question)
            ok = expected.lower() in result.answer.lower()
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'} | {question} | {result.answer}")
        print(f"\nGrounded answer accuracy: {passed}/{len(CASES)} ({passed / len(CASES):.0%})")


if __name__ == "__main__":
    main()
