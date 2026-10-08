"""
User-facing query endpoint/CLI (Phase 8).

FastAPI app with GET /query?q=... , plus a CLI entry point:
    python -m laptop.api "where's my charger?"
"""

from __future__ import annotations

import sys

from fastapi import FastAPI

from laptop.memory_store import MemoryStore
from laptop.rag_query import answer_question
from laptop.voice import attach_spoken

app = FastAPI(title="Episodic Perception Query API")
_store = MemoryStore()


@app.get("/query")
def query(q: str, k: int = 5):
    """`answer` is the grounded text to display, `spoken` the same facts
    phrased to be read aloud — see laptop/voice.py."""
    return attach_spoken(answer_question(q, _store, k=k))


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m laptop.api "where\'s my charger?"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    result = answer_question(question, _store)
    print(result["answer"])


if __name__ == "__main__":
    main()
