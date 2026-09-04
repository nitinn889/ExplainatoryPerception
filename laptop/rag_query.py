"""
RAG query answering (Phase 8).

Given a user question, retrieves top-k relevant episodes from
memory_store.py and synthesizes a grounded natural-language answer.

Default synthesis is extractive/template-based (quotes the retrieved
captions directly) so answers are guaranteed grounded with zero
hallucination risk and no external API dependency. Pass a different
`synthesize` callable (e.g. one that calls out to an LLM) to get more
natural prose — see `anthropic_synthesize` below for a drop-in example that
activates automatically when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from laptop.memory_store import MemoryStore

Synthesizer = Callable[[str, list[dict[str, Any]]], str]


def extractive_synthesize(question: str, episodes: list[dict[str, Any]]) -> str:
    if not episodes:
        return "I don't have any recorded observations relevant to that question."

    lines = []
    for ep in episodes:
        loc = ep.get("location_tag", "an unknown location")
        ts = ep.get("timestamp", "an unknown time")
        lines.append(f"- {ep['caption']} (location: {loc}, observed at {ts})")
    return "Based on what I've observed:\n" + "\n".join(lines)


def anthropic_synthesize(question: str, episodes: list[dict[str, Any]]) -> str:
    """Optional LLM-backed synthesis. Requires ANTHROPIC_API_KEY and the
    `anthropic` package (not a hard dependency of this module)."""
    if not episodes:
        return "I don't have any recorded observations relevant to that question."

    import anthropic

    client = anthropic.Anthropic()
    context = "\n".join(f"- {ep['caption']} (location: {ep.get('location_tag')}, at {ep.get('timestamp')})" for ep in episodes)
    prompt = (
        "Answer the question using ONLY the observations below. "
        "Do not state anything not supported by them.\n\n"
        f"Observations:\n{context}\n\nQuestion: {question}"
    )
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def answer_question(
    question: str,
    store: MemoryStore,
    k: int = 5,
    synthesize: Optional[Synthesizer] = None,
) -> dict[str, Any]:
    """Retrieve top-k episodes for `question` and synthesize a grounded answer."""
    episodes = store.search(question, k=k)

    if synthesize is None:
        synthesize = anthropic_synthesize if os.environ.get("ANTHROPIC_API_KEY") else extractive_synthesize

    return {
        "question": question,
        "answer": synthesize(question, episodes),
        "episodes": episodes,
    }
