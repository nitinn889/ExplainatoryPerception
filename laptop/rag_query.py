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
from datetime import datetime
from typing import Any, Callable, Optional

from laptop.memory_store import MemoryStore

Synthesizer = Callable[[str, list[dict[str, Any]]], str]


def _short_time(timestamp: Any) -> str:
    """'2026-10-08 06:20:06.810030+00:00' -> '06:20:06'."""
    text = str(timestamp)
    try:
        return datetime.fromisoformat(text).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return text


def extractive_synthesize(question: str, episodes: list[dict[str, Any]]) -> str:
    """Template-based, strictly grounded synthesis.

    Every word of the answer comes from the retrieved episodes' own captions and
    metadata, so there is nothing for a model to invent. The best-matching
    episode is stated as a direct answer first - "where is X?" deserves a
    sentence, not a ranked list - and the rest follow as supporting
    observations, with the retrieval order left visible.
    """
    if not episodes:
        return "I don't have any recorded observations relevant to that question."

    def text(episode: dict[str, Any]) -> str:
        """The clean sentence, not the embedded text with its retrieval hints."""
        return episode.get("display_caption") or episode["caption"]

    best = episodes[0]
    location = best.get("location_tag") or "an unrecorded location"
    seen_at = _short_time(best.get("timestamp", "an unknown time"))
    lead = f"{text(best)} Last seen at {location}, {seen_at}."

    if str(best.get("event_type")) == "moved":
        lead += " That was a change of location from where I had previously recorded it."

    count = best.get("observation_count")
    if isinstance(count, int) and count > 1:
        lead += f" I have logged that {count} times."

    if len(episodes) == 1:
        return lead

    others = "\n".join(
        f"- {text(ep)} ({ep.get('location_tag', '?')}, {_short_time(ep.get('timestamp', '?'))})"
        for ep in episodes[1:]
    )
    return f"{lead}\n\nOther observations that matched:\n{others}"


def anthropic_synthesize(question: str, episodes: list[dict[str, Any]]) -> str:
    """Optional LLM-backed synthesis. Requires ANTHROPIC_API_KEY and the
    `anthropic` package (not a hard dependency of this module)."""
    if not episodes:
        return "I don't have any recorded observations relevant to that question."

    import anthropic

    client = anthropic.Anthropic()
    context = "\n".join(
        f"- {ep.get('display_caption') or ep['caption']} "
        f"(location: {ep.get('location_tag')}, at {ep.get('timestamp')})"
        for ep in episodes
    )
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
