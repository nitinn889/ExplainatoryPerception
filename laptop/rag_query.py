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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for the type hint
    # Imported lazily: memory_store pulls in ChromaDB, and answer_question only
    # needs something with a .search(query, k) method. Keeping this out of the
    # runtime imports lets the query path work against any store - including the
    # in-process fallback the demo uses when ChromaDB is not installed.
    from laptop.memory_store import MemoryStore

Synthesizer = Callable[[str, List[Dict[str, Any]]], str]


def _parse_time(timestamp: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(timestamp))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _short_time(timestamp: Any) -> str:
    """'2026-10-08 06:20:06.810030+00:00' -> '11:50:06' in this machine's local time."""
    parsed = _parse_time(timestamp)
    return parsed.astimezone().strftime("%H:%M:%S") if parsed else str(timestamp)


def _ago(timestamp: Any, now: Optional[datetime] = None) -> str:
    """'just now', '42 s ago', '5 min ago', '3 h ago' - or '' if unknown."""
    parsed = _parse_time(timestamp)
    if parsed is None:
        return ""
    seconds = ((now or datetime.now(timezone.utc)) - parsed).total_seconds()
    if seconds < 0 or seconds > 30 * 86400:
        return ""
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)} s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} days ago"


def _most_recent_sighting(question: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Among the retrieved episodes that contain an object named in the
    question, the latest one - "where is X?" is about where X is now, not
    where it matched best. Falls back to the best match."""
    asked = question.lower()
    named = [
        ep for ep in episodes
        if any(obj.strip() and obj.strip().lower() in asked for obj in str(ep.get("objects", "")).split(","))
    ]
    if not named:
        return episodes[0]
    return max(named, key=lambda ep: _parse_time(ep.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))


def _subject(episode: Dict[str, Any]) -> str:
    return str(episode.get("objects", "")).split(",")[0].strip().lower()


def _previous_sighting(missing: Dict[str, Any], episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The latest episode, before `missing`, in which its object was actually seen."""
    subject = _subject(missing)
    missing_at = _parse_time(missing.get("timestamp"))
    seen = [
        ep for ep in episodes
        if ep is not missing
        and str(ep.get("event_type")) != "missing"
        and subject in [o.strip().lower() for o in str(ep.get("objects", "")).split(",")]
        and (missing_at is None or (_parse_time(ep.get("timestamp")) or missing_at) <= missing_at)
    ]
    # Prefer sightings where it was the thing being looked at, not a bystander.
    as_subject = [ep for ep in seen if _subject(ep) == subject]
    seen = as_subject or seen
    return max(seen, key=lambda ep: _parse_time(ep.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)) if seen else None


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

    best = _most_recent_sighting(question, episodes)
    location = best.get("location_tag") or "an unrecorded location"
    seen_at = _short_time(best.get("timestamp", "an unknown time"))
    ago = _ago(best.get("timestamp"))
    when = f"{seen_at}{f' ({ago})' if ago else ''}"
    if str(best.get("event_type")) == "missing":
        lead = f"{text(best)} Checked at {when}."
        before = _previous_sighting(best, episodes)
        if before is not None:
            lead += (
                f" Before that it was seen at {before.get('location_tag', '?')}, "
                f"{_short_time(before.get('timestamp'))}."
            )
    else:
        lead = f"{text(best)} Last seen at {location}, {when}."

    if str(best.get("event_type")) == "moved":
        lead += " That was a change of location from where I had previously recorded it."

    count = best.get("observation_count")
    if isinstance(count, int) and count > 1:
        lead += f" I have logged that {count} times."

    if len(episodes) == 1:
        return lead

    others = "\n".join(
        f"- {text(ep)} ({ep.get('location_tag', '?')}, {_short_time(ep.get('timestamp', '?'))})"
        for ep in episodes
        if ep is not best
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
    store: "MemoryStore",
    k: int = 5,
    synthesize: Optional[Synthesizer] = None,
) -> dict[str, Any]:
    """Retrieve top-k episodes for `question` and synthesize a grounded answer."""
    episodes = store.search(question, k=k)
    if episodes:
        # Lead with the latest sighting of the asked-about object, so the
        # displayed and the spoken answers describe the same, current fact.
        best = _most_recent_sighting(question, episodes)
        episodes = [best] + [ep for ep in episodes if ep is not best]

    if synthesize is None:
        synthesize = anthropic_synthesize if os.environ.get("ANTHROPIC_API_KEY") else extractive_synthesize

    return {
        "question": question,
        "answer": synthesize(question, episodes),
        "episodes": episodes,
    }
