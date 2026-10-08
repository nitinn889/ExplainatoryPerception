"""
Voice interface support: utterance routing and spoken-answer phrasing.

The voice feature is an *addition* to the dashboard, not a replacement for it.
Speech recognition and synthesis themselves run in the browser (Web Speech
API) — see `demo/static/dashboard.html`. What lives here is the part that
should not live in JavaScript:

  * `route_utterance` — decide whether what the user said is a control
    command ("pause", "start the patrol") or a question for episodic memory
    ("did anything move?"). Keyword matching alone gets this wrong: "did
    anything move?" is a question that contains a command verb, so question
    shape is checked first.

  * `spoken_answer` — re-phrase a `rag_query.answer_question` result for
    being read aloud. The displayed answer is deliberately dense: it carries
    a bulleted "Other observations that matched:" list and exact timestamps
    like 06:20:06, which are what you want on screen and unlistenable out
    loud. The spoken variant keeps the same grounded facts - it still only
    uses the retrieved episodes' own captions and metadata, so it carries the
    same zero-hallucination guarantee - but says them in two or three
    sentences, with relative times.

Both are pure functions over the dicts the rest of the pipeline already
passes around, so they are unit-testable without audio hardware, a browser
or a vector store.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

# What the robot says when retrieval came back empty. The displayed
# equivalent is rag_query's "I don't have any recorded observations relevant
# to that question."
NOTHING_FOUND = "I don't have any recorded observations about that yet."

# Captions carry trailing parentheticals from captioning.caption_from_event
# ("(moved from its previous location)"). Spoken, a parenthesis is either read
# out as a word or swallowed; the same information is re-added as a clause.
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)

# Leading words that make an utterance a question even when it also contains
# a command verb - "did anything move?" must not trip the `move_bottle`
# command. Checked before the command table for exactly that reason.
_QUESTION_LEADS = frozenset(
    """who what what's where where's when why which how did do does is
    are was were have has can could should tell show list""".split()
)

# Spoken phrasing -> the /control/{action} the dashboard buttons already post.
# Longest phrases first so "start patrol" is not shadowed by "start".
_COMMANDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("move the bottle", "move bottle", "stage a move", "staged event"),
        "move_bottle",
        "Moving the bottle to another station.",
    ),
    (
        ("start patrol", "start the patrol", "begin patrol", "start patrolling",
         "start", "begin", "go"),
        "start",
        "Starting the patrol.",
    ),
    (
        ("resume", "carry on", "continue", "keep going"),
        "resume",
        "Resuming the patrol.",
    ),
    (
        ("pause", "stop", "hold on", "wait", "halt", "freeze"),
        "pause",
        "Pausing the patrol.",
    ),
)


def _spell(value: int) -> str:
    """7 -> 'seven'. Past twenty, digits read fine on their own."""
    if 0 <= value < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[value]
    return str(value)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation a recognizer may or may not include."""
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).strip()


def humanize_location(tag: Any) -> str:
    """'desk_3' -> 'desk 3'. Location tags are identifiers, not speech."""
    if not tag:
        return ""
    return re.sub(r"\s+", " ", str(tag).replace("_", " ").replace("-", " ")).strip()


def _clean_caption(caption: Any) -> str:
    """Drop parentheticals and guarantee one terminating full stop."""
    text = _PARENTHETICAL_RE.sub("", str(caption or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Episode timestamps reach us as `str(event.timestamp)`; be forgiving."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    # Naive timestamps are UTC by convention - event_schema stamps
    # datetime.now(timezone.utc) - but a round-trip can lose the offset.
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def relative_time(timestamp: Any, now: Optional[datetime] = None) -> str:
    """'about four minutes ago'. Empty string when it cannot be phrased.

    A timestamp slightly in the future is clock skew between the edge and the
    fog node, not a prediction, so a small negative delta reads as "just now".
    A large one is phrased as nothing at all rather than as nonsense.
    """
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return ""

    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    seconds = (reference - parsed).total_seconds()
    if seconds < -60:
        return ""
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return "less than a minute ago"
    if seconds < 120:
        return "about a minute ago"
    if seconds < 3600:
        return f"about {_spell(int(seconds // 60))} minutes ago"
    if seconds < 7200:
        return "about an hour ago"
    if seconds < 86400:
        return f"about {_spell(int(seconds // 3600))} hours ago"
    if seconds < 172800:
        return "yesterday"
    return f"{_spell(int(seconds // 86400))} days ago"


def route_utterance(text: str) -> dict[str, Any]:
    """Classify a transcript as a control command or a memory question.

    Returns `{"kind": "command", "action": ..., "say": ...}`,
    `{"kind": "question", "question": ...}`, or
    `{"kind": "empty", "say": ...}` for silence.
    """
    raw = (text or "").strip()
    if not raw:
        return {"kind": "empty", "say": "I didn't catch that."}

    normalized = _normalize(raw)
    lead = normalized.split()[0] if normalized else ""
    looks_like_question = raw.rstrip().endswith("?") or lead in _QUESTION_LEADS

    if not looks_like_question:
        for phrases, action, confirmation in _COMMANDS:
            for phrase in phrases:
                # Whole-phrase match: "stop" should fire, "stopwatch" should not.
                if re.search(rf"(?:^|\s){re.escape(phrase)}(?:\s|$)", normalized):
                    return {"kind": "command", "action": action, "say": confirmation}

    return {"kind": "question", "question": raw}


def spoken_answer(result: dict[str, Any], now: Optional[datetime] = None) -> str:
    """Speech-shaped phrasing of an `answer_question` result.

    Every fact comes from the retrieved episodes, exactly as in the extractive
    displayed answer - this re-orders and re-words, it never adds.
    """
    episodes = result.get("episodes") or []
    if not episodes:
        return NOTHING_FOUND

    best = episodes[0]
    sentences = []

    caption = _clean_caption(best.get("display_caption") or best.get("caption"))
    if caption:
        sentences.append(caption)

    location = humanize_location(best.get("location_tag"))
    when = relative_time(best.get("timestamp"), now=now)
    if location and when:
        sentences.append(f"I last saw it at {location} {when}.")
    elif location:
        sentences.append(f"That was at {location}.")
    elif when:
        sentences.append(f"I last saw it {when}.")

    if str(best.get("event_type")) == "moved":
        sentences.append("It had moved from where I previously recorded it.")

    count = best.get("observation_count")
    if isinstance(count, int) and count > 1:
        times = "twice" if count == 2 else f"{_spell(count)} times"
        sentences.append(f"I've logged that {times}.")

    others = len(episodes) - 1
    if others > 0:
        plural = "s" if others > 1 else ""
        sentences.append(f"I have {_spell(others)} other matching observation{plural}.")

    return " ".join(s for s in sentences if s) or NOTHING_FOUND


def attach_spoken(result: dict[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
    """Add the `spoken` field to a query result, in place, and return it.

    Kept separate from `rag_query.answer_question` so the retrieval path has no
    opinion about speech; every surface that may be read aloud calls this.
    """
    result["spoken"] = spoken_answer(result, now=now)
    return result
