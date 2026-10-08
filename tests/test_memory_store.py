"""
Tests for laptop/memory_store.py (Phase 6).

Adds several synthetic episodes and runs a semantic query worded
differently than what was stored, confirming the right episode is
retrieved (per the Phase 6 acceptance check in the build spec).
"""

import shutil
import tempfile

import pytest

# The ChromaDB-backed store is an optional extra: demo/fog_pipeline.py falls
# back to an in-process store without it, so the demo runs on a lightweight
# install. These tests exercise ChromaDB specifically, so skip rather than
# error when it is not installed.
pytest.importorskip(
    "chromadb",
    reason="needs the full install: pip install -r requirements-laptop.txt",
)

from laptop import embeddings
from laptop.memory_store import MemoryStore
from shared.event_schema import BBox, Event, EventType

# The cross-wording retrieval test ("blue flask" -> "water bottle") is a test of
# *semantic* embeddings. laptop/embeddings.py falls back to a lexical embedder
# when the sentence-transformers model cannot be downloaded, and that fallback
# genuinely cannot make that match - so skip rather than fail, instead of
# weakening the assertion into something the fallback happens to pass.
requires_semantic_embeddings = pytest.mark.skipif(
    embeddings.active_backend() != embeddings.BACKEND_TRANSFORMER,
    reason=(
        "needs the sentence-transformers backend; running on the "
        f"'{embeddings.active_backend()}' embedder"
    ),
)

BBOX = BBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)


def make_event(objects, relationships, location_tag, event_type=EventType.NEW_OBSERVATION):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=event_type,
        bbox=BBOX,
        location_tag=location_tag,
    )


@pytest.fixture
def store():
    tmp_dir = tempfile.mkdtemp()
    yield MemoryStore(persist_dir=tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_semantic_embeddings
def test_add_and_semantic_search(store):
    store.add_episode(
        make_event(["flask", "laptop"], ["flask LEFT OF laptop"], "lab_desk_3"),
        "A blue flask beside the laptop.",
    )
    store.add_episode(
        make_event(["mug", "shelf"], ["mug ON shelf"], "lab_shelf_1"),
        "A ceramic mug sitting on the shelf.",
    )
    store.add_episode(
        make_event(["keys", "table"], ["keys ON table"], "lab_desk_2"),
        "A set of keys on the table.",
    )

    results = store.search("where's my water bottle?", k=1)

    assert len(results) == 1
    assert results[0]["location_tag"] == "lab_desk_3"
    assert "flask" in results[0]["caption"].lower()


def test_count_reflects_added_episodes(store):
    assert store.count() == 0
    store.add_episode(
        make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1"),
        "A bottle is on the table.",
    )
    assert store.count() == 1
