"""
Tests for the voice layer: utterance routing, spoken-answer phrasing, and the
POST /voice endpoint.

Recognition and synthesis live in the browser, so what is testable here - and
what is worth testing - is the part that decides what an utterance means and
how an answer sounds when it is read aloud.
"""

from datetime import datetime, timedelta, timezone

import pytest

from demo.fog_pipeline import FogPipeline
from laptop.voice import (
    NOTHING_FOUND,
    attach_spoken,
    humanize_location,
    relative_time,
    route_utterance,
    spoken_answer,
)
from shared.event_schema import BBox, Event, EventType

BBOX = BBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)
NOW = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def _episode(**overrides):
    episode = {
        "caption": "A bottle is on the dining table.",
        "display_caption": "A bottle is on the dining table.",
        "location_tag": "lab_desk_3",
        "timestamp": str(NOW - timedelta(minutes=4)),
        "event_type": "confirmed",
        "observation_count": 1,
    }
    episode.update(overrides)
    return episode


# --------------------------------------------------------------------------
# Routing: command vs question
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "utterance, action",
    [
        ("pause", "pause"),
        ("Pause.", "pause"),
        ("stop", "pause"),
        ("hold on", "pause"),
        ("resume", "resume"),
        ("keep going", "resume"),
        ("start patrol", "start"),
        ("start the patrol", "start"),
        ("begin", "start"),
        ("move the bottle", "move_bottle"),
    ],
)
def test_control_phrases_route_to_the_dashboard_actions(utterance, action):
    routed = route_utterance(utterance)

    assert routed["kind"] == "command"
    assert routed["action"] == action
    assert routed["say"]


@pytest.mark.parametrize(
    "utterance",
    [
        "where is the bottle?",
        "what is on desk 3?",
        "have you seen a laptop?",
        "did anything move",
        "tell me what you saw",
    ],
)
def test_questions_route_to_retrieval(utterance):
    routed = route_utterance(utterance)

    assert routed["kind"] == "question"
    assert routed["question"] == utterance


def test_a_question_containing_a_command_verb_is_still_a_question():
    """"did anything move?" must not fire the move_bottle command - this is
    the whole reason routing is not plain keyword matching."""
    assert route_utterance("did anything move?")["kind"] == "question"
    assert route_utterance("what did you stop for?")["kind"] == "question"


def test_a_command_word_inside_a_longer_word_does_not_fire():
    assert route_utterance("stopwatch on the bench")["kind"] == "question"


def test_silence_is_neither_a_command_nor_a_query():
    routed = route_utterance("   ")

    assert routed["kind"] == "empty"
    assert routed["say"]


# --------------------------------------------------------------------------
# Spoken phrasing
# --------------------------------------------------------------------------

def test_location_tags_are_spoken_as_words_not_identifiers():
    assert humanize_location("lab_desk_3") == "lab desk 3"
    assert humanize_location("charging-station") == "charging station"
    assert humanize_location(None) == ""


@pytest.mark.parametrize(
    "delta, expected",
    [
        (timedelta(seconds=2), "just now"),
        (timedelta(seconds=30), "less than a minute ago"),
        (timedelta(seconds=90), "about a minute ago"),
        (timedelta(minutes=4), "about four minutes ago"),
        (timedelta(minutes=90), "about an hour ago"),
        (timedelta(hours=5), "about five hours ago"),
        (timedelta(days=1, hours=2), "yesterday"),
        (timedelta(days=3), "three days ago"),
    ],
)
def test_timestamps_are_spoken_as_relative_times(delta, expected):
    assert relative_time(str(NOW - delta), now=NOW) == expected


def test_a_timestamp_slightly_ahead_of_the_clock_is_not_phrased_as_the_future():
    """Edge and fog clocks drift; a few seconds of skew is not a prediction."""
    assert relative_time(str(NOW + timedelta(seconds=5)), now=NOW) == "just now"
    assert relative_time(str(NOW + timedelta(hours=1)), now=NOW) == ""


def test_an_unparseable_timestamp_is_left_out_rather_than_guessed():
    assert relative_time("not a timestamp", now=NOW) == ""


def test_a_naive_timestamp_is_read_as_utc():
    naive = (NOW - timedelta(minutes=4)).replace(tzinfo=None)

    assert relative_time(str(naive), now=NOW) == "about four minutes ago"


def test_the_spoken_answer_leads_with_the_fact_then_where_and_when():
    spoken = spoken_answer({"episodes": [_episode()]}, now=NOW)

    assert spoken == (
        "A bottle is on the dining table. "
        "I last saw it at lab desk 3 about four minutes ago."
    )


def test_the_spoken_answer_has_no_bullet_list_or_wall_clock_timestamp():
    """The displayed answer carries both; read aloud they are unlistenable."""
    result = {"episodes": [_episode(), _episode(display_caption="A laptop is on the desk.")]}

    spoken = spoken_answer(result, now=NOW)

    assert "-" not in spoken
    assert "\n" not in spoken
    assert ":" not in spoken
    assert "two other matching observation" not in spoken  # one other, singular
    assert "one other matching observation." in spoken


def test_a_repeated_observation_is_counted_out_loud():
    spoken = spoken_answer({"episodes": [_episode(observation_count=4)]}, now=NOW)

    assert "I've logged that four times." in spoken


def test_seeing_something_twice_is_said_as_twice_not_as_two_times():
    spoken = spoken_answer({"episodes": [_episode(observation_count=2)]}, now=NOW)

    assert "I've logged that twice." in spoken


def test_a_move_is_called_out_and_the_caption_parenthetical_is_dropped():
    episode = _episode(
        event_type="moved",
        display_caption="A bottle is on the shelf. (moved from its previous location)",
    )

    spoken = spoken_answer({"episodes": [episode]}, now=NOW)

    assert "(" not in spoken and ")" not in spoken
    assert "It had moved from where I previously recorded it." in spoken


def test_an_empty_retrieval_says_so_plainly():
    assert spoken_answer({"episodes": []}) == NOTHING_FOUND
    assert spoken_answer({}) == NOTHING_FOUND


def test_a_missing_location_or_timestamp_does_not_produce_a_dangling_clause():
    spoken = spoken_answer(
        {"episodes": [_episode(location_tag=None, timestamp=None)]}, now=NOW
    )

    assert spoken == "A bottle is on the dining table."


def test_attach_spoken_adds_the_field_without_touching_the_displayed_answer():
    result = {"answer": "displayed text", "episodes": [_episode()]}

    attach_spoken(result, now=NOW)

    assert result["answer"] == "displayed text"
    assert result["spoken"].startswith("A bottle is on the dining table.")


# --------------------------------------------------------------------------
# End to end through the pipeline and the endpoint
# --------------------------------------------------------------------------

def _event(objects, relationships, location):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=EventType.CONFIRMED,
        bbox=BBOX,
        location_tag=location,
    )


def test_every_pipeline_answer_carries_a_spoken_variant():
    pipeline = FogPipeline(persist=False)
    pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3"))

    answer = pipeline.answer("where is the bottle?", k=1)

    assert "bottle" in answer["spoken"].lower()
    assert "lab desk 3" in answer["spoken"]
    # The displayed answer is unchanged by the voice feature being present.
    assert "A bottle is on the dining table." in answer["answer"]


def test_voice_endpoint_answers_a_spoken_question():
    from fastapi.testclient import TestClient

    from demo import server

    with TestClient(server.app) as client:
        client.post(
            "/event",
            json=_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3").model_dump(
                mode="json"
            ),
        )
        body = client.post("/voice", json={"text": "where is the bottle?"}).json()

    assert body["kind"] == "question"
    assert body["transcript"] == "where is the bottle?"
    assert "bottle" in body["spoken"].lower()
    assert body["episodes"], "a spoken question must return the retrieved episodes too"


def test_voice_endpoint_reports_a_command_it_cannot_run_without_an_edge_sim():
    """No 2D sim attached: the robot says so instead of silently doing nothing."""
    from fastapi.testclient import TestClient

    from demo import server

    with TestClient(server.app) as client:
        body = client.post("/voice", json={"text": "pause"}).json()

    assert body["kind"] == "command"
    assert body["action"] == "pause"
    assert body["control"]["status"] == "unavailable"
    assert "can't" in body["spoken"]


def test_voice_endpoint_runs_a_command_against_an_attached_sim():
    from fastapi.testclient import TestClient

    from demo import server

    class FakeAgent:
        running = True
        paused = False

        def pause(self):
            self.paused = True

        def stop(self):
            pass

    fake = FakeAgent()
    server.attach_agent(fake)
    try:
        with TestClient(server.app) as client:
            body = client.post("/voice", json={"text": "pause"}).json()
    finally:
        server.attach_agent(None)

    assert body["kind"] == "command"
    assert fake.paused
    assert body["spoken"] == "Pausing the patrol."


def test_voice_endpoint_handles_an_empty_transcript():
    from fastapi.testclient import TestClient

    from demo import server

    with TestClient(server.app) as client:
        body = client.post("/voice", json={"text": ""}).json()

    assert body["kind"] == "empty"
    assert body["spoken"]
