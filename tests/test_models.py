from clip_engine.models import Segment, Transcript, Word


def test_transcript_round_trips_through_json(tmp_path):
    original = Transcript(
        episode_id="ep1",
        segments=[
            Segment(0, 1000, "hello there", "SPEAKER_00", [Word(0, 400, "hello"), Word(450, 1000, "there")]),
            Segment(1000, 2000, "no speaker"),
        ],
        language="en",
        model="small",
    )
    path = tmp_path / "transcript.json"
    original.to_json(path)

    assert Transcript.from_json(path) == original
