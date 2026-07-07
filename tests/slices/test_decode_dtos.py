from dataclasses import FrozenInstanceError
import pytest
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response, SegmentResult


def test_dtos_are_frozen():
    cmd = StreamingDecode_Command(audio_path="a.flac", streaming=True)
    resp = StreamingDecode_Response(
        text="hi",
        segments=[SegmentResult("hi", [("hi", -0.1)])],
        rtf=0.2,
        first_partial_latency_s=0.4,
    )
    assert resp.segments[0].text == "hi"
    with pytest.raises(FrozenInstanceError):
        cmd.audio_path = "b.flac"  # type: ignore[misc]
