from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingDecode_Command:
    audio_path: str
    streaming: bool = True  # False -> offline full-context two-pass
