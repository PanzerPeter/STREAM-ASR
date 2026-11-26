from dataclasses import dataclass


@dataclass(frozen=True)
class BuildCommonVoiceManifestResponse:
    train_rows: int
    dev_rows: int
    test_rows: int
    dev_subset_rows: int
    # Clips removed from cv-train, per reason, in the order the filters run.
    dropped: dict[str, int]
    trimmed_rows: int
    train_hours_untrimmed: float
    train_hours: float
