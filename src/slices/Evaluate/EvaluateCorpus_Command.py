from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateCorpus_Command:
    manifest_path: str
    mode: str  # "offline" | "streaming"
    ablation_stage: str  # one of EvalConfig.ablation_stages
    limit: int | None = None  # cap utterances (smoke/dev); None = whole manifest
