from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateCorpus_Command:
    manifest_path: str
    mode: str  # "offline" | "streaming"
    ablation_stage: str  # one of EvalConfig.ablation_stages
    # Cap utterances (smoke runs, timing probes); None = whole manifest. The cap is an evenly
    # strided subsample, never a head slice: a manifest is sorted by uttid, so its head is a
    # handful of speakers reading a handful of chapters.
    limit: int | None = None
    # Whether this pass' timings are worth reporting. False for the concurrent quality pass, where
    # several decodes share the GPU and every measured second is partly somebody else's contention.
    measure_timing: bool = True
