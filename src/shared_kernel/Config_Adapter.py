# YAML-backed, pydantic-validated run config (infra)
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class AudioConfig(BaseModel):
    sample_rate: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    cmvn_eps: float


class AugmentConfig(BaseModel):
    specaug_num_freq_masks: int
    specaug_freq_width: int
    specaug_time_ratio: float
    specaug_max_time_masks: int
    specaug_max_time_fraction: float


class FeaturesConfig(BaseModel):
    cache_dir: str


class TrimConfig(BaseModel):
    threshold_fraction: float
    margin_s: float
    min_dynamic_range_db: float
    min_span_s: float


class ModelConfig(BaseModel):
    # An unknown key is an error, not a no-op: a misspelt architecture knob would otherwise train
    # the default model while the YAML claims otherwise.
    model_config = ConfigDict(extra="forbid")

    frontend_channels: int
    stack_dims: tuple[int, int, int, int]
    stack_blocks: tuple[int, int, int, int]
    head_dim: int
    conv_kernel_size: int
    gate_floor: float
    gate_init: float
    rope_base: float
    vocab_size: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blank_id(self) -> int:
        # CTC blank is appended after the SentencePiece unigram vocab.
        return self.vocab_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logits_width(self) -> int:
        return self.vocab_size + 1


class TransducerConfig(BaseModel):
    predictor_dim: int
    predictor_context: int
    joiner_dim: int
    ctc_loss_scale: float
    band_width: int


class TransducerTrainConfig(BaseModel):
    max_frames_per_batch: int
    max_tokens_per_batch: int
    max_lattice_per_batch: int
    grad_accum: int
    warmup_steps: int
    total_steps: int
    lr_decay_frac: float
    lr_min_ratio: float
    branch_points: tuple[int, ...]
    grad_clip: float
    guard_window: int
    guard_trend_factor: float
    guard_patience: int
    log_every: int
    val_every: int
    ckpt_every: int
    keep_last_n: int
    band_warmup_steps: int
    compile_blocks: bool
    seed: int
    chunk_sizes: tuple[int, ...]
    left_contexts: tuple[int, ...]
    spec_augment: bool


class TrainingConfig(BaseModel):
    transducer: TransducerTrainConfig


class DecodeConfig(BaseModel):
    chunk_size: int
    left_context: int
    tail_chunks: int
    beam_size: int
    max_symbols: int
    lm_checkpoint: str
    lm_weight: float
    ilm_weight: float
    length_bonus: float


class LmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer: str
    vocab_size: int
    eos_id: int
    d_model: int
    layers: int
    heads: int
    kv_groups: int
    ffn_expansion: int
    dropout: float
    context_len: int
    value_residual_lambda: float
    muon_lr: float
    z_loss: float
    lr_peak: float
    warmup_steps: int
    total_steps: int
    lr_decay_frac: float
    branch_points: tuple[int, ...]
    weight_decay: float
    grad_clip: float
    batch_size: int
    # batch_size is the EFFECTIVE (optimiser-step) batch; it is split into grad_accum micro-batches
    # so peak activation memory scales with batch_size/grad_accum while the gradient is unchanged.
    grad_accum: int = 1
    compile_blocks: bool
    eval_interval: int
    log_every: int
    ckpt_every: int
    subset_words: int
    val_words: int
    seed: int

    @model_validator(mode="after")
    def _ids_fit(self) -> "LmConfig":
        # Packed bins are uint16, and the EOS id indexes the embedding table.
        if not 0 <= self.eos_id < self.vocab_size <= 2**16:
            raise ValueError(
                f"need 0 <= eos_id < vocab_size <= 65536, got {self.eos_id}, {self.vocab_size}"
            )
        return self


class EvalConfig(BaseModel):
    ablation_stages: tuple[str, ...]
    report_path: str  # may contain a {split} placeholder (clean|other|cv)
    workers: int
    rtf_probe_utts: int


class OptimConfig(BaseModel):
    muon_lr: float
    adamw_lr: float
    muon_momentum: float
    ns_steps: int
    weight_decay: float


class StreamConfig(BaseModel):
    audio: AudioConfig
    augment: AugmentConfig
    features: FeaturesConfig
    trim: TrimConfig
    model: ModelConfig
    training: TrainingConfig
    decode: DecodeConfig
    lm: LmConfig
    eval: EvalConfig
    optim: OptimConfig
    transducer: TransducerConfig


@lru_cache(maxsize=None)
def get_config(config_dir: str | None = None) -> StreamConfig:
    root = Path(config_dir) if config_dir else _CONFIG_DIR
    return StreamConfig(
        **{k: yaml.safe_load((root / f"{k}.yaml").read_text()) for k in StreamConfig.model_fields}
    )
