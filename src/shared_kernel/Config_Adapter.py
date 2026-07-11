# src/shared_kernel/Config_Adapter.py — YAML-backed, pydantic-validated run config (infra)
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, computed_field

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class AudioConfig(BaseModel):
    sample_rate: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    cmvn_eps: float


class AugmentConfig(BaseModel):
    speed_perturb_factors: tuple[float, ...]
    specaug_num_freq_masks: int
    specaug_freq_width: int
    specaug_time_ratio: float
    specaug_max_time_masks: int


class ModelConfig(BaseModel):
    frontend_channels: int
    encoder_dims: tuple[int, ...]
    encoder_downsampling: tuple[int, ...]
    encoder_layers: tuple[int, ...]
    encoder_heads: tuple[int, ...]
    ffn_expansion: int
    conv_kernel_size: int
    encoder_dropout: float
    final_downsample: int
    rope_base: float
    encoder_value_residual_lambda: float
    vocab_size: int
    decoder_dim: int
    decoder_left_layers: int
    decoder_right_layers: int
    decoder_heads: int
    decoder_ffn_expansion: int
    decoder_dropout: float
    decoder_value_residual_lambda: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blank_id(self) -> int:
        # CTC blank is appended after the SentencePiece unigram vocab.
        return self.vocab_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logits_width(self) -> int:
        return self.vocab_size + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sos_id(self) -> int:
        # Decoder label space sits above the acoustic vocab; distinct from the CTC blank head.
        return self.vocab_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eos_id(self) -> int:
        return self.vocab_size + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decoder_vocab_size(self) -> int:
        return self.vocab_size + 2


class StageAConfig(BaseModel):
    max_frames_per_batch: int
    grad_accum: int
    lr_peak: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    grad_clip: float
    log_every: int
    val_every: int
    ckpt_every: int
    # Activation checkpointing recomputes each stack's forward in the backward pass to bound VRAM.
    # It is a pure compute-for-memory trade (~+30% step time, identical gradients). At 20k frames
    # the model peaks ~4.6 GB without it on a 12 GB card, so it stays off by default.
    grad_checkpoint: bool = False
    # Blank-collapse guard. A healthy CTC run leaves the all-blank saddle within the first epoch or
    # two; if the dev blank-argmax fraction is still above escape_max_blank_frac once training has
    # passed escape_check_step, alignment never formed and the remaining steps are wasted — abort
    # fast instead of riding a dead run to total_steps.
    # Checked after warmup completes plus a margin at peak LR, so a slow-but-healthy escape is not
    # mistaken for collapse. Keep it comfortably past warmup_steps.
    escape_check_step: int = 18000
    escape_max_blank_frac: float = 0.95
    # The escape onset is noisy: blank_frac can still read ~1.0 at the check step while dev WER has
    # already begun falling off 1.0 (alignment forming). Only abort if BOTH signals say "dead" —
    # blank still collapsed AND best dev WER never dropped below this floor. Prevents guillotining a
    # run that is escaping the saddle but slower than escape_check_step.
    escape_min_wer_progress: float = 0.99
    # RNG seed for model init + augmentation + batch order. The blank-collapse escape is init-
    # sensitive (a knife-edge); seeding makes a run reproducible so a winning init can be re-run and
    # a losing seed swapped. Not full determinism (cuDNN's CTC has no deterministic kernel).
    seed: int = 42


class StageBConfig(BaseModel):
    max_frames_per_batch: int
    grad_accum: int
    lr_peak: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    grad_clip: float
    log_every: int
    val_every: int
    ckpt_every: int
    grad_checkpoint: bool = False
    escape_check_step: int = 12000
    escape_max_blank_frac: float = 0.95
    escape_min_wer_progress: float = 0.99
    seed: int = 42
    ctc_weight: float
    reverse_weight: float
    label_smoothing: float
    chunk_sizes: tuple[int, ...]
    warm_start: str


class TrainingConfig(BaseModel):
    stage_a: StageAConfig
    stage_b: StageBConfig


class DecodeConfig(BaseModel):
    chunk_size: int
    beam_size: int
    rescore_lambda: float
    rescore_ctc_weight: float
    lm_weight: float
    lm_checkpoint: str


class LmConfig(BaseModel):
    d_model: int
    layers: int
    heads: int
    kv_groups: int
    ffn_expansion: int
    dropout: float
    context_len: int
    value_residual_lambda: float
    lr_peak: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    grad_clip: float
    batch_size: int
    eval_interval: int
    log_every: int
    subset_words: int
    val_words: int
    seed: int


class EvalConfig(BaseModel):
    ablation_stages: tuple[str, ...]
    report_path: str


class StreamConfig(BaseModel):
    audio: AudioConfig
    augment: AugmentConfig
    model: ModelConfig
    training: TrainingConfig
    decode: DecodeConfig
    lm: LmConfig
    eval: EvalConfig


@lru_cache(maxsize=None)
def get_config(config_dir: str | None = None) -> StreamConfig:
    root = Path(config_dir) if config_dir else _CONFIG_DIR
    data = {
        "audio": yaml.safe_load((root / "audio.yaml").read_text()),
        "augment": yaml.safe_load((root / "augment.yaml").read_text()),
        "model": yaml.safe_load((root / "model.yaml").read_text()),
        "training": yaml.safe_load((root / "training.yaml").read_text()),
        "decode": yaml.safe_load((root / "decode.yaml").read_text()),
        "lm": yaml.safe_load((root / "lm.yaml").read_text()),
        "eval": yaml.safe_load((root / "eval.yaml").read_text()),
    }
    return StreamConfig(**data)
