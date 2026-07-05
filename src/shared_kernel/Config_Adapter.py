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
    vocab_size: int
    decoder_dim: int
    decoder_left_layers: int
    decoder_right_layers: int
    decoder_heads: int
    decoder_ffn_expansion: int
    decoder_dropout: float

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
    ctc_weight: float
    reverse_weight: float
    label_smoothing: float
    chunk_sizes: tuple[int, ...]
    warm_start: str


class TrainingConfig(BaseModel):
    stage_a: StageAConfig
    stage_b: StageBConfig


class StreamConfig(BaseModel):
    audio: AudioConfig
    augment: AugmentConfig
    model: ModelConfig
    training: TrainingConfig


@lru_cache(maxsize=None)
def get_config(config_dir: str | None = None) -> StreamConfig:
    root = Path(config_dir) if config_dir else _CONFIG_DIR
    data = {
        "audio": yaml.safe_load((root / "audio.yaml").read_text()),
        "augment": yaml.safe_load((root / "augment.yaml").read_text()),
        "model": yaml.safe_load((root / "model.yaml").read_text()),
        "training": yaml.safe_load((root / "training.yaml").read_text()),
    }
    return StreamConfig(**data)
