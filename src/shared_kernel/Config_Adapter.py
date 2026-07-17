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
    specaug_num_freq_masks: int
    specaug_freq_width: int
    specaug_time_ratio: float
    specaug_max_time_masks: int


class FeaturesConfig(BaseModel):
    cache_dir: str
    enabled: bool


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blank_id(self) -> int:
        # CTC blank is appended after the SentencePiece unigram vocab.
        return self.vocab_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logits_width(self) -> int:
        return self.vocab_size + 1

    # sos_id/eos_id/decoder_vocab_size: the acoustic model's attention decoder (Stage-B, U2++) that
    # originally motivated this label space is gone (SP5 transducer replaces it), but STREAM-LM
    # (TrainLanguageModel slice) still frames next-token prediction as SOS-conditioned generation
    # over this same vocab, so the ids stay live.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sos_id(self) -> int:
        return self.vocab_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eos_id(self) -> int:
        return self.vocab_size + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decoder_vocab_size(self) -> int:
        return self.vocab_size + 2


class TransducerConfig(BaseModel):
    predictor_dim: int
    predictor_context: int
    joiner_dim: int
    ctc_aux_weight: float
    interctc_layers: tuple[int, ...]
    interctc_weights: tuple[float, ...]


class TransducerTrainConfig(BaseModel):
    max_frames_per_batch: int
    max_tokens_per_batch: int
    grad_accum: int
    warmup_steps: int
    total_steps: int
    weight_decay: float
    grad_clip: float
    log_every: int
    val_every: int
    ckpt_every: int
    grad_checkpoint: bool = False
    seed: int = 42
    chunk_sizes: tuple[int, ...]
    warm_start: str
    dev_wer_utts: int = 200
    spec_augment: bool = True


class TrainingConfig(BaseModel):
    transducer: TransducerTrainConfig


class DecodeConfig(BaseModel):
    chunk_size: int
    beam_size: int
    lm_weight: float
    lm_checkpoint: str
    max_symbols: int


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


class OptimConfig(BaseModel):
    optimizer: str
    muon_lr: float
    adamw_lr: float
    muon_momentum: float
    ns_steps: int
    weight_decay: float
    mup_enabled: bool
    mup_base_dims: tuple[int, ...]
    encoder_lr_scale: float = 1.0


class PretrainConfig(BaseModel):
    codebook_size: int
    codebook_dim: int
    mask_prob: float
    mask_span: int
    noise_std: float
    stack_frames: int
    warmup_steps: int
    total_steps: int
    seed: int
    grad_clip: float
    log_every: int
    save_every: int
    max_frames_per_batch: int


class StreamConfig(BaseModel):
    audio: AudioConfig
    augment: AugmentConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    decode: DecodeConfig
    lm: LmConfig
    eval: EvalConfig
    optim: OptimConfig
    pretrain: PretrainConfig
    transducer: TransducerConfig


@lru_cache(maxsize=None)
def get_config(config_dir: str | None = None) -> StreamConfig:
    root = Path(config_dir) if config_dir else _CONFIG_DIR
    data = {
        "audio": yaml.safe_load((root / "audio.yaml").read_text()),
        "augment": yaml.safe_load((root / "augment.yaml").read_text()),
        "features": yaml.safe_load((root / "features.yaml").read_text()),
        "model": yaml.safe_load((root / "model.yaml").read_text()),
        "training": yaml.safe_load((root / "training.yaml").read_text()),
        "decode": yaml.safe_load((root / "decode.yaml").read_text()),
        "lm": yaml.safe_load((root / "lm.yaml").read_text()),
        "eval": yaml.safe_load((root / "eval.yaml").read_text()),
        "optim": yaml.safe_load((root / "optim.yaml").read_text()),
        "pretrain": yaml.safe_load((root / "pretrain.yaml").read_text()),
        "transducer": yaml.safe_load((root / "transducer.yaml").read_text()),
    }
    return StreamConfig(**data)
