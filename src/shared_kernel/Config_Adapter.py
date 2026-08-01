# YAML-backed, pydantic-validated run config (infra)
from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # bos_id/eos_id/decoder_vocab_size: the acoustic model is a transducer and has no attention
    # decoder, but STREAM-LM (TrainLanguageModel slice) frames next-token prediction as
    # BOS-conditioned generation over this same vocab, so the ids stay live.
    #
    # BOS *is* EOS on purpose. PrepareLmData packs the corpus as `line tokens + eos_id` with no
    # separate start symbol, so the only sentence-start context the LM is ever trained on is
    # "previous line's EOS". A distinct start id would index an embedding row that never appears as
    # a model input during training, making the first-token score of every rescored hypothesis
    # come out of an untrained vector. Reusing EOS keeps decode-time scoring in-distribution.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def bos_id(self) -> int:
        return self.vocab_size + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eos_id(self) -> int:
        return self.vocab_size + 1

    # +2, not +1: id `vocab_size` is an unused hole (it was the old separate start symbol) that the
    # table must still span to make eos_id = vocab_size + 1 addressable. Renumbering it away would
    # invalidate every packed LM bin and LM checkpoint for one wasted embedding row.
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
    cr_ctc: bool = False
    cr_weight: float = 0.2


class TransducerTrainConfig(BaseModel):
    max_frames_per_batch: int
    max_tokens_per_batch: int
    max_lattice_per_batch: int
    # Width of the transcript-length re-sort inside the duration sort (FrameBucketSampler). 1 = off.
    token_sort_window: int = 1
    grad_accum: int
    warmup_steps: int
    total_steps: int
    # LR shape after warmup. "wsd" = hold lr_stable_ratio * peak until the last lr_decay_frac of
    # total_steps, then 1-sqrt anneal to lr_min_ratio * peak; "cosine" = anneal from step
    # warmup_steps to lr_min_ratio * peak at total_steps. Both land at total_steps exactly.
    lr_schedule: Literal["cosine", "wsd"] = "cosine"
    lr_stable_ratio: float = 1.0
    lr_decay_frac: float = 0.25
    lr_min_ratio: float = 0.0
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
    # Rolling snapshots kept for post-training checkpoint averaging (icefall/ESPnet-style). Every
    # ckpt_every steps the trainer writes transducer_step{N}.pt, pruned to the newest keep_last_n;
    # scripts/average_checkpoints.py means their weights into one decode checkpoint. 0 disables.
    keep_last_n: int = 5


class TrainingConfig(BaseModel):
    transducer: TransducerTrainConfig


class DecodeConfig(BaseModel):
    chunk_size: int
    beam_size: int
    lm_weight: float
    lm_checkpoint: str
    max_symbols: int
    # ILME subtraction weight (beta): score -= ilm_weight * internal-LM logprob. Cancels the
    # transducer's own language prior so the external LM is not counted on top of it. 0.0 = off.
    ilm_weight: float = 0.0
    # Per-token bonus added to each hyp at n-best re-ranking (score += length_bonus*len(ids)).
    # RNN-T acoustic scores are un-normalised sums of emission log-probs, so every extra token only
    # lowers the score -- a standing bias toward deletions. A small positive bonus offsets it. 0.0 =
    # off (regression lock); swept alongside lm_weight in the eval tuner.
    length_bonus: float = 0.0


class LmConfig(BaseModel):
    d_model: int
    layers: int
    heads: int
    kv_groups: int
    ffn_expansion: int
    dropout: float
    context_len: int
    value_residual_lambda: float
    optimizer: str
    muon_lr: float
    z_loss: float
    lr_peak: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    grad_clip: float
    batch_size: int
    # batch_size is the EFFECTIVE (optimiser-step) batch; it is split into grad_accum micro-batches
    # so peak activation memory scales with batch_size/grad_accum while the gradient is unchanged.
    grad_accum: int = 1
    eval_interval: int
    log_every: int
    ckpt_every: int
    subset_words: int
    val_words: int
    seed: int


class EvalConfig(BaseModel):
    ablation_stages: tuple[str, ...]
    report_path: str  # may contain a {split} placeholder (clean|other)
    workers: int
    rtf_probe_utts: int


class OptimConfig(BaseModel):
    optimizer: str
    muon_lr: float
    adamw_lr: float
    muon_momentum: float
    ns_steps: int
    weight_decay: float
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
