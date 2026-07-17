import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.shared_kernel.Config_Adapter import get_config, StreamConfig


def test_loads_representative_values():
    cfg = get_config()
    assert isinstance(cfg, StreamConfig)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.n_mels == 80
    assert cfg.model.encoder_dims == (192, 256, 384, 512, 384, 256)
    assert cfg.model.vocab_size == 500
    assert cfg.training.transducer.total_steps == 120000
    # lr_peak lives solely in optim.yaml (adamw_lr/muon_lr), not per-stage.
    assert cfg.training.transducer.warmup_steps == 10000


def test_derived_values():
    cfg = get_config()
    assert cfg.model.blank_id == cfg.model.vocab_size == 500
    assert cfg.model.logits_width == 501


def test_lists_coerce_to_tuples():
    cfg = get_config()
    assert isinstance(cfg.model.encoder_dims, tuple)


def test_validation_rejects_bad_type(tmp_path):
    src = Path("config")
    for name in [
        "audio.yaml",
        "augment.yaml",
        "features.yaml",
        "model.yaml",
        "training.yaml",
        "decode.yaml",
        "lm.yaml",
        "eval.yaml",
        "optim.yaml",
        "pretrain.yaml",
        "transducer.yaml",
    ]:
        shutil.copy(src / name, tmp_path / name)
    # n_mels must be int; write a non-coercible value.
    audio = tmp_path / "audio.yaml"
    audio.write_text(audio.read_text().replace("n_mels: 80", "n_mels: not_a_number"))
    with pytest.raises(ValidationError):
        get_config(str(tmp_path))


def test_sos_eos_decoder_vocab_ids():
    # These label-space ids no longer back an acoustic attention decoder (deleted with the
    # CTC/attention two-stage path), but STREAM-LM (TrainLanguageModel slice) still consumes them
    # for SOS-conditioned next-token prediction, so they stay live on ModelConfig.
    m = get_config().model
    assert m.sos_id == m.vocab_size == 500
    assert m.eos_id == m.vocab_size + 1 == 501
    assert m.decoder_vocab_size == m.vocab_size + 2 == 502


def test_decode_config_loads():
    d = get_config().decode
    assert d.beam_size >= 1
    assert d.chunk_size > 0


def test_features_config_present():
    cfg = get_config()
    assert cfg.features.cache_dir == "data/features/mel"
    assert cfg.features.enabled is True
