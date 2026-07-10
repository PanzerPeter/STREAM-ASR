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
    assert cfg.augment.speed_perturb_factors == (0.9, 1.0, 1.1)
    assert cfg.model.encoder_dims == (192, 256, 384, 512, 384, 256)
    assert cfg.model.vocab_size == 500
    assert cfg.training.stage_a.total_steps == 120000
    # Raised from 5e-4 to escape the CTC blank-collapse saddle on full data (see augment.yaml /
    # training.yaml notes); warmup was lengthened 3k->10k in the same change.
    assert cfg.training.stage_a.lr_peak == pytest.approx(1.5e-3)
    assert cfg.training.stage_a.warmup_steps == 10000


def test_derived_values():
    cfg = get_config()
    assert cfg.model.blank_id == cfg.model.vocab_size == 500
    assert cfg.model.logits_width == 501


def test_lists_coerce_to_tuples():
    cfg = get_config()
    assert isinstance(cfg.model.encoder_dims, tuple)
    assert isinstance(cfg.augment.speed_perturb_factors, tuple)


def test_validation_rejects_bad_type(tmp_path):
    src = Path("config")
    for name in [
        "audio.yaml",
        "augment.yaml",
        "model.yaml",
        "training.yaml",
        "decode.yaml",
        "lm.yaml",
        "eval.yaml",
    ]:
        shutil.copy(src / name, tmp_path / name)
    # n_mels must be int; write a non-coercible value.
    audio = tmp_path / "audio.yaml"
    audio.write_text(audio.read_text().replace("n_mels: 80", "n_mels: not_a_number"))
    with pytest.raises(ValidationError):
        get_config(str(tmp_path))


def test_decoder_and_stage_b_config():
    cfg = get_config()
    m = cfg.model
    assert m.decoder_dim == 512
    assert m.decoder_left_layers == 6
    assert m.decoder_right_layers == 3
    assert m.decoder_heads == 8
    # Decoder label space adds SOS/EOS above the acoustic vocab; blank stays in the CTC head.
    assert m.sos_id == m.vocab_size == 500
    assert m.eos_id == m.vocab_size + 1 == 501
    assert m.decoder_vocab_size == m.vocab_size + 2 == 502

    sb = cfg.training.stage_b
    assert sb.ctc_weight == pytest.approx(0.3)
    assert sb.reverse_weight == pytest.approx(0.3)
    assert sb.label_smoothing == pytest.approx(0.1)
    assert 0 in sb.chunk_sizes  # 0 encodes the full-context (no chunk) option
    assert sb.warm_start.endswith("stage_a_last.pt")


def test_decode_config_loads():
    d = get_config().decode
    assert d.beam_size >= 1
    assert d.chunk_size > 0
    assert 0.0 <= d.rescore_lambda <= 1.0
