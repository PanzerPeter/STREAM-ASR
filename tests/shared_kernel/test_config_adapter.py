import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.shared_kernel.Config_Adapter import StreamConfig, get_config


def test_loads_representative_values():
    cfg = get_config()
    assert isinstance(cfg, StreamConfig)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.n_mels == 80
    assert cfg.model.stack_dims == (320, 512, 640, 512)
    assert cfg.model.stack_blocks == (4, 8, 0, 8)
    assert cfg.model.vocab_size == 500
    assert cfg.training.transducer.total_steps == 1200000
    assert cfg.training.transducer.warmup_steps == 5000


def test_derived_values():
    cfg = get_config()
    assert cfg.model.blank_id == cfg.model.vocab_size == 500
    assert cfg.model.logits_width == 501


def test_lists_coerce_to_tuples():
    cfg = get_config()
    assert isinstance(cfg.model.stack_dims, tuple)


def test_validation_rejects_bad_type(tmp_path):
    src = Path("config")
    for name in [
        "audio.yaml",
        "augment.yaml",
        "features.yaml",
        "trim.yaml",
        "model.yaml",
        "training.yaml",
        "decode.yaml",
        "lm.yaml",
        "eval.yaml",
        "optim.yaml",
        "transducer.yaml",
    ]:
        shutil.copy(src / name, tmp_path / name)
    # n_mels must be int; write a non-coercible value.
    audio = tmp_path / "audio.yaml"
    audio.write_text(audio.read_text().replace("n_mels: 80", "n_mels: not_a_number"))
    with pytest.raises(ValidationError):
        get_config(str(tmp_path))


def test_lm_owns_its_token_ids():
    # STREAM-LM reads its own tokenizer; its EOS (also its BOS) must fit the table and a uint16 bin.
    lm = get_config().lm
    assert 0 <= lm.eos_id < lm.vocab_size <= 2**16
    assert not hasattr(get_config().model, "eos_id")


def test_lm_rejects_an_eos_outside_the_vocab(tmp_path):
    for f in (Path(__file__).resolve().parents[2] / "config").glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    lm = tmp_path / "lm.yaml"
    lm.write_text(lm.read_text().replace("eos_id: 1", "eos_id: 8192"))
    with pytest.raises(ValidationError):
        get_config(str(tmp_path))


def test_decode_config_loads():
    d = get_config().decode
    assert d.beam_size >= 1
    assert d.chunk_size > 0


def test_features_config_present():
    cfg = get_config()
    assert cfg.features.cache_dir == "data/features/mel"


def test_an_unknown_model_key_is_an_error(tmp_path):
    # A misspelt architecture knob must not train the default model in silence.
    shutil.copytree("config", tmp_path, dirs_exist_ok=True)
    model = tmp_path / "model.yaml"
    model.write_text(model.read_text() + "block_order: parallel\n")
    with pytest.raises(ValidationError, match="block_order"):
        get_config(str(tmp_path))
