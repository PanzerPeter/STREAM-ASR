import numpy as np
import pytest
from torch.utils.data import RandomSampler

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command
from src.slices.TrainLanguageModel.TrainLm_Handler import TrainLm_Handler


def test_loader_samples_with_replacement(tmp_path):
    # Regression guard: shuffle=True over the ~1.6e9-window production corpus makes RandomSampler
    # materialize torch.randperm(1.6e9).tolist() (~13 GB tensor + billion-element list) and OOM-swap
    # before step 1. The loader MUST sample with replacement so the sampler stays lazy/flat-memory.
    bin_path = tmp_path / "toy.bin"
    bin_path.write_bytes(np.arange(500, dtype=np.uint16).tobytes())
    loader = TrainLm_Handler()._loader(str(bin_path), ctx=8, batch=4)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is True


@pytest.mark.slow
def test_lm_overfits_tiny_corpus(tmp_path, monkeypatch):
    # Shrink the model + schedule to a CPU-seconds smoke that still genuinely trains: a tiny
    # deep-narrow LM must memorize a period-10 token pattern, so val perplexity collapses toward 1.
    # Production config is untouched — monkeypatch reverts these fields after the test.
    lm = get_config().lm
    for field, value in {
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "kv_groups": 2,
        "ffn_expansion": 2,
        "context_len": 32,
        "batch_size": 16,
        "warmup_steps": 5,
        "eval_interval": 100,
        "lr_peak": 1.0e-3,
    }.items():
        monkeypatch.setattr(lm, field, value)

    # Both bins must exceed context_len so LmDataset yields at least one window.
    train = np.tile(np.arange(10, dtype=np.uint16), 200)  # 2000 tokens
    val = np.tile(np.arange(10, dtype=np.uint16), 20)  # 200 tokens > context_len
    (tmp_path / "train.bin").write_bytes(train.tobytes())
    (tmp_path / "val.bin").write_bytes(val.tobytes())
    cmd = TrainLm_Command(
        train_bin=str(tmp_path / "train.bin"),
        val_bin=str(tmp_path / "val.bin"),
        out_dir=str(tmp_path / "ckpt"),
        max_steps=300,
        log_dir=str(tmp_path / "runs"),  # keep TensorBoard events out of the repo's runs/
    )
    best_ppl = TrainLm_Handler().run(cmd)
    assert best_ppl < 3.0  # memorized the deterministic pattern
    assert (tmp_path / "ckpt" / "lm_best.pt").exists()
