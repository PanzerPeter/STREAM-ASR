import numpy as np
import torch

from src.slices.ExtractFeatures.FeatureCache import write_feature_cache
from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.BestRqPretrainer_Handler import run_pretrain
from src.slices.TrainAcousticModel.AcousticModel import AcousticModel


def _tiny_cache_and_manifest(tmp_path):
    # 6 short utterances of cached mel + a matching manifest.
    mels = [np.random.randn(120 + 8 * i, 80).astype(np.float16) for i in range(6)]
    write_feature_cache(str(tmp_path), "train", mels)
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, m in enumerate(mels):
            f.write(
                '{"uttid": "u%d", "audio_filepath": "x", "text": "a", "num_samples": %d}\n'
                % (i, m.shape[0] * 160)
            )
    return str(manifest)


def test_pretrain_smoke_emits_warmstartable_encoder(tmp_path):
    manifest = _tiny_cache_and_manifest(tmp_path)
    cmd = BestRqPretrainCommand(
        train_manifest=manifest,
        cache_dir=str(tmp_path),
        cache_split="train",
        cmvn_path="data/features/cmvn.pt",
        ckpt_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
        max_steps_smoke=3,
        # CPU smoke test: worker processes fork after torch/OpenMP threads are already live, which
        # deadlocks (same footgun SP1's precompute_features hit) — force single-process loading.
        num_workers=0,
    )
    out = run_pretrain(cmd)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    # Warm-start: the emitted encoder state_dict loads cleanly into a fresh AcousticModel encoder.
    model = AcousticModel(cmvn_path=None)
    missing, unexpected = model.encoder.load_state_dict(ckpt["model"], strict=False)
    assert unexpected == []  # no stray keys — it is exactly the encoder


def test_pretrain_resumes_and_bumps_resume_count(tmp_path):
    import os

    manifest = _tiny_cache_and_manifest(tmp_path)
    ck = str(tmp_path / "ck")
    base = dict(
        train_manifest=manifest,
        cache_dir=str(tmp_path),
        cache_split="train",
        cmvn_path="data/features/cmvn.pt",
        ckpt_dir=ck,
        log_dir=str(tmp_path / "runs"),
        device="cpu",
        num_workers=0,
    )
    run_pretrain(BestRqPretrainCommand(**base, max_steps_smoke=3))
    first = torch.load(os.path.join(ck, "bestrq_last.pt"), map_location="cpu", weights_only=False)
    assert first["resume_count"] == 0
    assert first["step"] == 3

    run_pretrain(BestRqPretrainCommand(**base, resume=True, max_steps_smoke=5))
    second = torch.load(os.path.join(ck, "bestrq_last.pt"), map_location="cpu", weights_only=False)
    # resumed from prior bestrq_last.pt (count bumped), not a fresh step-0 restart
    assert second["resume_count"] == 1
    assert second["step"] >= 3
