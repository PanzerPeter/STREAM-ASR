import numpy as np
import pytest
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCache import manifest_fingerprint, write_feature_cache
from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.BestRqPretrainer_Handler import run_pretrain
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def _tiny_cache_and_manifest(tmp_path):
    # 6 short utterances of cached mel + a matching manifest.
    mels = [np.random.randn(120 + 8 * i, 80).astype(np.float16) for i in range(6)]
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, m in enumerate(mels):
            f.write(
                '{"uttid": "u%d", "audio_filepath": "x", "text": "a", "num_samples": %d}\n'
                % (i, m.shape[0] * 160)
            )
    write_feature_cache(str(tmp_path), "train", mels, manifest_fingerprint(str(manifest)))
    return str(manifest)


def test_pretrain_smoke_emits_warmstartable_encoder(tmp_path):
    manifest = _tiny_cache_and_manifest(tmp_path)
    cmd = BestRqPretrainCommand(
        train_manifest=manifest,
        cache_dir=str(tmp_path),
        cache_split="train",
        # The held-out probe is built eagerly at startup, so point it at the tiny cache too --
        # the real dev split would put 2,703 utterances through the encoder on CPU.
        dev_manifest=manifest,
        dev_cache_split="train",
        cmvn_path="",  # deliberate: no cmvn.pt in a fresh checkout (data/ is gitignored)
        ckpt_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
        max_steps_smoke=3,
        # CPU smoke test: worker processes fork after torch/OpenMP threads are already live, which
        # deadlocks (the same footgun precompute_features hit), so force single-process loading.
        num_workers=0,
    )
    out = run_pretrain(cmd)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    # Warm-start: the emitted encoder state_dict loads cleanly into a fresh TransducerModel encoder.
    model = TransducerModel(cmvn_path=None)
    missing, unexpected = model.encoder.load_state_dict(ckpt["model"], strict=False)
    assert unexpected == []  # no stray keys, it is exactly the encoder


def test_pretrain_resumes_and_bumps_resume_count(tmp_path):
    import os

    manifest = _tiny_cache_and_manifest(tmp_path)
    ck = str(tmp_path / "ck")
    base = dict(
        train_manifest=manifest,
        cache_dir=str(tmp_path),
        cache_split="train",
        dev_manifest=manifest,
        dev_cache_split="train",
        cmvn_path="",  # deliberate: no cmvn.pt in a fresh checkout (data/ is gitignored)
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


def test_pretrain_does_not_apply_encoder_lr_scale(tmp_path, monkeypatch):
    """The encoder must train at the full calibrated peak LR in this stage.

    `optim.encoder_lr_scale` exists to fine-tune a WARM-STARTED encoder gently in the supervised
    stage. `BestRqModel.encoder` matches the `encoder.` prefix `build_optimizer` keys on, so
    without an explicit override the configured 0.5 silently halved the LR of 53.8 M of the
    model's 55.9 M parameters for the whole pretrain -- leaving only the 2.1 M `pred_head` at full.
    """
    captured = []
    import src.slices.PretrainEncoder.BestRqPretrainer_Handler as handler

    real = handler.build_optimizer

    def spy(model, cfg):
        captured.append(cfg.encoder_lr_scale)
        return real(model, cfg)

    monkeypatch.setattr(handler, "build_optimizer", spy)
    manifest = _tiny_cache_and_manifest(tmp_path)
    run_pretrain(
        BestRqPretrainCommand(
            train_manifest=manifest,
            cache_dir=str(tmp_path),
            cache_split="train",
            dev_manifest=manifest,
            dev_cache_split="train",
            cmvn_path="",  # deliberate: no cmvn.pt in a fresh checkout (data/ is gitignored)
            ckpt_dir=str(tmp_path / "ck"),
            log_dir=str(tmp_path / "runs"),
            device="cpu",
            max_steps_smoke=1,
            num_workers=0,
        )
    )
    assert captured == [1.0]
    assert get_config().optim.encoder_lr_scale != 1.0  # the override is doing real work


def test_pretrain_scales_every_peak_lr_by_pretrain_lr_scale(tmp_path, monkeypatch):
    """`pretrain.lr_scale` is this stage's own LR knob, and it must reach EVERY group.

    Removing `encoder_lr_scale` from this stage (the test above) doubled the encoder's LR, which
    was never a measured choice, and the trunk the transducer stage inherits scaled with it: over
    the two 180k-step pretrains the in_proj gains moved by ~sqrt(2) per stack (product 1.2 -> 5.4)
    and the realized trunk amplitude on dev audio went from RMS 2.2-3.6 to 5.7-17.0, against
    1.5-2.7 for the model that shipped 3.43 % test-clean. Uniform across groups is the point: an
    LR split between the encoder and `pred_head` is the accident this stage just removed.
    """
    import src.slices.PretrainEncoder.BestRqPretrainer_Handler as handler

    scale = get_config().pretrain.lr_scale
    assert scale != 1.0, "a scale of 1 would make this test vacuous"

    built: list[list[float]] = []
    real = handler.build_optimizer
    optimizers: list = []

    def spy(model, cfg):
        opts = real(model, cfg)
        built.append([g["lr"] for opt in opts for g in opt.param_groups])
        optimizers.extend(opts)
        return opts

    # Pin the schedule at its peak so the assertion is exact rather than schedule-dependent.
    monkeypatch.setattr(handler, "build_optimizer", spy)
    monkeypatch.setattr(handler, "lr_at", lambda *a, **k: 1.0)
    manifest = _tiny_cache_and_manifest(tmp_path)
    run_pretrain(
        BestRqPretrainCommand(
            train_manifest=manifest,
            cache_dir=str(tmp_path),
            cache_split="train",
            dev_manifest=manifest,
            dev_cache_split="train",
            cmvn_path="",  # deliberate: no cmvn.pt in a fresh checkout (data/ is gitignored)
            ckpt_dir=str(tmp_path / "ck"),
            log_dir=str(tmp_path / "runs"),
            device="cpu",
            max_steps_smoke=1,
            num_workers=0,
        )
    )
    applied = [g["lr"] for opt in optimizers for g in opt.param_groups]
    assert built and len(applied) == len(built[0])
    assert applied == pytest.approx([base * scale for base in built[0]])


def test_pretrain_refuses_a_missing_cmvn_file(tmp_path):
    # Here the fallback is worse than unnormalised input: the mask fill is de-normalised through
    # the same statistics, so mean 0 / std 1 restores the constant +1.46 sigma plateau, in the one
    # stage that logs no amplitude at all. Refused before any data is touched.
    cmd = BestRqPretrainCommand(
        train_manifest=str(tmp_path / "absent.jsonl"),
        cache_dir=str(tmp_path),
        cmvn_path=str(tmp_path / "absent.pt"),
        ckpt_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
        num_workers=0,
    )
    with pytest.raises(FileNotFoundError, match="cmvn"):
        run_pretrain(cmd)
