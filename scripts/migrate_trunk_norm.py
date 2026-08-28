"""Insert the trunk normalisers into an existing transducer checkpoint.

`model.trunk_norm` adds a `BiasNorm` to each of the five stacks whose `in_proj` is a `Linear`,
which is 10 new parameters the checkpoint does not carry -- so a plain resume fails on
`load_state_dict`. This writes a checkpoint that resumes cleanly, keeping the step counter, the
LR-schedule position, the RNG state, the guard floor and both optimizers' accumulated state.

Each new `log_scale` is initialised to `min(log(measured trunk RMS), trunk_norm_log_scale_max)`,
where the RMS is measured on real dev audio through forward hooks on `in_proj` itself, so a stack
already inside the ceiling starts inert in the mean and only a stack above it is clamped. The
`bias` starts at 0, as it does everywhere else in the model.

MEASURED on `transducer_step81000.pt` with the ceiling at exp(1.2) = 3.32: joint loss 0.504 ->
1.186 on 8 dev utterances, of which 82 % is InterCTC -- a linear head calibrated to the old
amplitude, which re-fits in a few hundred steps -- against rnnt +26 % and ctc +7.5 % on the main
path. THE INSERTION POINT MATTERS: the same operation on step97200, 16k steps later with the trunk
at RMS 16.92, costs 0.517 -> 3.168. Migrate the earliest checkpoint that predates the inflation.

Muon's parameter list is untouched by this (a `BiasNorm` contributes no 2-D `Linear.weight`), so
its state carries over verbatim; only AdamW's list gains entries, and its state is remapped by
parameter name rather than by position. The names are read off the BUILT optimizer's
`param_groups`, which is the order its `state_dict` keys index -- not the order
`partition_params` emits, which `_lr_groups` re-buckets by (lr, weight_decay).
"""

import argparse
import math
from typing import cast

import torch

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack

_SUFFIX = ".trunk_norm."


def adamw_param_names(model: torch.nn.Module, adamw: torch.optim.Optimizer) -> list[str]:
    """The names AdamW's state keys index, in param-GROUP order.

    `Optimizer.state_dict` numbers parameters by their position in the concatenated
    `param_groups`, and `_lr_groups` re-buckets `partition_params`' output by (lr, weight_decay)
    before the optimizer ever sees it. Reading the names off the built optimizer is therefore the
    only ordering that lines a saved `state` dict up with the parameters it belongs to; walking
    the model instead silently permutes 115 of the 466 entries, which resumes as a shape mismatch
    inside `Adam.step`.
    """
    by_id = {id(p): name for name, p in model.named_parameters()}
    return [by_id[id(p)] for group in adamw.param_groups for p in group["params"]]


@torch.no_grad()
def measure_trunk_rms(model: TransducerModel, args: argparse.Namespace) -> dict[int, float]:
    """Per stack, the RMS `in_proj` emits on real audio, read BEFORE the new norm sees it.

    Hooked on `in_proj` rather than taken from `ZipformerStack.trunk_rms`, which by then is the
    post-norm value and would report the norm's own output instead of what it has to match.

    The norms are swapped out for `Identity` for the duration, not merely read around. Amplitude
    COMPOUNDS along the trunk, so a stack whose input an upstream norm has already flattened emits
    a different RMS than it does in the checkpoint being migrated. MEASURED on step81000, same
    batch, stacks 1-5:

        norms live at their init gain of 1.0    2.581 / 2.895 / 4.049 / 0.706 / 1.369
        norms disabled (what the run logged)    2.581 / 3.695 / 5.840 / 1.760 / 2.443

    Stack 1 is identical because nothing upstream of it is normalised, and every stack after it is
    wrong -- in the direction that under-sizes the very stacks that have to be clamped.
    """
    tokenizer = SentencePieceTokenizer(args.tokenizer_model)
    dataset = LibriSpeechDataset(
        args.dev_manifest, tokenizer, FeatureCacheReader(args.cache_dir, args.dev_cache_split)
    )
    step = max(1, len(dataset) // args.utts)
    batch = collate_features([dataset[i * step] for i in range(args.utts)])

    rms: dict[int, float] = {}
    handles = []
    saved: list[tuple[ZipformerStack, torch.nn.Module]] = []
    for i, stack in enumerate(m for m in model.modules() if isinstance(m, ZipformerStack)):
        if not isinstance(stack.in_proj, torch.nn.Linear):
            continue

        def hook(_mod, _inp, out, idx=i):
            rms[idx] = (
                float(torch.linalg.vector_norm(out, dtype=torch.float32)) / out.numel() ** 0.5
            )

        handles.append(stack.in_proj.register_forward_hook(hook))
        saved.append((stack, stack.trunk_norm))
        stack.trunk_norm = torch.nn.Identity()
    model.eval()
    model.joint_loss(batch, chunk_size=0)
    for handle in handles:
        handle.remove()
    for stack, norm_module in saved:
        stack.trunk_norm = norm_module
    return rms


def main() -> None:
    cfg = get_config()
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="checkpoint to migrate")
    p.add_argument("--out", required=True, help="where to write the migrated checkpoint")
    p.add_argument("--dev-manifest", default=base.dev_manifest)
    p.add_argument("--tokenizer-model", default=base.tokenizer_model)
    p.add_argument("--cache-dir", default=base.cache_dir)
    p.add_argument("--dev-cache-split", default=base.dev_cache_split)
    p.add_argument("--utts", type=int, default=8, help="dev utterances the RMS is measured over")
    args = p.parse_args()

    if not cfg.model.trunk_norm:
        raise SystemExit("model.trunk_norm is false, so there is nothing to migrate into.")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    model = TransducerModel()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise SystemExit(f"checkpoint has keys this model does not: {unexpected}")
    if not missing:
        raise SystemExit(f"{args.src} already carries the trunk norms.")
    if any(_SUFFIX not in k for k in missing):
        raise SystemExit(f"missing more than the trunk norms, refusing: {missing}")

    rms = measure_trunk_rms(model, args)
    ceiling = cfg.model.trunk_norm_log_scale_max
    stacks = [s for s in model.modules() if isinstance(s, ZipformerStack)]
    print(f"{'stack':>5s} {'measured RMS':>13s} {'log_scale':>10s} {'emits':>7s}")
    for i, stack in enumerate(stacks):
        if i not in rms:
            continue
        log_scale = min(math.log(rms[i]), ceiling)
        with torch.no_grad():
            cast(BiasNorm, stack.trunk_norm).log_scale.fill_(log_scale)
        clipped = " (clipped)" if log_scale < math.log(rms[i]) - 1e-9 else ""
        print(f"{i:5d} {rms[i]:13.3f} {log_scale:10.4f} {math.exp(log_scale):7.3f}{clipped}")

    # Muon's list is unchanged; AdamW's gained the 10 new scalars/vectors, which shifts every
    # index after each insertion point. Line the two up by name and carry the moments across.
    optimizers = build_optimizer(model, cfg.optim)
    fresh = [opt.state_dict() for opt in optimizers]
    if len(fresh) != len(ckpt["optimizers"]):
        raise SystemExit("optimizer count changed; migration cannot map the state.")

    migrated = []
    for opt, fresh_state, old_state in zip(optimizers, fresh, ckpt["optimizers"]):
        n_old = sum(len(g["params"]) for g in old_state["param_groups"])
        n_new = sum(len(g["params"]) for g in fresh_state["param_groups"])
        if n_old == n_new:  # Muon
            migrated.append(old_state)
            continue
        # Dropping the new names from the new order reproduces the OLD order: `_lr_groups`
        # preserves insertion order inside each bucket, and every bucket the trunk norms land in
        # already exists without them, so no bucket changes position either.
        names_new = adamw_param_names(model, opt)
        names_old = [n for n in names_new if _SUFFIX not in n]
        old_index = {name: i for i, name in enumerate(names_old)}
        new_index = {name: i for i, name in enumerate(names_new)}
        if n_old != len(names_old) or n_new != len(names_new):
            raise SystemExit(
                f"AdamW list mismatch: {n_old}/{len(names_old)} {n_new}/{len(names_new)}"
            )
        # param_groups from the fresh optimizer, so the new parameters land in the right lr/wd
        # group; state from the old one, so nothing that was already training loses its moments.
        state = {
            new_index[n]: old_state["state"][old_index[n]]
            for n in names_old
            if old_index[n] in old_state["state"]
        }
        migrated.append({"state": state, "param_groups": fresh_state["param_groups"]})
        print(f"AdamW: {len(state)} of {n_new} parameters carried their moments across")

    ckpt["model"] = model.state_dict()
    ckpt["optimizers"] = migrated
    torch.save(ckpt, args.out)
    print(f"wrote {args.out} at step {ckpt['step']:,}")


if __name__ == "__main__":
    main()
