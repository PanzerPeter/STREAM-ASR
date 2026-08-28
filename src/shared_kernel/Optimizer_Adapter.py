# Builds the optimizer stack: 2D hidden weight matrices -> Muon (spectrally normalized updates),
# everything else (embeddings, biases, norms, input frontend, output heads) -> AdamW.
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import OptimConfig
from src.shared_kernel.Muon_Optimizer import Muon

# Output heads / readouts that must stay on AdamW, matched as substrings of the dotted module name.
# `pred_head` is the BEST-RQ pretrain head (BestRqModel); `ctc_head`/`interctc` are the main and
# auxiliary CTC readouts; `joiner.out` is the transducer readout; `simple_am_proj`/`simple_lm_proj`
# are the pruned objective's linear-joiner readouts, vocab-width like the others. The joiner's
# `enc_proj`/`pred_proj` are hidden 2D projections and are intentionally omitted here so they fall
# through to Muon.
_DEFAULT_HEAD_PATTERNS = (
    "frontend",
    "ctc_head",
    "pred_head",
    "interctc",
    "joiner.out",
    "simple_am_proj",
    "simple_lm_proj",
)


def partition_params(
    model: nn.Module, head_patterns: tuple[str, ...] = _DEFAULT_HEAD_PATTERNS
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    # Tied weights (STREAM-LM ties its readout to the embedding table) surface under two module
    # names; taking both would step the same tensor twice per optimizer step. First placement wins,
    # and module order puts the embedding ahead of the readout.
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        excluded = any(pat in module_name for pat in head_patterns)
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_hidden_matrix = isinstance(module, nn.Linear) and pname == "weight" and p.ndim == 2
            if is_hidden_matrix and not excluded:
                muon.append(p)
            else:
                adamw.append(p)
    return muon, adamw


def _lr_groups(
    params: list[nn.Parameter],
    base_lr: float,
    enc_ids: set[int],
    enc_scale: float,
    weight_decay: float,
) -> list[dict]:
    # One group per distinct (peak LR, weight decay) pair.
    #
    # LR: encoder params get base_lr * enc_scale (warm-started encoder fine-tuned gently while fresh
    # heads adapt at full LR). The trainer rescales every group's lr per step.
    #
    # WEIGHT DECAY: only parameters with ndim >= 2 -- the weight matrices and convolution kernels.
    # For everything else in this model, zero is not a neutral shrink target but a specific
    # degenerate setting, so decaying toward it is a standing pull away from whatever the parameter
    # learned. In the transducer that is 415 tensors / 160,418 elements (0.3 % of the model):
    #
    #     289  *.bias                     the standard exclusion
    #      16  *.res_lambda               -> value residual DISABLED (these learn -1.40 to +0.47)
    #       6  ZipformerStack.bypass      -> the stack skipped entirely
    #       6  SimpleDownsample.weights   -> pre-softmax logits, so 0 forces uniform pooling
    #      98  BiasNorm.log_scale         -> gain 1, which IS the neutral value, see below
    #
    # `log_scale` is the exception that proves the rule and it is exempted for a different reason:
    # 0 there means unit gain, so decay would be a sane prior, but at this model's LRs it is far
    # too weak to be the mechanism that holds the gain down. Decoupled decay reaches equilibrium
    # where lr*wd*theta balances the drift, and the drift for a SCALAR under Adam is the full lr
    # times the gradient's sign consistency -- measured at 0.93 for the three gains that ran away
    # on the 600k run, which puts the equilibrium at |log_scale| ~ 93. What actually bounds the
    # gain is the projection in BiasNorm.project; leaving decay off keeps the two mechanisms from
    # being confused for each other.
    #
    # Being 0.3 % of the parameters, this is not a change in regularisation strength; it removes a
    # systematic pull toward degenerate gate settings. Muon's params are all 2D by construction, so
    # its groups are unaffected and only AdamW splits in two.
    #
    # A param group carries only hyperparameters -- AdamW's update is per parameter -- so bucketing
    # params that share both is exactly equivalent to giving each its own group, and it is what
    # lets the fused/multi-tensor kernels do their job: the transducer's 472 AdamW parameters
    # collapse to 4 groups and the step drops from 6.1 ms to 0.6 ms. Insertion order follows the
    # caller's parameter order, so the group layout is deterministic across runs (and therefore
    # across a checkpoint save/resume).
    buckets: dict[tuple[float, float], list[nn.Parameter]] = {}
    for p in params:
        lr = base_lr * (enc_scale if id(p) in enc_ids else 1.0)
        decay = weight_decay if p.ndim >= 2 else 0.0
        buckets.setdefault((lr, decay), []).append(p)
    return [
        {"params": group, "lr": lr, "weight_decay": decay} for (lr, decay), group in buckets.items()
    ]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> list[torch.optim.Optimizer]:
    fused = next(model.parameters()).is_cuda  # fused AdamW kernel on CUDA; False on CPU (tests)
    enc_ids = {id(p) for name, p in model.named_parameters() if name.startswith("encoder.")}
    enc_scale = cfg.encoder_lr_scale
    # Every group `_lr_groups` emits carries its own `weight_decay`, so the optimizer-level value is
    # only the default a group would inherit if one were ever added without it.
    wd = cfg.weight_decay
    if cfg.optimizer == "adamw":
        groups = _lr_groups(list(model.parameters()), cfg.adamw_lr, enc_ids, enc_scale, wd)
        return [torch.optim.AdamW(groups, weight_decay=wd, betas=(0.9, 0.98), fused=fused)]
    muon_p, adamw_p = partition_params(model)
    muon = Muon(
        _lr_groups(muon_p, cfg.muon_lr, enc_ids, enc_scale, wd),
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.ns_steps,
        weight_decay=wd,
    )
    adamw = torch.optim.AdamW(
        _lr_groups(adamw_p, cfg.adamw_lr, enc_ids, enc_scale, wd),
        weight_decay=wd,
        betas=(0.9, 0.98),
        fused=fused,
    )
    return [muon, adamw]
