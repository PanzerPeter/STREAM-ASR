# src/slices/ExtractFeatures/SpeedPerturb_Transform.py
from fractions import Fraction

import torch
import torchaudio


def apply_speed_perturb(wave: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return wave

    # Duration scales by 1/factor with pitch shifting together — the standard 3-way speed
    # perturbation (Ko et al. 2015). factor<1 (0.9) => slower/longer; factor>1 (1.1) => faster.
    #
    # resample() only uses the new/orig *ratio*, and its polyphase kernel has orig/gcd phases.
    # Passing (16000, int(16000/0.9)=17777) makes those coprime => a ~16000-phase kernel and a
    # >1s convolution per utterance (the Stage-A dataloader bottleneck). The reduced fraction of
    # 1/factor gives the identical ratio with a ~10-phase kernel — ~350x faster, same output.
    ratio = Fraction(1.0 / factor).limit_denominator(1000)  # new/orig = 1/factor
    return torchaudio.functional.resample(wave, ratio.denominator, ratio.numerator)
