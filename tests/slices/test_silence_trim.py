import math

import torch

from src.shared_kernel.Config_Adapter import TrimConfig
from src.slices.BuildManifest.SilenceTrim import speech_span

_SR = 16000
_CFG = TrimConfig(threshold_fraction=0.25, margin_s=0.5, min_dynamic_range_db=15.0, min_span_s=1.0)


def _tone(seconds: float) -> torch.Tensor:
    t = torch.arange(int(seconds * _SR)) / _SR
    return 0.3 * torch.sin(2 * math.pi * 220 * t)


def _hiss(seconds: float, amplitude: float) -> torch.Tensor:
    return amplitude * torch.randn(int(seconds * _SR), generator=torch.Generator().manual_seed(0))


def test_trims_silence_to_the_margin_and_never_into_the_tone():
    lead, tone, tail = 2.0, 1.5, 3.0
    wave = torch.cat([_hiss(lead, 1e-4), _tone(tone), _hiss(tail, 1e-4)])
    start, num = speech_span(wave, _SR, _CFG)
    tone_start, tone_end = int(lead * _SR), int((lead + tone) * _SR)
    assert start <= tone_start and start + num >= tone_end  # the tone is whole
    margin = int(_CFG.margin_s * _SR)
    assert abs(start - (tone_start - margin)) <= 0.03 * _SR  # cut to the margin, one window slack
    assert abs(start + num - (tone_end + margin)) <= 0.03 * _SR


def test_low_dynamic_range_clip_is_untouched():
    # Loud noise with a tone barely above it: the floor/peak estimate cannot be trusted.
    wave = torch.cat([_hiss(1.0, 0.2), _tone(1.0) * 0.5 + _hiss(1.0, 0.2), _hiss(1.0, 0.2)])
    assert speech_span(wave, _SR, _CFG) == (0, wave.numel())


def test_span_under_the_minimum_is_untouched():
    # Two margins already exceed the minimum, so it binds only at a clip edge: a lone burst at
    # record start (a click) would keep ~0.5 s. That is the estimate failing, not a short clip.
    wave = torch.cat([_tone(0.02), _hiss(3.0, 1e-4)])
    assert speech_span(wave, _SR, _CFG) == (0, wave.numel())


def test_clip_shorter_than_a_window_is_untouched():
    wave = torch.zeros(100)
    assert speech_span(wave, _SR, _CFG) == (0, 100)
