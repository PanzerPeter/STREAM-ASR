# Conservative energy trim of leading/trailing silence. Common Voice clips open and close on the
# browser recorder's dead air (often a second or more); LibriSpeech was segmented tight. Every rule
# that is unsure keeps audio: a click before the speech counts as speech, a noisy clip is not
# trimmed, and a kept span that looks too short means the estimate failed, so nothing is cut.
import torch

from src.shared_kernel.Config_Adapter import TrimConfig

_WINDOW_S = 0.025
_HOP_S = 0.010
_FLOOR_QUANTILE = 0.10
_PEAK_QUANTILE = 0.99
# Digital silence would be log10(0); this floor (-100 dB re full scale) is below any real recording.
_POWER_EPS = 1e-10


def speech_span(wave: torch.Tensor, sample_rate: int, cfg: TrimConfig) -> tuple[int, int]:
    """(start_sample, num_samples) of the span to keep; (0, len(wave)) means untrimmed."""
    total = int(wave.numel())
    window, hop = int(_WINDOW_S * sample_rate), int(_HOP_S * sample_rate)
    if total < window:
        return 0, total
    frames = wave.unfold(0, window, hop)
    energy_db = 10.0 * torch.log10(frames.pow(2).mean(dim=-1) + _POWER_EPS)
    floor = float(torch.quantile(energy_db, _FLOOR_QUANTILE))
    peak = float(torch.quantile(energy_db, _PEAK_QUANTILE))
    if peak - floor < cfg.min_dynamic_range_db:
        return 0, total
    speech = (energy_db > floor + cfg.threshold_fraction * (peak - floor)).nonzero().flatten()
    if speech.numel() == 0:
        return 0, total
    margin = int(cfg.margin_s * sample_rate)
    start = max(0, int(speech[0]) * hop - margin)
    end = min(total, int(speech[-1]) * hop + window + margin)
    if end - start < int(cfg.min_span_s * sample_rate):
        return 0, total
    return start, end - start
