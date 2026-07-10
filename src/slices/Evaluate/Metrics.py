# src/slices/Evaluate/Metrics.py — corpus WER/CER via jiwer (pure functions).
import jiwer


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.wer(refs, hyps))


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.cer(refs, hyps))
