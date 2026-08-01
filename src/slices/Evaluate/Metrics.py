# Corpus WER/CER via jiwer (pure functions).
#
# The *_errors helpers return raw counts rather than a ratio, because both callers need counts:
# a corpus score is sum(errors)/sum(reference length) (a mean of per-utterance rates would weight
# a 3-word utterance like a 40-word one), and the rescoring-weight sweep scores a whole
# (alpha, beta) grid by summing precomputed per-hypothesis counts instead of re-aligning at every
# point. Alignment is jiwer's in both paths, so the counts and corpus_wer agree by construction.
import jiwer


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    # (edit distance in words, reference word count). ref_words = S + D + H reconstructs the
    # reference length from the alignment, so an empty hypothesis still reports the true divisor.
    out = jiwer.process_words(ref, hyp)
    return (
        out.substitutions + out.deletions + out.insertions,
        out.substitutions + out.deletions + out.hits,
    )


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    out = jiwer.process_characters(ref, hyp)
    return (
        out.substitutions + out.deletions + out.insertions,
        out.substitutions + out.deletions + out.hits,
    )


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.wer(refs, hyps))


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.cer(refs, hyps))
