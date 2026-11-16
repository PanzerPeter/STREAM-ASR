from src.slices.Evaluate.WordTiming import EdgeCounts, emission_latencies


class _Tok:
    # Word starts are the even ids, so [0, 1] is one two-piece word and [2] another.
    def encode(self, text: str) -> list[int]:
        return []

    def is_word_start(self, token_id: int) -> bool:
        return token_id % 2 == 0


def test_edge_counts_split_first_middle_last_with_deletions():
    edges = EdgeCounts()
    # 6 ref words: first 2 = A B, middle = C D, last 2 = E F. B deleted, E substituted.
    edges.add("A B C D E F", "A C D X F")
    rates = edges.rates()
    assert rates["first_err"] == 0.5 and rates["first_del"] == 0.5
    assert rates["middle_err"] == 0.0
    assert rates["last_err"] == 0.5 and rates["last_del"] == 0.0


def test_emission_latency_reads_the_first_token_of_each_correct_word():
    # Hyp words: [0, 1] -> "HELLO" shown at 0.32 s, [2] -> "WORLD" shown at 0.64 s.
    lats = emission_latencies(
        "HELLO WORLD", "HELLO WORLD", [0.30, 0.70], [0, 1, 2], [0.32, 0.40, 0.64], _Tok()
    )
    assert [round(x, 6) for x in lats] == [0.02, -0.06]


def test_emission_latency_skips_wrong_words_and_mismatched_segmentation():
    assert emission_latencies("A B", "A C", [0.1, 0.2], [0, 2], [0.2, 0.3], _Tok()) == [0.2 - 0.1]
    assert emission_latencies("A B", "A B", [0.1, 0.2], [0], [0.2], _Tok()) == []
