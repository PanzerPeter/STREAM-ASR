from src.shared_kernel.Config_Adapter import get_config


def test_transducer_config_loads():
    t = get_config().transducer
    assert t.predictor_dim == 512 and t.predictor_context == 2 and t.joiner_dim == 512
    assert t.ctc_loss_scale == 0.5 and t.band_width == 16


def test_transducer_train_config_loads():
    tr = get_config().training.transducer
    assert tr.total_steps > 0
    assert 0 in tr.chunk_sizes  # full context stays in the dynamic-chunk pool
    assert all(c % 4 == 0 for c in tr.chunk_sizes)
    assert all(left == -1 or left % 4 == 0 for left in tr.left_contexts)
    # Each branched-cooldown point is the WSD decay start of a budget it is the 3/4 point of.
    assert all(b < tr.total_steps for b in tr.branch_points)


def test_decode_window_is_chunk_aligned():
    d = get_config().decode
    assert d.max_symbols >= 1 and d.tail_chunks >= 1
    assert d.chunk_size % 4 == 0 and (d.left_context == -1 or d.left_context % 4 == 0)
