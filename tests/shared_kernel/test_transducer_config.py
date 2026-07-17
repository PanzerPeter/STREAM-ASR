from src.shared_kernel.Config_Adapter import get_config


def test_transducer_config_loads():
    t = get_config().transducer
    assert t.predictor_dim == 512
    assert t.predictor_context == 2
    assert t.joiner_dim == 512
    assert 0.0 < t.ctc_aux_weight < 1.0
    assert len(t.interctc_layers) == len(t.interctc_weights)
    assert all(0 <= i < len(get_config().model.encoder_dims) for i in t.interctc_layers)


def test_transducer_train_config_loads():
    tr = get_config().training.transducer
    assert tr.total_steps > 0
    assert tr.chunk_sizes[0] == 0  # 0 = full context in the dynamic-chunk set
    assert tr.max_tokens_per_batch > 0
    assert tr.warm_start.endswith("bestrq_encoder.pt")


def test_decode_has_max_symbols():
    assert get_config().decode.max_symbols >= 1
