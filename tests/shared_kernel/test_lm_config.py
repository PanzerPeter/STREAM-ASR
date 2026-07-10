from src.shared_kernel.Config_Adapter import get_config


def test_lm_config_loads():
    lm = get_config().lm
    assert lm.d_model == 320
    assert lm.layers == 16
    assert lm.heads % lm.kv_groups == 0
    assert lm.context_len > 0
    assert lm.subset_words > lm.val_words
