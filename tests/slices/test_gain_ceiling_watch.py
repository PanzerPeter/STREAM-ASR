import math

import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.TrainAcousticModel._train_utils import (
    branch_gain_params,
    gains_at_ceiling,
    GainCeilingWatch,
)


def test_a_warm_started_population_at_the_ceiling_is_not_a_warning():
    # The bug this exists for. The transducer warm-starts from bestrq_encoder.pt, and BEST-RQ's
    # own projection had already parked 6 of its 97 gains on the bound -- so a level test
    # (`max_gain >= exp(bound)`) fired on step 0 of a healthy run and on every log line after it,
    # for 600k steps. A signal that is always on carries nothing.
    watch = GainCeilingWatch()
    assert not watch.update(6), "the warm-start population is the reference, not an event"
    for _ in range(100):
        assert not watch.update(6)


def test_warns_when_the_pinned_population_grows_past_the_warm_start():
    watch = GainCeilingWatch()
    watch.update(6)
    assert watch.update(7), "one more gain riding the bound is the event worth a warning"


def test_each_new_high_water_mark_warns_exactly_once():
    # Log lines land every 250 steps for 600k steps. A condition that stays true must not reprint
    # ~2,400 times per level, or it drowns the line that mattered.
    watch = GainCeilingWatch()
    watch.update(6)
    assert watch.update(8)
    for _ in range(50):
        assert not watch.update(8)
    assert watch.update(9)


def test_a_population_that_falls_back_does_not_re_warn_below_its_high_water_mark():
    # The gains train ON the bound (project() keeps them off clamp's dead zone), so the count
    # rattles. Only a NEW worst level is news.
    watch = GainCeilingWatch()
    watch.update(6)
    assert watch.update(9)
    assert not watch.update(6)
    assert not watch.update(9)
    assert watch.update(10)


def test_a_fresh_run_warns_on_the_very_first_pin():
    # No warm start: the reference is 0, so any gain reaching the bound is already an event.
    watch = GainCeilingWatch()
    assert not watch.update(0)
    assert watch.update(1)


def test_replays_the_measured_600k_migration():
    # The 2026-08-09 collapse, as counted off its checkpoints: one log_scale at the (then 2.5)
    # ceiling in step291600, three in step307800, while dev ctc-WER went 0.0818 -> 0.1136. The
    # watch must fire on that migration and must not have fired during the 280k healthy steps
    # before it.
    watch = GainCeilingWatch()
    for _ in range(1120):  # 280k steps at log_every 250
        assert not watch.update(0)
    assert watch.update(1), "the first gain reaching the bound went unreported for 16k steps"
    assert watch.update(3), "the pressure migrating to three tensors is the collapse itself"


def test_the_baseline_survives_a_resume_into_a_pinned_regime():
    # Same argument as GradNormGuard's floor, which rides in the checkpoint for this reason: this
    # run resumed 16 times. A watch that re-latched its baseline from the state it resumes INTO
    # would adopt a diverged population as normal and never fire again.
    fresh = GainCeilingWatch()
    inherited = GainCeilingWatch(baseline=6)
    assert not fresh.update(9), "a cold watch cannot know 9 is abnormal"
    assert inherited.update(9), "an inherited baseline still recognises the migration"


def test_the_baseline_is_reported_for_persistence():
    watch = GainCeilingWatch()
    watch.update(6)
    assert watch.baseline == 6
    watch.update(9)
    assert watch.baseline == 6, "the baseline is the run's reference and never follows it up"


def test_gains_at_ceiling_counts_the_parameters_resting_on_the_bound():
    # project() clamps with clamp_, so a pinned gain lands on the bound to the last bit -- but
    # `forward`'s clamp is on the value USED, and a checkpoint written under a wider bound stores
    # values ABOVE it. Both must count, and an interior gain must not.
    hi = 1.0
    gains = torch.tensor([hi, hi - 1e-6, hi + 3.0, hi - 0.2, -1.0])
    assert int(gains_at_ceiling(gains, hi)) == 3


def test_gains_at_ceiling_returns_a_tensor_so_it_folds_into_the_one_log_sync():
    # Every .item() drains the CUDA queue. The count is stacked with the other logged scalars and
    # synced once, so it has to stay on-device.
    count = gains_at_ceiling(torch.zeros(4), 0.0)
    assert isinstance(count, torch.Tensor) and count.shape == ()


def test_the_configured_window_is_asymmetric():
    # The measured mechanism is one-sided: work falls off as 1/RMS^2 of a stack's input, so
    # AMPLIFICATION is what silences the next stack. Attenuation has no such argument, and the
    # healthy encoder distribution runs well below -1.0 (p05 -1.54 over 97 BiasNorms at warm
    # start), so a symmetric floor clips the normal population instead of a pathology.
    model = get_config().model
    assert model.biasnorm_log_scale_max == 1.0
    assert model.biasnorm_log_scale_min <= -1.54, "the floor must sit below the healthy p05"
    assert math.exp(model.biasnorm_log_scale_max) < 3.0


def test_counts_the_pinned_population_of_a_real_encoder():
    # Integration, over the tree the trainer actually walks: `branch_gain_params` collects 97+
    # 0-d parameters from nested modules, and the count has to survive being stacked with the
    # other logged scalars (which arrive as bf16/fp32 loss terms) into the single host sync.
    encoder = ZipformerEncoder(cmvn_path=None)
    hi = get_config().model.biasnorm_log_scale_max
    params = branch_gain_params(encoder)
    assert len(params) > 50
    with torch.no_grad():
        for param in params[:3]:
            param.fill_(hi + 3.0)  # as a checkpoint written under a wider bound would store them
    gains = torch.stack(params)
    count = gains_at_ceiling(gains, hi)
    assert int(count) == 3

    loss = torch.tensor(4.2, dtype=torch.bfloat16)
    packed = torch.stack([t.float() for t in (loss, gains.max().exp(), count)]).tolist()
    assert int(packed[2]) == 3, "the count must survive the one-sync pack the trainer logs through"


def test_a_resume_does_not_re_warn_a_level_the_run_already_reached():
    # MEASURED on the 600k run, 2026-08-26. The count is an instantaneous per-window population
    # and it rattles: over every 20k-step window from 160k on it runs min 0 / median 2 / max 5.
    # It first reached 5 around step 160k. Resume #4 at step 210,799 then re-printed the warning
    # at 3 (step 212,000) and at 5 (step 213,750), because only the BASELINE rode in the
    # checkpoint and `_high_water` was reconstructed from it -- re-arming the watch against noise
    # it had already reported 50k steps earlier. Four resumes, four rounds of false alarms, on the
    # one signal that is supposed to lead a collapse by 25k steps.
    watch = GainCeilingWatch()
    watch.update(2)
    assert watch.update(5)

    resumed = GainCeilingWatch(baseline=watch.baseline, high_water=watch.high_water)
    assert not resumed.update(3), "a level already reported is not news again after a resume"
    assert not resumed.update(5), "least of all the exact level that was reported"
    assert resumed.update(6), "a genuinely new worst level still warns"


def test_the_high_water_mark_is_reported_for_persistence():
    watch = GainCeilingWatch()
    watch.update(2)
    assert watch.high_water == 2
    watch.update(5)
    assert watch.high_water == 5, "the mark follows the population up, unlike the baseline"
    watch.update(1)
    assert watch.high_water == 5


def test_a_checkpoint_without_a_stored_high_water_mark_falls_back_to_the_baseline():
    # Pre-2026-08-26 `extra` carries `gain_ceiling_baseline` alone. That is the old behaviour and
    # it must survive resuming into this code, not raise.
    watch = GainCeilingWatch(baseline=6)
    assert watch.high_water == 6
    assert not watch.update(6)
    assert watch.update(7)
