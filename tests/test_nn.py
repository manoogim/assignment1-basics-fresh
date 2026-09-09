import pytest

from tests.nn_utils import get_lr_cosine_sched, derive_ckpt_name


def test_lr_warmup_zero_skips_warmup():
    assert get_lr_cosine_sched(t=1, alphamax=1.0, alphamin=0.0, tw=0, tc=100) < 1.0  # straight into cosine

def test_lr_reaches_alphamin_exactly_at_tc():
    assert get_lr_cosine_sched(t=100, alphamax=1.0, alphamin=0.1, tw=10, tc=100) == pytest.approx(0.1)

def test_lr_tw_equals_tc_no_crash():
    get_lr_cosine_sched(50, alphamax=1.0, alphamin=0.1, tw=50, tc=50)  # should not raise

def test_lr_post_annealing_flat():
    assert get_lr_cosine_sched(t=200, alphamax=1.0, alphamin=0.1, tw=10, tc=100) == 0.1

def test_lr_never_exceeds_bounds_while_cosining():
    for t in range(20, 200):
        lr = get_lr_cosine_sched(t, alphamax=1.0, alphamin=0.1, tw=20, tc=100)
        assert 0.1 <= lr <= 1.0, f'lr between 0.1 and 1.0 {lr} for t={t}'


# ================================================================
# Tests: correct cyclical naming on valid save steps
# ===================================================================
#
# Behavior:
#   - A checkpoint is saved ONLY when step % save_every_steps == 0
#   - save_event_idx = (step // save_every_steps) - 1
#   - slot = save_event_idx % keep_last
#   - suffix = chr(ord('a') + slot)
#   - filename = f"ckpt_{suffix}.pt"
#
# ================================================================
# Example: keep_last = 4, save_every_steps = 100
# ================================================================
#
# Save steps: 100, 200, 300, 400, 500, 600, 700, 800, ...
#
# step: 100 → save_event_idx=0 → slot=0 → ckpt_a.pt
# step: 200 → save_event_idx=1 → slot=1 → ckpt_b.pt
# step: 300 → save_event_idx=2 → slot=2 → ckpt_c.pt
# step: 400 → save_event_idx=3 → slot=3 → ckpt_d.pt
#
# rollover:
# step: 500 → save_event_idx=4 → slot=0 → ckpt_a.pt
# step: 600 → save_event_idx=5 → slot=1 → ckpt_b.pt
# step: 700 → save_event_idx=6 → slot=2 → ckpt_c.pt
# step: 800 → save_event_idx=7 → slot=3 → ckpt_d.pt
#
# ================================================================
# keep_last = 2, save_every_steps = 100
# ================================================================
#
# cycle: a, b
#
# step: 100 → idx=0 → slot=0 → ckpt_a.pt
# step: 200 → idx=1 → slot=1 → ckpt_b.pt
# step: 300 → idx=2 → slot=0 → ckpt_a.pt
# step: 400 → idx=3 → slot=1 → ckpt_b.pt
#
# ================================================================
# keep_last = 3, save_every_steps = 100
# ================================================================
#
# cycle: a, b, c
#
# step: 100 → idx=0 → slot=0 → ckpt_a.pt
# step: 200 → idx=1 → slot=1 → ckpt_b.pt
# step: 300 → idx=2 → slot=2 → ckpt_c.pt
# step: 400 → idx=3 → slot=0 → ckpt_a.pt
# step: 500 → idx=4 → slot=1 → ckpt_b.pt
# step: 600 → idx=5 → slot=2 → ckpt_c.pt
#
# ================================================================
# Large-step sanity checks
# ================================================================
#
# keep_last = 4, save_every_steps = 100
#
# step: 10_000 → idx=99 → slot=99 % 4 = 3 → ckpt_d.pt
# step: 1_000_000 → idx=9999 → slot=9999 % 4 = 3 → ckpt_d.pt
#
# keep_last = 26, save_every_steps = 100
#
# step: 2600 → idx=25 → slot=25 → ckpt_z.pt
# step: 2700 → idx=26 → slot=0 → ckpt_a.pt
# step: 2800 → idx=27 → slot=1 → ckpt_b.pt
# step: 100_000 → idx=999 → slot=999 % 26 = 11 → ckpt_l.pt
#
# ================================================================

@pytest.mark.parametrize("save_every_steps, keep_last", [
    (100, 2),
    (100, 3),
    (100, 4),
])
def test_ckpt_first_save_event_is_slot_zero(save_every_steps, keep_last):
    assert derive_ckpt_name(step=save_every_steps,
                            save_every_steps=save_every_steps,
                            keep_last=keep_last) == "ckpt_a.pt"


@pytest.mark.parametrize("save_every_steps, keep_last", [
    (100, 2),
    (100, 3),
    (100, 4),
])
def test_ckpt_second_save_event(save_every_steps, keep_last):
    assert derive_ckpt_name(step=2 * save_every_steps,
                            save_every_steps=save_every_steps,
                            keep_last=keep_last) == "ckpt_b.pt"


@pytest.mark.parametrize("save_every_steps, keep_last", [
    (100, 3),
    (100, 4),
])
def test_ckpt_third_save_event(save_every_steps, keep_last):
    assert derive_ckpt_name(step=3 * save_every_steps,
                            save_every_steps=save_every_steps,
                            keep_last=keep_last) == "ckpt_c.pt"


# ================================================================
#  example: keep_last = 4, save_every_steps = 100
# ================================================================

def test_ckpt_keep_last_4_cycle():
    save_every_steps = 100
    keep_last = 4

    assert derive_ckpt_name(100, save_every_steps, keep_last) == "ckpt_a.pt"
    assert derive_ckpt_name(200, save_every_steps, keep_last) == "ckpt_b.pt"
    assert derive_ckpt_name(300, save_every_steps, keep_last) == "ckpt_c.pt"
    assert derive_ckpt_name(400, save_every_steps, keep_last) == "ckpt_d.pt"

    # rollover
    assert derive_ckpt_name(500, save_every_steps, keep_last) == "ckpt_a.pt"
    assert derive_ckpt_name(600, save_every_steps, keep_last) == "ckpt_b.pt"
    assert derive_ckpt_name(700, save_every_steps, keep_last) == "ckpt_c.pt"
    assert derive_ckpt_name(800, save_every_steps, keep_last) == "ckpt_d.pt"


# ================================================================
# Large-step rollover sanity checks
# ================================================================

def test_ckpt_large_steps_rollover():
    save_every_steps = 100
    keep_last = 4

    # step = 10_000 → idx = 99 → slot = 3 → 'd'
    assert derive_ckpt_name(10_000, save_every_steps, keep_last) == "ckpt_d.pt"

    # step = 1_000_000 → idx = 9999 → slot = 3 → 'd'
    assert derive_ckpt_name(1_000_000, save_every_steps, keep_last) == "ckpt_d.pt"
