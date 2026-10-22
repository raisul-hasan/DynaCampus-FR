"""Research validity safeguards: boundaries, ties, floors and paired resampling."""
import os
import subprocess
import sys

import numpy as np
import pytest

from src.evaluation.chokepoint import (accept, aggregate, calibrate,
    consecutive_tracks, degrade, paired_identity_interval, stable_seed, verification_metrics)


def test_seeds_survive_process_hash_randomization():
    script = "from src.evaluation.chokepoint import stable_seed; print(stable_seed(42,'P1E_S2_C2/0001/00001234.pgm','severe'))"
    results = [subprocess.check_output([sys.executable, '-c', script],
               env={**os.environ, 'PYTHONHASHSEED': str(seed)}).strip() for seed in [1, 9876]]
    assert results[0] == results[1]
    assert stable_seed(42, 'a', 'severe') != stable_seed(42, 'a', 'mild')


def test_tracks_stop_at_gap_and_do_not_overlap():
    records = [dict(identity='a', sequence='P1E_S2_C2', frame=f) for f in [1,2,3,9,10,11,12,13,14,15,16,17,18]]
    tracks = consecutive_tracks(records, length=5)
    assert [[r['frame'] for r in t] for t in tracks] == [[9,10,11,12,13], [14,15,16,17,18]]
    with pytest.raises(ValueError):
        consecutive_tracks(records + [dict(identity='b', sequence='P1E_S2_C2', frame=30)])
    with pytest.raises(ValueError):
        consecutive_tracks(records + [records[0]])


def test_floor_cannot_be_bypassed_by_quality_score_shift():
    assert not accept(.19, 0., .1, 'aosm', floor=.2)
    assert accept(.21, 0., .1, 'aosm', floor=.2)
    assert accept(.5, 1., .5, 'fixed')


def test_calibration_ties_respect_actual_false_accept_budget():
    scores = np.r_[np.full(98, .1), [.8, .8]]
    tau = calibrate(scores, np.ones(100), 'fixed', target=.01)
    assert tau > .8
    assert accept(scores, np.ones(100), tau, 'fixed').sum() == 0
    # 80 calibration unknowns cannot support a nonzero empirical count at 1%.
    tau = calibrate(np.arange(80)/100, np.zeros(80), 'aosm', target=.01)
    assert accept(np.arange(80)/100, np.zeros(80), tau, 'aosm').sum() == 0


def test_known_truth_is_absent_from_calibration_interface():
    import inspect
    assert 'truth' not in str(inspect.signature(calibrate))


def test_degradation_reproducible_and_preserves_source():
    image = np.random.default_rng(5).integers(0, 256, (112,112,3), dtype=np.uint8)
    backup = image.copy()
    a, b = degrade(image, 'severe', 1), degrade(image, 'severe', 1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(image, backup)
    assert not np.array_equal(a, degrade(image, 'severe', 2))
    with pytest.raises(ValueError):
        degrade(image[:96, :96], 'severe', 1)


def test_pooling_uses_raw_norms_and_remains_unit_length():
    feats = np.array([[1.,0.], [0.,1.]])
    mean = aggregate(feats, [1,10], [.2,.1], 'mean')
    np.testing.assert_allclose(mean, [2**-.5,2**-.5], atol=1e-7)
    np.testing.assert_array_equal(aggregate(feats, [1,10], [.2,.1], 'best_norm'), [0,1])
    soft = aggregate(feats, [1,10], [.2,.1], 'softmax_norm_2')
    assert soft[1] > soft[0] and np.isclose(np.linalg.norm(soft), 1)


def test_pairwise_and_query_denominators_differ():
    scores = np.array([[.9,.1], [.8,.2], [.7,.6]])
    result = verification_metrics(scores, np.array([0,1,-1]))
    assert result['genuine_pairs'] == 2
    assert result['impostor_pairs'] == 4


def test_paired_identity_bootstrap_zero_and_constant_differences():
    assert paired_identity_interval([0,0,0], ['a','a','b']) == [0.,0.]
    assert paired_identity_interval([1,1,1], ['a','a','b']) == [1.,1.]
