import numpy as np
from adaptive_ofdm.policy import outcome_labels, choose_continue, paired_summary, binary_entropy_from_llr


def test_signed_gain_equals_paired_bler_reduction():
    rng = np.random.default_rng(19)
    shallow = rng.integers(0, 5, 1000)
    deep = rng.integers(0, 5, 1000)
    result = paired_summary(shallow, deep)
    np.testing.assert_allclose(result["signed_gain"], result["shallow_bler"]-result["deep_bler"], atol=1e-12)


def test_harm_changes_action_despite_same_benefit():
    p = np.array([[.6, .35, .05], [.35, .35, .30]])
    np.testing.assert_array_equal(choose_continue(p, extra_cost=.1, error_penalty=1), [True, False])


def test_decision_agrees_with_enumerated_expected_losses():
    rng = np.random.default_rng(20)
    joint = rng.dirichlet(np.ones(4), 100)  # SS, SF, FS, FF
    p = np.c_[joint[:, 0]+joint[:, 3], joint[:, 2], joint[:, 1]]
    penalty, cost = 4.2, .13
    stop_loss = penalty*(joint[:, 2]+joint[:, 3])
    continue_loss = cost+penalty*(joint[:, 1]+joint[:, 3])
    np.testing.assert_array_equal(choose_continue(p, cost, penalty), continue_loss < stop_loss)


def test_entropy_extremes_and_sign_invariance():
    llr = np.array([[0., 0.], [1000., -1000.], [2., -3.]])
    entropy = binary_entropy_from_llr(llr)
    np.testing.assert_allclose(entropy[0], np.log(2))
    assert entropy[1] == 0
    np.testing.assert_allclose(entropy, binary_entropy_from_llr(-llr))


def test_label_truth_table():
    np.testing.assert_array_equal(outcome_labels([0, 0, 1, 1], [0, 1, 0, 1]), [0, 2, 1, 0])
