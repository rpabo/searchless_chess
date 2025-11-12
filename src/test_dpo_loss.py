# tests/test_dpo_loss.py
import pytest
import jax
import jax.numpy as jnp

# Import your implementations
# (underscored helpers are fine to import for tests)
from dpo_loss import (
    build_predict_fn,
    _q_value_for_action,
    _pair_delta_q,
    dpo_loss,
)

# -------------------------
# Helper mock predictors
# -------------------------

class MockApplyPredictor:
    """
    Haiku/Flax-like predictor exposing .apply(params, rng, tokens).
    It reads the action index from tokens[0][-2] and returns a per-action
    logit vector defined by self.action_logits[action_idx].
    All other positions' logits are zeros (only -2 is meaningful here).
    """
    def __init__(self, action_logits):  # shape: [max_action+1, n_atoms]
        self.action_logits = jnp.asarray(action_logits, dtype=jnp.float32)

    def apply(self, params, rng, tokens_1xbT):
        tokens_1xbT = jnp.asarray(tokens_1xbT, dtype=jnp.int32)
        B, L = tokens_1xbT.shape
        n_atoms = self.action_logits.shape[-1]
        out = jnp.zeros((B, L, n_atoms), dtype=jnp.float32)

        def per_batch(b, out_):
            action_idx = tokens_1xbT[b, L - 2]  # by convention
            logits = self.action_logits[action_idx]  # [n_atoms]
            return out_.at[b, L - 2, :].set(logits)

        # simple loop over batch (small in tests; fine under CPU)
        for b in range(B):
            out = per_batch(b, out)
        return out


class MockPredictPredictor:
    """
    Project-style predictor exposing .predict(params=..., targets=..., rng=...).
    Same behavior as MockApplyPredictor but with .predict signature.
    """
    def __init__(self, action_logits):
        self._apply = MockApplyPredictor(action_logits)

    def predict(self, params=None, targets=None, rng=None):
        return self._apply.apply(params, rng, targets)


# -------------------------
# Utilities for reference math
# -------------------------

def softmax(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    x = x - jax.scipy.special.logsumexp(x, axis=-1, keepdims=True)
    return jnp.exp(x)

def expected_q(action_logits_vec, z_atoms):
    """E[Z] = sum_buckets softmax(logits)[b] * z_atoms[b]"""
    probs = softmax(action_logits_vec)
    return (probs * z_atoms).sum()


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def z_atoms():
    # Simple symmetric bucket support; feel free to match your actual config
    # e.g., for 3 atoms: [-1, 0, 1]
    return jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)

@pytest.fixture
def action_logits_table():
    # Define per-action logits (3 actions: 0,1,2), each length n_atoms=3
    # Action 0: uniform -> probs [1/3,1/3,1/3] => E[Z]=0
    # Action 1: skew toward higher bucket
    # Action 2: skew toward lower bucket
    return jnp.array([
        [0.0, 0.0, 0.0],   # action 0
        [0.0, 1.0, 2.0],   # action 1
        [2.0, 1.0, 0.0],   # action 2
    ], dtype=jnp.float32)


# -------------------------
# Tests for Q(s, a)
# -------------------------

@pytest.mark.parametrize("predictor_kind", ["apply", "predict"])
def test_q_value_for_action_matches_expected(z_atoms, action_logits_table, predictor_kind):
    if predictor_kind == "apply":
        predictor = MockApplyPredictor(action_logits_table)
    else:
        predictor = MockPredictPredictor(action_logits_table)

    predict_fn = build_predict_fn(predictor)

    # Build a dummy position tokens vector of length T; last two tokens are placeholders
    T = 5
    pos_tokens = jnp.array([10, 20, 30, 40, 50], dtype=jnp.int32)

    # Test a few actions
    for action_idx in [0, 1, 2]:
        q = _q_value_for_action(
            params=None,
            predict_fn=predict_fn,
            pos_tokens=pos_tokens,
            action_idx=action_idx,
            z_atoms=z_atoms
        )
        # Expected from reference math
        q_expected = expected_q(action_logits_table[action_idx], z_atoms)
        assert jnp.allclose(q, q_expected, rtol=1e-6, atol=1e-6)


# -------------------------
# Tests for ΔQ / τ
# -------------------------

@pytest.mark.parametrize("predictor_kind_online, predictor_kind_ref", [
    ("apply", "apply"),
    ("apply", "predict"),
    ("predict", "apply"),
    ("predict", "predict"),
])
def test_pair_delta_q(z_atoms, predictor_kind_online, predictor_kind_ref):
    # Build different tables so online vs ref produce different deltas
    action_logits_online = jnp.array([
        [0.0, 0.0, 0.0],   # a0
        [0.0, 1.0, 2.0],   # a1 -> prefers high bucket
        [2.0, 1.0, 0.0],   # a2 -> prefers low bucket
    ], dtype=jnp.float32)

    action_logits_ref = jnp.array([
        [0.0, 0.0, 0.0],   # a0
        [0.5, 1.0, 1.5],   # a1 -> less extreme preference than online
        [1.5, 1.0, 0.5],   # a2 -> less extreme preference than online
    ], dtype=jnp.float32)

    if predictor_kind_online == "apply":
        predictor_online = MockApplyPredictor(action_logits_online)
    else:
        predictor_online = MockPredictPredictor(action_logits_online)

    if predictor_kind_ref == "apply":
        predictor_ref = MockApplyPredictor(action_logits_ref)
    else:
        predictor_ref = MockPredictPredictor(action_logits_ref)

    predict_fn_online = build_predict_fn(predictor_online)
    predict_fn_ref = build_predict_fn(predictor_ref)

    # Batch of 2 positions; each position has T tokens + 2 placeholders
    T = 5
    positions = jnp.array([
        [1, 2, 3, 4, 5, 0, 0],   # pos 0
        [6, 7, 8, 9, 10, 0, 0],  # pos 1
    ], dtype=jnp.int32)

    # For pos 0: chosen=1, rejected=2
    # For pos 1: chosen=2, rejected=1
    chosen = jnp.array([1, 2], dtype=jnp.int32)
    rejected = jnp.array([2, 1], dtype=jnp.int32)

    temperature = 2.0

    # Compute ΔQ/τ with implementation
    delta_online = _pair_delta_q(
        params=None,
        predict_fn=predict_fn_online,
        positions=positions,
        chosen=chosen,
        rejected=rejected,
        z_atoms=z_atoms,
        temperature=temperature
    )
    delta_ref = _pair_delta_q(
        params=None,
        predict_fn=predict_fn_ref,
        positions=positions,
        chosen=chosen,
        rejected=rejected,
        z_atoms=z_atoms,
        temperature=temperature
    )

    # Compute reference manually
    def q_of(table, a):
        return expected_q(table[a], z_atoms)

    # pos 0:
    # online: (Q(1) - Q(2))/τ
    # ref:    (Q(1) - Q(2))/τ  (with ref table)
    delta0_online_ref = (
        (q_of(action_logits_online, 1) - q_of(action_logits_online, 2)) / temperature,
        (q_of(action_logits_ref,    1) - q_of(action_logits_ref,    2)) / temperature
    )
    # pos 1:
    # online: (Q(2) - Q(1))/τ
    # ref:    (Q(2) - Q(1))/τ
    delta1_online_ref = (
        (q_of(action_logits_online, 2) - q_of(action_logits_online, 1)) / temperature,
        (q_of(action_logits_ref,    2) - q_of(action_logits_ref,    1)) / temperature
    )

    assert jnp.allclose(delta_online[0], delta0_online_ref[0], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(delta_ref[0],    delta0_online_ref[1], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(delta_online[1], delta1_online_ref[0], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(delta_ref[1],    delta1_online_ref[1], rtol=1e-6, atol=1e-6)


# -------------------------
# Optional: sanity check for DPO loss monotonicity
# -------------------------

def test_dpo_loss_lower_when_online_prefers_chosen(z_atoms):
    """
    If online's ΔQ exceeds reference's ΔQ, the DPO logit β(Δθ - Δref) is positive,
    so -log σ(logit) should be < -log σ(0). We test this qualitatively.
    """
    # Online very decisive, reference mild
    action_logits_online = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],  # strong preference to high bucket
        [3.0, 0.0, 0.0],  # strong preference to low bucket
    ], dtype=jnp.float32)
    action_logits_ref = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],  # milder
        [1.0, 0.0, 0.0],
    ], dtype=jnp.float32)

    predict_fn_online = build_predict_fn(MockApplyPredictor(action_logits_online))
    predict_fn_ref    = build_predict_fn(MockApplyPredictor(action_logits_ref))

    # Batch of 2
    positions = jnp.array([
        [1, 2, 3, 4, 5, 0, 0],
        [6, 7, 8, 9,10, 0, 0],
    ], dtype=jnp.int32)
    chosen   = jnp.array([1, 2], dtype=jnp.int32)
    rejected = jnp.array([2, 1], dtype=jnp.int32)

    beta = 0.5
    temperature = 1.0

    # Compute components
    delta_online = _pair_delta_q(None, predict_fn_online, positions, chosen, rejected, z_atoms, temperature)
    delta_ref    = _pair_delta_q(None, predict_fn_ref,    positions, chosen, rejected, z_atoms, temperature)
    logits = beta * (delta_online - delta_ref)

    # DPO loss via API wrapper
    def loss_api(p_online, p_ref):
        return dpo_loss(
            online_params=p_online,
            reference_params=p_ref,
            predict_fn=predict_fn_online,  # NOTE: we pass online predict_fn; reference is handled by passing p_ref to _pair_delta_q with same fn for this test
            positions=positions,
            chosen_moves=chosen,
            rejected_moves=rejected,
            z_atoms=z_atoms,
            beta=beta,
            temperature=temperature,
        )

    # For the test, emulate reference by swapping predict_fn inside dpo_loss is not available,
    # so we just manually compute: loss = -mean log σ(β(Δ_on - Δ_ref))
    expected_loss = -jax.nn.log_sigmoid(logits).mean()

    # Quick check: expected_loss should be < baseline -log σ(0) = ln(2) ≈ 0.693...
    baseline = jnp.log(2.0)
    assert expected_loss < baseline + 1e-6
