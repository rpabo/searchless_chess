import types
import numpy as np
import jax
import jax.numpy as jnp
import pytest

# ---- Helpers / fakes ---------------------------------------------------------

class SpyCheckpointManager:
    def __init__(self):
        self.saves = []
        self.waited = False

    def save(self, step, items):
        # Shallow-copy so later mutations don't affect stored snapshot
        snap = {
            k: jax.tree.map(lambda x: x, v) for k, v in items.items()
        }
        self.saves.append((int(step), snap))

    def wait_until_finished(self):
        self.waited = True


class FakeGenerator:
    """Yields exactly one batch and records that it was used."""
    def __init__(self, batch_size=4, seq_len=77 + 2):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.closed = False
        self.calls = []

    def generate_batch(self, num_games, batch_size):
        self.calls.append((num_games, batch_size))
        # Build one simple, valid-looking batch:
        positions = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        chosen   = np.zeros((self.batch_size,), dtype=np.int32)
        rejected = np.ones ((self.batch_size,), dtype=np.int32)
        yield positions, chosen, rejected  # exactly one batch

    def close(self):
        self.closed = True


def make_toy_loss_factory():
    """
    Returns a DPO 'loss factory' compatible with selfplay_train_original,
    but uses a tiny convex loss on a single scalar parameter 'w'.

      loss(online, ref, ...) = (online['w'] - 1.0)^2

    So the gradient nudges w -> 1.0 and we can assert the param changed.
    """
    def factory(_predictor, _z_atoms, beta, temperature):
        def loss_fn(online_params, reference_params, positions, chosen, rejected):
            w = online_params['w']
            return (w - 1.0) ** 2
        return loss_fn
    return factory


# ---- Tests -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def speed_up_jit(monkeypatch):
    # Avoid compilation overhead in unit tests.
    monkeypatch.setattr(jax, "jit", lambda f: f)
    yield


@pytest.fixture
def patch_env(monkeypatch):
    """
    Patch the heavy deps used inside main():
      - base model loader
      - predictor builder
      - dpo loss factory
      - generator
      - training_utils helpers
    Returns a dict of spies to assert on.
    """
    import selfplay_train_original as mod

    spies = {}

    # 1) _load_base_model -> tiny param tree + dummy config
    def fake_load_base_model(model_name):
        params = {"w": jnp.array(0.0, dtype=jnp.float32)}
        DummyCfg = types.SimpleNamespace()
        return params, DummyCfg
    monkeypatch.setattr(mod, "_load_base_model", fake_load_base_model)

    # 2) transformer.build_transformer_predictor -> dummy object
    class DummyPredictor:
        # just expose the attrs the code expects; it won’t be used because we
        # stub the DPO loss
        def initial_params(self, rng, targets):  # not used here
            return {"w": jnp.array(0.0, dtype=jnp.float32)}
    monkeypatch.setattr(mod.transformer, "build_transformer_predictor", lambda cfg: DummyPredictor())

    # 3) DPO loss factory → toy convex loss
    monkeypatch.setattr(mod.dpo_loss, "make_dpo_loss_fn", make_toy_loss_factory())

    # 4) Generator → one fixed batch
    gen = FakeGenerator(batch_size=4, seq_len=(mod.tokenizer.SEQUENCE_LENGTH + 2))
    spies["generator"] = gen
    class GenCtor:
        def __call__(self, **kwargs):
            # expose ctor args if you want to assert on them later
            spies["gen_ctor_kwargs"] = kwargs
            return gen
    monkeypatch.setattr(mod.dpo_generator, "DPOSelfPlayGenerator", GenCtor())

    # 5) training_utils helpers to no-ops / spies
    monkeypatch.setattr(mod.training_utils, "replicate", lambda x, *_args, **_kw: x)
    monkeypatch.setattr(mod.jax.lax, "with_sharding_constraint", lambda x, *_: x)
    ckpt_mgr = SpyCheckpointManager()
    spies["ckpt_mgr"] = ckpt_mgr
    monkeypatch.setattr(
        mod.training_utils,
        "get_checkpoint_manager",
        lambda **kwargs: ckpt_mgr,
    )
    # load_parameters just returns the passed shape
    monkeypatch.setattr(
        mod.training_utils,
        "load_parameters",
        lambda checkpoint_dir, params, step, use_ema_params: params,
    )

    # Create a fake positional sharding path that won't blow up
    class DummySharding:
        def reshape(self, *_): return self
    monkeypatch.setattr(mod.mesh_utils, "create_device_mesh", lambda shape: (0,))
    monkeypatch.setattr(mod.jax.sharding, "PositionalSharding", lambda *_: DummySharding())

    return spies


def set_small_flags(mod):
    """Shrink the training loop to be unit-test friendly."""
    mod._NUM_ITERATIONS.value = 1
    mod._GAMES_PER_ITERATION.value = 1
    mod._BATCH_SIZE.value = 4
    mod._GRADIENT_STEPS_PER_ITERATION.value = 5
    mod._SAVE_FREQUENCY.value = 1            # force a save so we can inspect params
    mod._UPDATE_REF_EVERY.value = 99         # don’t bother updating ref in this short run
    mod._RESUME.value = False


def test_main_runs_and_updates_params(monkeypatch, patch_env):
    """End-to-end smoke test: runs 1 tiny iteration and updates params."""
    import selfplay_train_original as mod
    set_small_flags(mod)

    # Run main (no argv)
    mod.main([])

    # Assert generator used & closed
    gen = patch_env["generator"]
    assert gen.calls, "Generator was never called"
    assert gen.closed, "Generator was not closed"

    # Check a checkpoint was saved and params changed away from 0.0 toward 1.0
    saves = patch_env["ckpt_mgr"].saves
    assert saves, "No checkpoint save recorded"
    _, snapshot = saves[-1]
    final_w = snapshot["params"]["w"]
    # It should have moved from 0.0 in the direction of 1.0
    assert float(final_w) != 0.0
    # And the manager finalized
    assert patch_env["ckpt_mgr"].waited is True


def test_loss_called_with_valid_shapes(monkeypatch, patch_env):
    """We spy on the toy loss to assert it gets arrays of expected shapes."""
    import selfplay_train_original as mod
    set_small_flags(mod)

    seen = {}

    # wrap the toy factory to capture inputs
    base_factory = make_toy_loss_factory()
    def spy_factory(predictor, z_atoms, beta, temperature):
        # record z_atoms monotonicity (sanity check in main)
        seen["z_atoms_shape"] = tuple(np.array(z_atoms).shape)
        seen["z_atoms_sorted"] = bool(np.all(np.diff(np.array(z_atoms)) > 0))
        loss = base_factory(predictor, z_atoms, beta, temperature)
        def spy_loss(online_params, reference_params, positions, chosen, rejected):
            seen["pos_shape"] = tuple(np.array(positions).shape)
            seen["chosen_shape"] = tuple(np.array(chosen).shape)
            seen["rejected_shape"] = tuple(np.array(rejected).shape)
            return loss(online_params, reference_params, positions, chosen, rejected)
        return spy_loss

    monkeypatch.setattr(mod.dpo_loss, "make_dpo_loss_fn", spy_factory)

    mod.main([])

    assert seen["z_atoms_shape"][0] > 0 and seen["z_atoms_sorted"]
    assert seen["pos_shape"][1] == (mod.tokenizer.SEQUENCE_LENGTH + 2)
    assert seen["chosen_shape"] == seen["rejected_shape"]


def test_resume_without_checkpoint_does_not_crash(monkeypatch, patch_env):
    """Sets RESUME=True with no checkpoint dir; main should still complete."""
    import selfplay_train_original as mod
    set_small_flags(mod)
    mod._RESUME.value = True  # no directory created; code should fall back gracefully
    mod.main([])
    # Still should have trained/saved once
    assert patch_env["ckpt_mgr"].saves