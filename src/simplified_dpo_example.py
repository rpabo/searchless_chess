# dpo_jax_example.py
"""
Simplified JAX/Flax implementation of DPO loss for direct preference optimization.
Designed for adaptation into a chess-policy pipeline.
"""

from typing import Any, Tuple
import debugpy
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial

class TinyPolicyModel(nn.Module):
    """Toy policy network for demonstration (replace with your chess-predictor)."""
    # num_actions: e.g. number of legal moves or action space size
    num_actions: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch, feature_dim] (e.g., board encoding)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        logits = nn.Dense(self.num_actions)(h)
        log_probs = jax.nn.log_softmax(logits)
        return log_probs


def dpo_loss_fn(
    online_logp_chosen: jnp.ndarray,
    online_logp_rejected: jnp.ndarray,
    ref_logp_chosen: jnp.ndarray,
    ref_logp_rejected: jnp.ndarray,
    beta: float = 0.1,
) -> jnp.ndarray:
    """
    Compute the DPO loss:
      L = - E[ log σ( β * ( (logπ_θ(c) − logπ_ref(c)) − (logπ_θ(r) − logπ_ref(r)) ) ) ]
    
    Args:
      online_logp_chosen: [batch] log π_θ(c)
      online_logp_rejected: [batch] log π_θ(r)
      ref_logp_chosen: [batch] log π_ref(c)
      ref_logp_rejected: [batch] log π_ref(r)
      beta: coefficient
    Returns:
      scalar loss
    """
    r_chosen = online_logp_chosen - ref_logp_chosen
    r_rejected = online_logp_rejected - ref_logp_rejected
    logits = beta * (r_chosen - r_rejected)
    loss = - jnp.mean(jax.nn.log_sigmoid(logits))
    return loss


@partial(jax.jit, static_argnums=(0,))
def train_step(
    online_params: Any,
    ref_params: Any,
    optimizer_state: Any,
    batch: dict,
    model: TinyPolicyModel,
    optimizer: optax.GradientTransformation,
    beta: float
) -> Tuple[Any, Any, dict]:
    """
    A single training step performing DPO update of the online policy given
    a frozen reference policy.
    
    batch: {
      "features": [batch, feat_dim],
      "chosen_actions": [batch],
      "rejected_actions": [batch],
    }
    """
    features = batch["features"]
    chosen = batch["chosen_actions"]
    rejected = batch["rejected_actions"]
    
    def loss_fn(params):
        # online model forward
        logp_online = model.apply({"params": params}, features)
        # reference model forward
        logp_ref = model.apply({"params": ref_params}, features)
        
        # select log-probs
        online_lp_ch = jnp.take_along_axis(logp_online, chosen[:, None], axis=1).squeeze(axis=1)
        online_lp_rej = jnp.take_along_axis(logp_online, rejected[:, None], axis=1).squeeze(axis=1)
        ref_lp_ch = jnp.take_along_axis(logp_ref, chosen[:, None], axis=1).squeeze(axis=1)
        ref_lp_rej = jnp.take_along_axis(logp_ref, rejected[:, None], axis=1).squeeze(axis=1)
        
        loss_value = dpo_loss_fn(
            online_lp_ch, online_lp_rej,
            ref_lp_ch, ref_lp_rej,
            beta=beta
        )
        return loss_value
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = grad_fn(online_params)
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params=online_params)
    new_params = optax.apply_updates(online_params, updates)
    
    metrics = {"dpo_loss": loss_val}
    return new_params, new_opt_state, metrics


def initialize_models(rng: jax.random.PRNGKey, feat_dim: int, num_actions: int, init_scale: float = 1e-2):
    model = TinyPolicyModel(num_actions=num_actions)
    dummy = jnp.zeros((1, feat_dim))
    params = model.init(rng, dummy)["params"]
    return model, params


if __name__ == "__main__":
    # Demo usage
    import numpy as np
    debugpy.listen(5678)
    debugpy.wait_for_client()    
    
    rng = jax.random.PRNGKey(0)
    feat_dim = 20
    num_actions = 10
    beta = 0.1
    batch_size = 4
    
    model, online_params = initialize_models(rng, feat_dim, num_actions)
    # copy online to reference
    ref_params = online_params
    
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(online_params)
    
    # dummy data
    features = jnp.array(np.random.randn(batch_size, feat_dim), dtype=jnp.float32)
    chosen = jnp.array(np.random.randint(0, num_actions, size=(batch_size,)), dtype=jnp.int32)
    rejected = jnp.array(np.random.randint(0, num_actions, size=(batch_size,)), dtype=jnp.int32)
    batch = {"features": features, "chosen_actions": chosen, "rejected_actions": rejected}
    
    online_params, opt_state, metrics = train_step(
        online_params,
        ref_params,
        opt_state,
        batch,
        model,
        optimizer,
        beta
    )
    print("Metrics:", metrics)
