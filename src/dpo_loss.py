"""DPO (Direct Preference Optimization) loss for chess move learning.

This implements Direct Preference Optimization from Rafailov et al. 2023
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model".

DPO directly optimizes a model to prefer better moves (from Stockfish) over
worse moves (the model's own mistakes) without needing a separate reward model.

The loss function is:
  L_DPO = -E[log σ(β * (r_θ(s,w) - r_θ(s,l)))]

where:
  r_θ(s,a) = log π_θ(a|s) - log π_ref(a|s)
  w = winning/better move (Stockfish)
  l = losing/worse move (LLM)
  β = KL penalty coefficient
  σ = sigmoid function

References:
  Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
  Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
  https://arxiv.org/abs/2305.18290
"""

import jax
import jax.numpy as jnp
from scipy.special import logsumexp


# def compute_move_log_probs(
#     params,
#     predictor,
#     positions: jnp.ndarray,
#     moves: jnp.ndarray,
#     z_atoms: jnp.ndarray,
#     temperature: float = 1.0,
# ) -> jnp.ndarray:
#   """Computes log probabilities of moves given positions.

#   For action-value models, we derive move probabilities from Q-values:
#     π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)

#   where Q(s,a) = E[Z(s,a)] (expected return).

#   Args:
#     params: Model parameters.
#     predictor: Transformer predictor function.
#     positions: [batch_size, seq_len] Tokenized positions (FENs).
#     moves: [batch_size] Move indices to compute log probs for.
#     z_atoms: [n_atoms] Support atoms for Q-value distribution.
#     temperature: Temperature for softmax over Q-values.

#   Returns:
#     [batch_size] Log probabilities of the specified moves.
#   """
#   batch_size = positions.shape[0]
#   n_atoms = z_atoms.shape[0]

#   # To get Q-values for all actions, we need to query the model for each action
#   # The model takes [position_tokens, action, dummy_return] as input
#   # and outputs a distribution over returns for that action

#   # We'll compute Q-values for all possible actions (inefficient but correct)
#   # In practice, we only need Q-values for legal moves, but for simplicity:

#   # For each position, we need to:
#   # 1. Get Q-value distributions for all actions
#   # 2. Compute expected Q-values
#   # 3. Apply softmax to get move probabilities
#   # 4. Extract log prob for the specified move

#   # This is computationally expensive - we need to do num_actions forward passes
#   # per position. For now, let's compute it for the moves we care about.

#   # Actually, looking at the model architecture more carefully:
#   # The model outputs [batch_size, seq_len, n_atoms]
#   # where seq_len includes position tokens + action token + return token
#   # The output at position -2 (action token) gives the Q-value distribution

#   # But we need Q-values for ALL actions to compute the probability.
#   # This is the key challenge: the action-value model requires evaluating
#   # all actions to get a policy distribution.

#   # For DPO, we need both the chosen and rejected move probabilities.
#   # Let's implement a helper that computes Q-values for specific actions.

#   def get_q_value(pos_tokens, action_idx):
#     """Get Q-value for a specific action."""
#     # Create input: [pos_tokens, action_idx, 0]
#     dummy_return = jnp.array([0], dtype=jnp.int32)
#     action_token = jnp.array([action_idx], dtype=jnp.int32)
#     input_seq = jnp.concatenate([pos_tokens, action_token, dummy_return])

#     # Get distribution over returns
#     log_probs = predictor.predict(
#         params=params,
#         targets=input_seq[None, :],
#         rng=None
#     )[0, -2]  # Distribution at action position

#     probs = jnp.exp(log_probs)

#     # Expected Q-value
#     q_value = jnp.sum(probs * z_atoms)
#     return q_value

#   # For DPO, we actually need to compute probabilities over a set of legal moves
#   # This is problematic because we don't have legal move masks here.
#   #
#   # Alternative approach: Use a simplified version where we approximate
#   # the probability using only the Q-values of chosen vs rejected:
#   #   π(a|s) ≈ exp(Q(s,a)/τ) / [exp(Q(s,chosen)/τ) + exp(Q(s,rejected)/τ)]
#   #
#   # This is an approximation but makes the computation tractable.
#   #
#   # Actually, for DPO we don't need the exact probabilities, we need
#   # log π(a|s). If we assume the partition function is approximately constant
#   # (or cancels out in the ratio), we can use:
#   #   log π(a|s) ≈ Q(s,a) / τ - log Z(s)
#   #
#   # where Z(s) is the partition function. In the DPO loss, we compute:
#   #   log π(chosen) - log π(rejected) = [Q(chosen) - Q(rejected)] / τ
#   #
#   # The partition function cancels! This makes DPO tractable with action-value models.

#   # So we just need to compute Q-values for the specified moves
#   log_probs_list = []

#   for i in range(batch_size):
#     pos_tokens = positions[i, :-2]  # Remove action and return tokens
#     action_idx = moves[i]

#     # Create input sequence
#     dummy_return = jnp.array([0], dtype=jnp.int32)
#     action_token = jnp.array([action_idx], dtype=jnp.int32)
#     input_seq = jnp.concatenate([pos_tokens, action_token, dummy_return])

#     # Get Q-value distribution
#     log_dist = predictor.predict(
#         params=params,
#         targets=input_seq[None, :],
#         rng=None
#     )[0, -2]  # [n_atoms]

#     probs = jnp.exp(log_dist)
#     q_value = jnp.sum(probs * z_atoms)

#     # Log probability (up to partition function)
#     log_prob = q_value / temperature
#     log_probs_list.append(log_prob)

#   return jnp.array(log_probs_list)


# def dpo_loss(
#     online_params,
#     reference_params,
#     predictor,
#     positions: jnp.ndarray,
#     chosen_moves: jnp.ndarray,
#     rejected_moves: jnp.ndarray,
#     z_atoms: jnp.ndarray,
#     beta: float = 0.1,
#     temperature: float = 1.0,
# ) -> jnp.ndarray:
#   """Computes DPO loss for preference pairs.

#   The loss encourages the model to assign higher probability to chosen moves
#   (from Stockfish) compared to rejected moves (model's mistakes), relative
#   to a reference model.

#   Args:
#     online_params: Current model parameters.
#     reference_params: Reference model parameters (frozen).
#     predictor: Transformer predictor function.
#     positions: [batch_size, seq_len] Tokenized positions.
#     chosen_moves: [batch_size] Better move indices (Stockfish).
#     rejected_moves: [batch_size] Worse move indices (model's moves).
#     z_atoms: [n_atoms] Support atoms for Q-value distributions.
#     beta: KL penalty coefficient (default 0.1).
#     temperature: Temperature for action selection (default 1.0).

#   Returns:
#     Scalar loss value.
#   """
#   # Compute log probabilities from online model
#   log_pi_chosen = compute_move_log_probs(
#       online_params, predictor, positions, chosen_moves, z_atoms, temperature
#   )
#   log_pi_rejected = compute_move_log_probs(
#       online_params, predictor, positions, rejected_moves, z_atoms, temperature
#   )

#   # Compute log probabilities from reference model (stop gradient)
#   log_ref_chosen = compute_move_log_probs(
#       jax.lax.stop_gradient(reference_params),
#       predictor,
#       positions,
#       chosen_moves,
#       z_atoms,
#       temperature
#   )
#   log_ref_rejected = compute_move_log_probs(
#       jax.lax.stop_gradient(reference_params),
#       predictor,
#       positions,
#       rejected_moves,
#       z_atoms,
#       temperature
#   )

#   # Compute reward ratios: r_θ(s,a) = log π_θ(a|s) - log π_ref(a|s)
#   r_chosen = log_pi_chosen - log_ref_chosen
#   r_rejected = log_pi_rejected - log_ref_rejected

#   # DPO loss: -E[log σ(β * (r_chosen - r_rejected))]
#   logits = beta * (r_chosen - r_rejected)
#   loss = -jax.nn.log_sigmoid(logits).mean()

#   return loss

def build_predict_fn(predictor):
  """
  Returns a callable: predict_fn(params, tokens[None, :]) -> [1, L, n_atoms]
  Works for predictors exposing:
    - .apply(params, rng, tokens)
    - .predict(params=..., targets=..., rng=...)
    - or a plain callable(params, rng, tokens)
  """
  if hasattr(predictor, "apply"):
    def _predict_fn(params, tokens_1xbT):
      return predictor.apply(params, None, tokens_1xbT)
    return _predict_fn

  if hasattr(predictor, "predict"):
    def _predict_fn(params, tokens_1xbT):
      return predictor.predict(params=params, targets=tokens_1xbT, rng=None)
    return _predict_fn

  # Fallback: assume callable(params, rng, tokens)
  def _predict_fn(params, tokens_1xbT):
    return predictor(params, None, tokens_1xbT)
  return _predict_fn

def _q_value_for_action(params, predict_fn, pos_tokens, action_idx, z_atoms):
  """Q(s,a) from AV head: E[Z(s,a)], where Z is return distribution."""
  # Ensure everything is a JAX array with stable dtypes
  pos_tokens = jnp.asarray(pos_tokens, dtype=jnp.int32)
  action_idx = jnp.asarray(action_idx, dtype=jnp.int32)
  z = jnp.asarray(z_atoms, dtype=jnp.float32)

  seq = jnp.concatenate([
      pos_tokens,
      jnp.array([action_idx], dtype=jnp.int32),
      jnp.array([0], dtype=jnp.int32),   # dummy return bucket
  ])

  # Model forward -> raw logits/log-probs over return buckets at the action slot
  out = predict_fn(params, seq[None, :])        # [1, L, n_atoms]
  logit_vec = jnp.asarray(out)[0, -2].astype(jnp.float32)  # [n_atoms]

  # Pure-JAX normalization (no SciPy/NumPy): use log_softmax for stability
  log_probs = jax.nn.log_softmax(logit_vec, axis=-1)  # [n_atoms]
  probs = jnp.exp(log_probs)                          # [n_atoms]

  # Expected return E[Z] = sum_buckets p(b) * z(b)
  return jnp.dot(probs, z)  # scalar Q(s,a)

def _pair_delta_q(params, predict_fn, positions, chosen, rejected, z_atoms, temperature):
  """Vectorized ΔQ/τ over the batch."""
  pos_only = positions[:, :-2]  # strip [action, return] placeholders
  q_w = jax.vmap(_q_value_for_action, in_axes=(None, None, 0, 0, None))(
      params, predict_fn, pos_only, chosen, z_atoms
  )
  q_l = jax.vmap(_q_value_for_action, in_axes=(None, None, 0, 0, None))(
      params, predict_fn, pos_only, rejected, z_atoms
  )
  return (q_w - q_l) / temperature  # [B]

def dpo_loss(
    online_params,
    reference_params,
    predict_fn,                 
    positions: jnp.ndarray,
    chosen_moves: jnp.ndarray,
    rejected_moves: jnp.ndarray,
    z_atoms: jnp.ndarray,
    beta: float = 0.1,
    temperature: float = 1.0,
) -> jnp.ndarray:
  """JIT-friendly DPO with pairwise normalization for AV models."""
  delta_online = _pair_delta_q(
      online_params, predict_fn, positions, chosen_moves, rejected_moves, z_atoms, temperature
  )
  delta_ref = _pair_delta_q(
      jax.lax.stop_gradient(reference_params), predict_fn, positions, chosen_moves, rejected_moves, z_atoms, temperature
  )
  logits = beta * (delta_online - delta_ref)          # [B]
  return -jax.nn.log_sigmoid(logits).mean()

def make_dpo_loss_fn(predictor, z_atoms, beta=0.1, temperature=1.0):
  """Factory returning (online_params, reference_params, batch) -> scalar loss."""
  predict_fn = build_predict_fn(predictor)

  def loss_fn(online_params, reference_params, positions, chosen_moves, rejected_moves):
    return dpo_loss(
        online_params=online_params,
        reference_params=reference_params,
        predict_fn=predict_fn,
        positions=positions,
        chosen_moves=chosen_moves,
        rejected_moves=rejected_moves,
        z_atoms=z_atoms,
        beta=beta,
        temperature=temperature,
    )
  return loss_fn