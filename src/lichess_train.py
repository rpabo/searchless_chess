"""Training script using Lichess evaluation database for DPO.

Uses pre-computed Stockfish evaluations from Lichess. Processes the entire database
in one pass, saving checkpoints periodically.
"""

import json
import logging
import os
import shutil
import warnings

from absl import app
from absl import flags
import debugpy
from jax import random as jrandom
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import lichess_dpo_generator
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import neural_engines

# Suppress JAX warnings
warnings.filterwarnings('ignore', message='.*PositionalSharding.*')
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)
logging.getLogger('jax._src.dispatch').setLevel(logging.CRITICAL)
logging.getLogger('jax._src.interpreters.pxla').setLevel(logging.CRITICAL)

# Training hyperparameters
_BASE_MODEL = flags.DEFINE_string(
    'base_model',
    '9M',
    'Base model to fine-tune (9M, 136M, 270M).',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    32,
    'Training batch size.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    2e-6,
    'Learning rate for DPO training.',
)

_BETA = flags.DEFINE_float(
    'beta',
    0.1,
    'DPO KL penalty coefficient.',
)

_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    1.0,
    'Temperature for converting action values to policy.',
)

_MAX_GRAD_NORM = flags.DEFINE_float(
    'max_grad_norm',
    1.0,
    'Maximum gradient norm for clipping.',
)

_MAX_KL_DIVERGENCE = flags.DEFINE_float(
    'max_kl_divergence',
    0.5,
    'Stop training if KL divergence exceeds this.',
)

_LICHESS_DB_PATH = flags.DEFINE_string(
    'lichess_db_path',
    '../data/lichess_db_eval.jsonl.zst',
    'Path to Lichess evaluation database.',
)

_CHECKPOINT_EVERY = flags.DEFINE_integer(
    'checkpoint_every',
    100000,
    'Save checkpoint every N pairs trained.',
)

_MAX_PAIRS = flags.DEFINE_integer(
    'max_pairs',
    -1,
    'Maximum number of DPO pairs to generate (-1 = unlimited).',
)

_EMA_DECAY = 0.999


def dpo_loss_fn(
    params,
    reference_params,
    positions,
    chosen_moves,
    rejected_moves,
    predictor,
    beta,
    temperature,
):
  """DPO loss function for action-value models."""
  batch_size = len(chosen_moves)

  # Create sequences: [position, action, dummy_return]
  dummy_returns = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  chosen_actions = chosen_moves[:, None]
  rejected_actions = rejected_moves[:, None]

  chosen_sequences = jnp.concatenate([positions, chosen_actions, dummy_returns], axis=1)
  rejected_sequences = jnp.concatenate([positions, rejected_actions, dummy_returns], axis=1)

  # Get return distributions
  chosen_return_logprobs = predictor.predict(params=params, targets=chosen_sequences, rng=None)[:, -1]
  rejected_return_logprobs = predictor.predict(params=params, targets=rejected_sequences, rng=None)[:, -1]
  ref_chosen_return_logprobs = predictor.predict(params=reference_params, targets=chosen_sequences, rng=None)[:, -1]
  ref_rejected_return_logprobs = predictor.predict(params=reference_params, targets=rejected_sequences, rng=None)[:, -1]

  # Convert to Q-values
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
  return_values = jnp.array(return_buckets_values, dtype=jnp.float32)

  chosen_return_probs = jnp.exp(chosen_return_logprobs)
  rejected_return_probs = jnp.exp(rejected_return_logprobs)
  ref_chosen_return_probs = jnp.exp(ref_chosen_return_logprobs)
  ref_rejected_return_probs = jnp.exp(ref_rejected_return_logprobs)

  chosen_q = jnp.sum(chosen_return_probs * return_values, axis=-1)
  rejected_q = jnp.sum(rejected_return_probs * return_values, axis=-1)
  ref_chosen_q = jnp.sum(ref_chosen_return_probs * return_values, axis=-1)
  ref_rejected_q = jnp.sum(ref_rejected_return_probs * return_values, axis=-1)

  # Temperature-scaled logits
  chosen_logit = chosen_q / temperature
  rejected_logit = rejected_q / temperature
  ref_chosen_logit = ref_chosen_q / temperature
  ref_rejected_logit = ref_rejected_q / temperature

  # DPO loss
  pi_logratios = chosen_logit - rejected_logit
  ref_logratios = ref_chosen_logit - ref_rejected_logit
  logits = beta * (pi_logratios - ref_logratios)
  loss = -jnp.mean(jax.nn.log_sigmoid(logits))

  # Metrics
  accuracy = jnp.mean(pi_logratios > 0)
  kl_chosen = jnp.abs(chosen_q - ref_chosen_q)
  kl_rejected = jnp.abs(rejected_q - ref_rejected_q)
  kl_div = kl_chosen + kl_rejected

  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'kl_divergence_mean': jnp.mean(kl_div),
      'kl_divergence_max': jnp.max(kl_div),
  }

  return loss, metrics


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('='*80)
  logging.info('Lichess DPO Training')
  logging.info('='*80)
  logging.info(f'Base model: {_BASE_MODEL.value}')
  logging.info(f'Batch size: {_BATCH_SIZE.value}')
  logging.info(f'Learning rate: {_LEARNING_RATE.value}')
  logging.info(f'Beta: {_BETA.value}')
  logging.info(f'Temperature: {_TEMPERATURE.value}')
  logging.info(f'Max KL divergence: {_MAX_KL_DIVERGENCE.value}')
  logging.info(f'Checkpoint every: {_CHECKPOINT_EVERY.value} pairs')
  max_pairs_str = 'unlimited' if _MAX_PAIRS.value == -1 else f'{_MAX_PAIRS.value:,}'
  logging.info(f'Max pairs to generate: {max_pairs_str}')
  logging.info(f'Lichess DB path: {_LICHESS_DB_PATH.value}')
  logging.info('='*80)

  # Build model
  logging.info('\nBuilding model...')
  match _BASE_MODEL.value:
    case '9M':
      num_layers, embedding_dim, num_heads = 8, 256, 8
    case '136M':
      num_layers, embedding_dim, num_heads = 8, 1024, 8
    case '270M':
      num_layers, embedding_dim, num_heads = 16, 1024, 8
    case _:
      raise ValueError(f'Unknown model: {_BASE_MODEL.value}')

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=128,  # num_return_buckets
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)

  # Initialize
  rng = jrandom.PRNGKey(42)
  dummy_targets = np.ones((1, 1), dtype=np.uint32)
  initial_params = predictor.initial_params(rng=rng, targets=dummy_targets)

  # Load base model
  logging.info(f'Loading base model {_BASE_MODEL.value}...')
  base_checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{_BASE_MODEL.value}')
  params = training_utils.load_parameters(
      checkpoint_dir=base_checkpoint_dir,
      params=initial_params,
      step=-1,
  )
  params_ema = params

  # Setup
  checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{_BASE_MODEL.value}_lichess')
  os.makedirs(checkpoint_dir, exist_ok=True)

  optimizer = optax.adamw(learning_rate=_LEARNING_RATE.value)
  opt_state = optimizer.init(params)
  reference_params = params

  # Check for cached DPO pairs (JSONL format with metadata)
  cache_file = os.path.join(checkpoint_dir, 'dpo_pairs_cache.jsonl')
  metadata_file = os.path.join(checkpoint_dir, 'cache_metadata.json')

  # Load existing cache if present
  all_positions = []
  all_chosen = []
  all_rejected = []
  cached_position_hashes = set()  # Track which positions we've already cached
  cache_complete = False

  if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
      metadata = json.load(f)
    cache_complete = metadata.get('complete', False)

    if cache_complete:
      logging.info(f'\n=== Loading Complete Cache ===')
      logging.info(f'Cache file: {cache_file}')
      logging.info(f'Loading {metadata["num_pairs"]:,} cached pairs...')

      # Load from JSONL
      with open(cache_file, 'r') as f:
        for line in f:
          pair = json.loads(line)
          pos_array = np.array(pair['position'], dtype=np.uint32)
          all_positions.append(pos_array)
          all_chosen.append(pair['chosen'])
          all_rejected.append(pair['rejected'])
          cached_position_hashes.add(tuple(pair['position']))

      logging.info(f'Loaded {len(all_positions):,} cached DPO pairs')
      logging.info(f'Model: {metadata["model"]}')
    else:
      logging.info(f'\n=== Resuming Cache Generation ===')
      logging.info(f'Found incomplete cache with {metadata["num_pairs"]:,} pairs')
      logging.info(f'Loading existing pairs and continuing...')

      # Load existing pairs and build hash set
      with open(cache_file, 'r') as f:
        for line in f:
          pair = json.loads(line)
          pos_array = np.array(pair['position'], dtype=np.uint32)
          all_positions.append(pos_array)
          all_chosen.append(pair['chosen'])
          all_rejected.append(pair['rejected'])
          cached_position_hashes.add(tuple(pair['position']))

      logging.info(f'Loaded {len(all_positions):,} existing pairs')
      logging.info(f'Will skip these positions when generating new pairs')

  # Check if we already have enough pairs
  if _MAX_PAIRS.value > 0 and len(all_positions) >= _MAX_PAIRS.value:
    logging.info(f'\n=== Already have {len(all_positions):,} pairs (max: {_MAX_PAIRS.value:,}) ===')
    cache_complete = True

  # Generate pairs if cache is not complete
  if not cache_complete:
    if not os.path.exists(metadata_file):
      logging.info(f'\n=== Generating DPO Pairs ===')
      logging.info('Cache not found - generating pairs from entire database...')
      logging.info(f'This will be saved to: {cache_file}\n')

    # Check if we need to generate more pairs
    if _MAX_PAIRS.value > 0:
      pairs_needed = _MAX_PAIRS.value - len(all_positions)
      logging.info(f'Need {pairs_needed:,} more pairs to reach {_MAX_PAIRS.value:,}')

    # Build predict function for generator
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
    predict_fn = neural_engines.wrap_predict_fn(
        predictor=predictor,
        params=params_ema,
        batch_size=64,
    )

    # Initialize generator
    generator = lichess_dpo_generator.LichessDPOGenerator(
        predict_fn=predict_fn,
        database_path=_LICHESS_DB_PATH.value,
    )

    # Open cache file in append mode
    cache_handle = open(cache_file, 'a')

    pair_count = len(all_positions)  # Start from existing count
    new_pairs_added = 0
    skipped_duplicates = 0
    save_interval = 100000  # Save metadata every 100,000 pairs
    last_save_count = pair_count

    try:
      for positions_batch, chosen_batch, rejected_batch, stats in generator.generate_streaming_batches(
          positions_per_batch=10000,
          batch_size=32,
          target_pairs=1000000000,
      ):
        for pos, chosen, rejected in zip(positions_batch, chosen_batch, rejected_batch):
          pos_hash = tuple(pos.tolist())

          # Skip if already cached
          if pos_hash in cached_position_hashes:
            skipped_duplicates += 1
            continue

          # Add new pair
          all_positions.append(pos)
          all_chosen.append(chosen)
          all_rejected.append(rejected)
          cached_position_hashes.add(pos_hash)

          # Write to cache file immediately (JSONL format)
          pair_data = {
              'position': pos.tolist(),
              'chosen': chosen,
              'rejected': rejected,
          }
          cache_handle.write(json.dumps(pair_data) + '\n')
          pair_count += 1
          new_pairs_added += 1

          # Check if we've reached the max pairs limit
          if _MAX_PAIRS.value > 0 and pair_count >= _MAX_PAIRS.value:
            logging.info(f'Reached max pairs limit: {_MAX_PAIRS.value:,}')
            break

        # Flush cache file and update metadata periodically
        if new_pairs_added > 0 and pair_count - last_save_count >= save_interval:
          cache_handle.flush()
          os.fsync(cache_handle.fileno())

          # Update metadata
          metadata = {
              'model': _BASE_MODEL.value,
              'num_pairs': pair_count,
              'complete': False,
          }
          with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

          last_save_count = pair_count
          logging.info(f'Generated {pair_count:,} pairs ({new_pairs_added:,} new, {skipped_duplicates:,} skipped) - checkpoint saved')
        elif new_pairs_added > 0 and new_pairs_added % 10000 == 0:
          logging.info(f'Generated {pair_count:,} pairs ({new_pairs_added:,} new, {skipped_duplicates:,} skipped)...')

        # Break outer loop if we've reached max pairs
        if _MAX_PAIRS.value > 0 and pair_count >= _MAX_PAIRS.value:
          break

      logging.info(f'\nGeneration complete!')
      logging.info(f'Total pairs: {len(all_positions):,}')
      logging.info(f'New pairs added: {new_pairs_added:,}')
      logging.info(f'Duplicates skipped: {skipped_duplicates:,}')

    finally:
      cache_handle.close()

    # Mark cache as complete
    logging.info('Marking cache as complete...')
    metadata = {
        'model': _BASE_MODEL.value,
        'num_pairs': len(all_positions),
        'complete': True,
    }
    with open(metadata_file, 'w') as f:
      json.dump(metadata, f)

    logging.info(f'Cache saved to: {cache_file}')

  # Training loop
  logging.info('\n=== Starting Training ===')
  logging.info(f'Training on {len(all_positions):,} DPO pairs...\n')

  @jax.jit
  def update_step(params, params_ema, ref_params, opt_state, pos, chosen, rejected):
    def loss_fn(p):
      return dpo_loss_fn(p, ref_params, pos, chosen, rejected, predictor,
                        _BETA.value, _TEMPERATURE.value)

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad_norm = optax.global_norm(grads)
    grads, _ = optax.clip_by_global_norm(_MAX_GRAD_NORM.value).update(grads, opt_state)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params_ema = jax.tree.map(
        lambda ema, new: _EMA_DECAY * ema + (1 - _EMA_DECAY) * new,
        params_ema,
        new_params,
    )
    return new_params, new_params_ema, new_opt_state, loss_val, grad_norm, metrics

  # Shuffle pairs for training
  indices = np.arange(len(all_positions))
  np.random.shuffle(indices)

  batch_count = 0
  total_pairs = 0
  total_loss = 0.0
  total_kl = 0.0
  last_checkpoint_pairs = 0

  # Train on all cached pairs
  for batch_start in range(0, len(all_positions), _BATCH_SIZE.value):
    batch_end = min(batch_start + _BATCH_SIZE.value, len(all_positions))
    batch_indices = indices[batch_start:batch_end]

    # Get batch data
    positions_batch = [all_positions[i] for i in batch_indices]
    chosen_batch = [all_chosen[i] for i in batch_indices]
    rejected_batch = [all_rejected[i] for i in batch_indices]

    # Pad sequences
    max_len = max(len(seq) for seq in positions_batch)
    positions = np.zeros((len(positions_batch), max_len), dtype=np.uint32)
    for i, seq in enumerate(positions_batch):
      positions[i, :len(seq)] = seq

    chosen_moves = np.array(chosen_batch, dtype=np.int32)
    rejected_moves = np.array(rejected_batch, dtype=np.int32)

    # Update
    params, params_ema, opt_state, loss_val, grad_norm, metrics = update_step(
        params, params_ema, reference_params, opt_state,
        jnp.array(positions), jnp.array(chosen_moves), jnp.array(rejected_moves)
    )

    total_loss += float(loss_val)
    total_kl += float(metrics['kl_divergence_mean'])
    batch_count += 1
    total_pairs += len(chosen_moves)

    # Check KL divergence safety
    if float(metrics['kl_divergence_mean']) > _MAX_KL_DIVERGENCE.value:
      logging.warning(f'\nKL divergence ({metrics["kl_divergence_mean"]:.4f}) exceeded threshold!')
      logging.warning('Stopping training to prevent catastrophic forgetting.')
      break

    # Progress logging
    if batch_count % 50 == 0:
      avg_loss = total_loss / batch_count
      avg_kl = total_kl / batch_count
      logging.info(f'Pairs: {total_pairs:,} | Batch: {batch_count} | '
                   f'Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | '
                   f'Acc: {metrics["accuracy"]:.2%}')

    # Save checkpoint
    if total_pairs - last_checkpoint_pairs >= _CHECKPOINT_EVERY.value:
      checkpoint_name = f'{total_pairs}'
      logging.info(f'\nSaving checkpoint: {checkpoint_name}')
      step_dir = os.path.join(checkpoint_dir, checkpoint_name)
      os.makedirs(step_dir, exist_ok=True)

      training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params,
                                     step=checkpoint_name, use_ema_params=False)
      training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params_ema,
                                     step=checkpoint_name, use_ema_params=True)
      opt_state_path = os.path.join(step_dir, 'opt_state')
      if os.path.exists(opt_state_path):
        shutil.rmtree(opt_state_path)
      checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
      checkpointer.save(opt_state_path, opt_state)

      last_checkpoint_pairs = total_pairs
      logging.info('Checkpoint saved!\n')

  # Final checkpoint
  if total_pairs > last_checkpoint_pairs:
    checkpoint_name = f'{total_pairs}'
    logging.info(f'\nSaving final checkpoint: {checkpoint_name}')
    step_dir = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(step_dir, exist_ok=True)

    training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params,
                                   step=checkpoint_name, use_ema_params=False)
    training_utils.save_parameters(checkpoint_dir=checkpoint_dir, params=params_ema,
                                   step=checkpoint_name, use_ema_params=True)
    opt_state_path = os.path.join(step_dir, 'opt_state')
    if os.path.exists(opt_state_path):
      shutil.rmtree(opt_state_path)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    checkpointer.save(opt_state_path, opt_state)

  # Summary
  logging.info('\n' + '='*80)
  logging.info('Training Complete!')
  logging.info('='*80)
  logging.info(f'Total pairs trained: {total_pairs:,}')
  logging.info(f'Total batches: {batch_count:,}')
  if batch_count > 0:
    logging.info(f'Average loss: {total_loss/batch_count:.4f}')
    logging.info(f'Average KL: {total_kl/batch_count:.4f}')
  logging.info(f'Checkpoints saved to: {checkpoint_dir}')
  logging.info('='*80)


if __name__ == '__main__':
  debugpy.listen(5678)
  debugpy.wait_for_client()
  app.run(main)
