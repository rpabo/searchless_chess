"""Self-play training script with DPO (Direct Preference Optimization)."""

from collections.abc import Sequence
import copy
import functools
import logging as python_logging
import os
import warnings

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax

# Suppress harmless warnings and verbose logging
warnings.filterwarnings('ignore', message='.*sharding.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/XLA warnings
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Suppress TPU warnings

# Suppress verbose library logging (keep training script messages)
python_logging.getLogger('orbax.checkpoint').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.sharding').setLevel(python_logging.ERROR)
python_logging.getLogger('jax._src.xla_bridge').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.array_metadata_store').setLevel(python_logging.WARNING)
python_logging.getLogger('jax._src.sharding_impls').setLevel(python_logging.ERROR)

from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import orbax.checkpoint as ocp

from searchless_chess.src import constants
from searchless_chess.src import dpo_generator
from searchless_chess.src import dpo_loss
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import constants as engine_constants
from searchless_chess.src.engines import neural_engines


_BASE_MODEL = flags.DEFINE_enum(
    'base_model',
    '9M',
    ['9M', '136M', '270M'],
    'The base model to start from for self-play training.',
)

_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations',
    10,
    'Number of self-play training iterations.',
)

_GAMES_PER_ITERATION = flags.DEFINE_integer(
    'games_per_iteration',
    20,
    'Number of self-play games per iteration.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size for training.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    1e-4,
    'Learning rate for Adam optimizer.',
)

_GRADIENT_STEPS_PER_ITERATION = flags.DEFINE_integer(
    'gradient_steps_per_iteration',
    50,
    'Number of gradient steps per self-play iteration (reduced to prevent overfitting).',
)

_STOCKFISH_TIME = flags.DEFINE_float(
    'stockfish_time',
    0.1,
    'Time limit for Stockfish analysis per position (seconds).',
)

_STOCKFISH_DEPTH = flags.DEFINE_integer(
    'stockfish_depth',
    20,
    'Depth for Stockfish analysis (starts here, increases with curriculum).',
)

_EVAL_THRESHOLD = flags.DEFINE_float(
    'eval_threshold',
    0.3,
    'Minimum evaluation difference (in pawns) to create a preference pair.',
)

_BETA = flags.DEFINE_float(
    'beta',
    0.1,
    'KL penalty coefficient for DPO loss.',
)

_UPDATE_REF_EVERY = flags.DEFINE_integer(
    'update_ref_every',
    3,
    'Update reference model every N iterations.',
)

_DPO_TEMPERATURE = flags.DEFINE_float(
    'dpo_temperature',
    1.0,
    'Temperature for DPO probability computation.',
)

_SAVE_FREQUENCY = flags.DEFINE_integer(
    'save_frequency',
    1,
    'How often to save checkpoints (in iterations).',
)

_RESUME = flags.DEFINE_boolean(
    'resume',
    False,
    'Resume training from latest checkpoint if available.',
)

def _load_base_model(model_name: str) -> tuple[hk.Params, transformer.TransformerConfig]:
  """Loads a pretrained base model.

  Args:
    model_name: Name of the model ('9M', '136M', or '270M').

  Returns:
    Tuple of (parameters, config).
  """
  logging.info(f'Loading base model: {model_name}')

  # Determine config based on model size
  num_return_buckets = 128

  if model_name == '9M':
    num_layers = 8
    embedding_dim = 256
    num_heads = 8
  elif model_name == '136M':
    num_layers = 8
    embedding_dim = 1024
    num_heads = 8
  else:  # 270M
    num_layers = 16
    embedding_dim = 1024
    num_heads = 8

  # Keep the original action-value architecture (predicting Q-value distributions)
  # We'll derive action probabilities from Q-values via softmax
  config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=num_return_buckets,  # Keep original: predict Q-value distributions
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  # Build predictor to get param structure
  predictor = transformer.build_transformer_predictor(config)
  dummy_params = predictor.initial_params(
      rng=jrandom.PRNGKey(0),
      targets=np.zeros((1, 1), dtype=np.uint32),
  )

  # Load checkpoint
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{model_name}',
  )

  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=dummy_params,
      step=6_400_000,
      use_ema_params=False,
  )

  logging.info(f'Successfully loaded {model_name} model from {checkpoint_dir}')
  return params, config


def _make_dpo_loss_fn(predictor, z_atoms, beta=0.1, temperature=1.0):
  """Creates a DPO loss function.

  DPO directly optimizes the model to prefer better moves (from Stockfish)
  over worse moves (the model's own mistakes) without needing a separate
  reward model or value function.

  Args:
    predictor: The transformer predictor.
    z_atoms: Array of return bucket atoms (support values).
    beta: KL penalty coefficient (default 0.1).
    temperature: Temperature for action selection (default 1.0).

  Returns:
    Loss function that takes (online_params, reference_params, batch_data).
  """
  def loss_fn(online_params, reference_params, positions, chosen_moves, rejected_moves):
    """Computes DPO loss for preference pairs.

    Args:
      online_params: Current model parameters.
      reference_params: Reference model parameters (frozen).
      positions: [batch_size, seq_len] Tokenized positions.
      chosen_moves: [batch_size] Better move indices (Stockfish).
      rejected_moves: [batch_size] Worse move indices (model's moves).

    Returns:
      Scalar loss value.
    """
    return dpo_loss.dpo_loss(
        online_params=online_params,
        reference_params=reference_params,
        predict_fn=predictor,
        positions=positions,
        chosen_moves=chosen_moves,
        rejected_moves=rejected_moves,
        z_atoms=z_atoms,
        beta=beta,
        temperature=temperature,
    )

  return loss_fn


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting DPO self-play training')
  logging.info(f'Base model: {_BASE_MODEL.value}')
  logging.info(f'Iterations: {_NUM_ITERATIONS.value}')
  logging.info(f'Games per iteration: {_GAMES_PER_ITERATION.value}')
  logging.info(f'Stockfish depth: {_STOCKFISH_DEPTH.value}')
  logging.info(f'Eval threshold: {_EVAL_THRESHOLD.value} pawns')
  logging.info(f'DPO beta: {_BETA.value}')

  # Load base model
  params, config = _load_base_model(_BASE_MODEL.value)
  params_ema = copy.deepcopy(params)  # For checkpointing (fast EMA)
  reference_params = copy.deepcopy(params)  # For DPO reference (updated periodically)

  # Build predictor
  predictor = transformer.build_transformer_predictor(config)

  # Create optimizer
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(_LEARNING_RATE.value),
  )
  opt_state = optimizer.init(params)

  # Setup checkpoint directory
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{_BASE_MODEL.value}_selfplay',
  )
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Check for existing checkpoints to resume from (BEFORE sharding)
  start_iteration = 0
  if _RESUME.value and os.path.exists(checkpoint_dir):
    # Find latest checkpoint
    checkpoint_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()
    ]
    if checkpoint_dirs:
      latest_iteration = max(int(d) for d in checkpoint_dirs)
      logging.info(f'Found checkpoint at iteration {latest_iteration}')
      logging.info('Resuming training from checkpoint...')

      # Load params using training_utils (handles sharding correctly)
      params = training_utils.load_parameters(
          checkpoint_dir=checkpoint_dir,
          params=params,
          step=latest_iteration,
          use_ema_params=False,
      )

      # Load EMA params
      params_ema = training_utils.load_parameters(
          checkpoint_dir=checkpoint_dir,
          params=params_ema,
          step=latest_iteration,
          use_ema_params=True,
      )

      # Initialize reference params from EMA (for DPO)
      reference_params = copy.deepcopy(params_ema)

      # Load optimizer state (use raw checkpointer with restore_args)
      latest_checkpoint = os.path.join(checkpoint_dir, str(latest_iteration))
      checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

      # Create restore args to handle sharded checkpoints
      restore_args = ocp.checkpoint_utils.construct_restore_args(opt_state)
      opt_state = checkpointer.restore(
          os.path.join(latest_checkpoint, 'opt_state'),
          item=opt_state,
          restore_args=restore_args,
      )

      start_iteration = latest_iteration
      logging.info(f'Resumed from iteration {latest_iteration}')
      logging.info(f'Continuing training for {_NUM_ITERATIONS.value} more iterations')
    else:
      logging.info('No checkpoints found, starting from base model')
  elif _RESUME.value:
    logging.info('Resume flag set but no checkpoint directory found, starting from base model')

  # Setup sharding for distributed training (AFTER loading checkpoints)
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  sharding = jax.sharding.PositionalSharding(devices)
  sharding = sharding.reshape((jax.device_count(), 1))

  params = training_utils.replicate(params, sharding)
  params_ema = training_utils.replicate(params_ema, sharding)
  reference_params = training_utils.replicate(reference_params, sharding)
  opt_state = training_utils.replicate(opt_state, sharding)

  # Get return bucket values (support atoms) for Q-value computation
  num_return_buckets = 128
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
  # Convert to JAX array
  z_atoms = jnp.array(return_buckets_values, dtype=jnp.float32)
  
  #Perform sanity check
  assert jnp.all(jnp.diff(z_atoms) > 0)

  # Create DPO loss and gradient functions
  loss_fn = dpo_loss.make_dpo_loss_fn(
    predictor=predictor,
    z_atoms=z_atoms,
    beta=_BETA.value,
    temperature=_DPO_TEMPERATURE.value,
  )

  # Gradient wrt first argument (online_params)
  grad_fn = jax.value_and_grad(loss_fn, argnums=0)

  @jax.jit
  def update_step(params, params_ema, reference_params, opt_state, positions, chosen_moves, rejected_moves):
    """Single gradient update step with DPO."""
    # Compute loss and gradients using online params and reference params
    loss_val, grads = grad_fn(params, reference_params, positions, chosen_moves, rejected_moves)

    # Apply gradients to online params
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Update fast EMA params (for checkpointing, decay=0.99)
    fast_ema_decay = 0.99
    params_ema = jax.tree.map(
        lambda ema, new: fast_ema_decay * ema + (1 - fast_ema_decay) * new,
        params_ema,
        params,
    )

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)

    return params, params_ema, opt_state, loss_val, grad_norm

  # Setup checkpoint manager
  checkpoint_manager = training_utils.get_checkpoint_manager(
      ckpt_frequency=_SAVE_FREQUENCY.value,
      max_to_keep=5,
      save_frequency=_SAVE_FREQUENCY.value,
      checkpoint_dir=checkpoint_dir,
  )

  # Main DPO self-play training loop
  total_iterations = start_iteration + _NUM_ITERATIONS.value
  current_sf_depth = _STOCKFISH_DEPTH.value

  for iteration in range(start_iteration, total_iterations):
    logging.info(f'\n=== Iteration {iteration + 1}/{total_iterations} ===')

    # Create neural engine with current parameters
    # Un-shard params for inference
    local_params = jax.device_get(params_ema)

    # Use ActionValueEngine since we're keeping the Q-value architecture
    neural_engine = neural_engines.ActionValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=local_params,
            batch_size=1,
        ),
        temperature=1.0,
    )

    # Generate self-play games and create preference pairs using DPO
    generator = dpo_generator.DPOSelfPlayGenerator(
        neural_engine=neural_engine,
        stockfish_depth=current_sf_depth,
        stockfish_time_limit=_STOCKFISH_TIME.value,
        max_moves_per_game=200,
        eval_threshold=_EVAL_THRESHOLD.value,
        max_position_eval=3.0,
        temperature=1.0,
    )

    logging.info(f'Generating {_GAMES_PER_ITERATION.value} self-play games...')
    logging.info(f'Using Stockfish depth: {current_sf_depth}')

    # Collect preference pairs: (position, chosen_move, rejected_move)
    all_positions = []
    all_chosen_moves = []
    all_rejected_moves = []

    for positions, chosen_moves, rejected_moves in generator.generate_batch(
        num_games=_GAMES_PER_ITERATION.value,
        batch_size=_BATCH_SIZE.value,
    ):
      all_positions.append(positions)
      all_chosen_moves.append(chosen_moves)
      all_rejected_moves.append(rejected_moves)

    generator.close()

    if not all_positions:
      logging.warning('No preference pairs generated, skipping training.')
      logging.warning('Model may have converged or eval_threshold is too high.')
      continue

    # Train on collected preference pairs
    logging.info(f'Training for {_GRADIENT_STEPS_PER_ITERATION.value} steps...')

    total_batches = len(all_positions)
    batch_idx = 0

    for step in range(_GRADIENT_STEPS_PER_ITERATION.value):
      # Cycle through batches
      positions = all_positions[batch_idx % total_batches]
      chosen_moves = all_chosen_moves[batch_idx % total_batches]
      rejected_moves = all_rejected_moves[batch_idx % total_batches]
      batch_idx += 1

      # Shard data
      positions = jax.lax.with_sharding_constraint(positions, sharding)
      chosen_moves = jax.lax.with_sharding_constraint(chosen_moves, sharding)
      rejected_moves = jax.lax.with_sharding_constraint(rejected_moves, sharding)

      # Update parameters with DPO
      params, params_ema, opt_state, loss_val, grad_norm = update_step(
          params, params_ema, reference_params, opt_state, positions, chosen_moves, rejected_moves
      )

      if step % 10 == 0:
        logging.info(
            f'  Step {step}/{_GRADIENT_STEPS_PER_ITERATION.value}: '
            f'loss={float(loss_val):.4f}, grad_norm={float(grad_norm):.4f}'
        )

    # Update reference model periodically
    if (iteration + 1) % _UPDATE_REF_EVERY.value == 0:
      logging.info('Updating reference model...')
      reference_params = jax.tree.map(lambda x: x, params_ema)

    # Progressive curriculum: increase Stockfish depth after iteration 5
    if iteration >= 5:
      current_sf_depth = min(current_sf_depth + 2, 25)

    # Save checkpoint
    if (iteration + 1) % _SAVE_FREQUENCY.value == 0:
      logging.info(f'Saving checkpoint for iteration {iteration + 1}')
      checkpoint_manager.save(
          step=iteration + 1,
          items=dict(
              params=params,
              params_ema=params_ema,
              opt_state=opt_state,
          ),
      )


  # Wait for all checkpoints to finish saving
  logging.info('Waiting for checkpoint finalization...')
  checkpoint_manager.wait_until_finished()

  logging.info('Self-play training complete!')
  logging.info(f'Final model saved to: {checkpoint_dir}')


if __name__ == '__main__':
  app.run(main)