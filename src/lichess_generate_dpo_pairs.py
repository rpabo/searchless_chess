import json
import logging
import os

import debugpy
import numpy as np
from jax import random as jrandom

from searchless_chess.src import lichess_dpo_generator
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import neural_engines

def generate_dpo_pairs(
        base_model: str,
        max_pairs: int,
        lichess_db_path: str,
        checkpoint_dir: str,
        predictor,
        params_ema,        
    )-> tuple[list[np.ndarray], list[int], list[int]]:
  # Check for cached DPO pairs (JSONL format with metadata)
  cache_file = os.path.join(checkpoint_dir, 'dpo_pairs_cache.jsonl')
  metadata_file = os.path.join(checkpoint_dir, 'cache_metadata.json')

  # Load existing cache if present
  all_positions = []
  all_chosen = []
  all_rejected = []
  all_cp_diff = []
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
          all_cp_diff.append(pair['cp_diff'])
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
          all_cp_diff.append(pair['cp_diff'])
          cached_position_hashes.add(tuple(pair['position']))

      logging.info(f'Loaded {len(all_positions):,} existing pairs')
      logging.info(f'Will skip these positions when generating new pairs')

  # Check if we already have enough pairs
  if max_pairs > 0 and len(all_positions) >= max_pairs:
    logging.info(f'\n=== Already have {len(all_positions):,} pairs (max: {max_pairs:,}) ===')
    cache_complete = True

  # Generate pairs if cache is not complete
  if not cache_complete:
    if not os.path.exists(metadata_file):
      logging.info(f'\n=== Generating DPO Pairs ===')
      logging.info('Cache not found - generating pairs from entire database...')
      logging.info(f'This will be saved to: {cache_file}\n')

    # Check if we need to generate more pairs
    if max_pairs > 0:
      pairs_needed = max_pairs - len(all_positions)
      logging.info(f'Need {pairs_needed:,} more pairs to reach {max_pairs:,}')

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
        database_path=lichess_db_path,
    )

    # Open cache file in append mode
    cache_handle = open(cache_file, 'a')

    pair_count = len(all_positions)  # Start from existing count
    new_pairs_added = 0
    skipped_duplicates = 0
    save_interval = 100  # Save metadata every 1000 pairs
    last_save_count = pair_count

    try:
      for positions_batch, chosen_batch, rejected_batch, cp_diff, stats in generator.generate_streaming_batches(
          positions_per_batch=10000,
          batch_size=32,
          target_pairs=3500,
      ):
        for pos, chosen, rejected, cp_diff in zip(positions_batch, chosen_batch, rejected_batch, cp_diff):
          pos_hash = tuple(pos.tolist())

          # Skip if already cached
          if pos_hash in cached_position_hashes:
            skipped_duplicates += 1
            continue

          # Add new pair
          all_positions.append(pos)
          all_chosen.append(chosen)
          all_rejected.append(rejected)
          all_cp_diff.append(cp_diff)
          cached_position_hashes.add(pos_hash)

          # Write to cache file immediately (JSONL format)
          pair_data = {
              'position': pos.tolist(),
              'chosen': chosen,
              'rejected': rejected,
              'cp_diff': cp_diff
          }
          cache_handle.write(json.dumps(pair_data) + '\n')
          pair_count += 1
          new_pairs_added += 1

          # Check if we've reached the max pairs limit
          if max_pairs > 0 and pair_count >= max_pairs:
            logging.info(f'Reached max pairs limit: {max_pairs:,}')
            break

        # Flush cache file and update metadata periodically
        if new_pairs_added > 0 and pair_count - last_save_count >= save_interval:
          cache_handle.flush()
          os.fsync(cache_handle.fileno())

          # Update metadata
          metadata = {
              'model': base_model,
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
        if max_pairs > 0 and pair_count >= max_pairs:
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
        'model': base_model,
        'num_pairs': len(all_positions),
        'complete': True,
    }
    with open(metadata_file, 'w') as f:
      json.dump(metadata, f)

    logging.info(f'Cache saved to: {cache_file}')
    
    return all_positions, all_chosen, all_rejected

if __name__ == '__main__':
    debugpy.listen(5678)
    debugpy.wait_for_client()
    
    num_layers, embedding_dim, num_heads = 8, 256, 8 #Assume configuration for 9M
    base_model = '9M'
    max_pairs = 4000 #Set to a small value for debugging purposes.
    lichess_db_path = '../data/lichess_db_eval.jsonl.zst'    
    
    #Setup predictor
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
    
    base_checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{base_model}')
    
    #Load base model
    params = training_utils.load_parameters(
        checkpoint_dir=base_checkpoint_dir,
        params=initial_params,
        step=-1,
    )
    
    checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{base_model}_lichess') 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    params_ema = params    
            
    generate_dpo_pairs(
        base_model,
        max_pairs,
        lichess_db_path,
        checkpoint_dir,
        predictor,
        params_ema
    )