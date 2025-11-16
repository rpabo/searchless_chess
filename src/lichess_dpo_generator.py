"""DPO preference pair generation from Lichess evaluation database.

Uses pre-computed Stockfish evaluations from Lichess (depth 36+) to generate
high-quality preference pairs without needing to run Stockfish ourselves.
"""

import io
import json
import os
from collections.abc import Iterator
from typing import Optional

import chess
import numpy as np
import zstandard as zstd

from searchless_chess.src import tokenizer
from searchless_chess.src import utils

class PreferencePair:
  """A preference pair for DPO training."""
  def __init__(
      self, 
      position: str, 
      chosen_move: str, 
      rejected_move: str, 
      cp_diff: float      
    ):
    
    self.position = position
    self.chosen_move = chosen_move
    self.rejected_move = rejected_move
    self.cp_diff = cp_diff    
    
class LichessDPOGenerator:
  """Generates DPO preference pairs from Lichess evaluation database."""

  def __init__(self, 
               predict_fn, 
               database_path: str, 
               cp_margin: int = 200,
               ):
    """Initialize generator.

    Args:
      predict_fn: Function that takes a chess.Board and returns predicted action values.
      database_path: Path to lichess_db_eval.jsonl.zst file.
    """
    self.predict_fn = predict_fn
    self.database_path = database_path
    self.cp_margin = cp_margin
    
  def _cp_margin_for_side(self, best_cp: float, model_cp: float, board: chess.Board) -> float:
    """Return cp margin where positive means engine-best is better for side to move."""
    if board.turn == chess.WHITE:
      # Higher cp is better for White
      return best_cp - model_cp
    else:
      # Lower cp (more negative) is better for Black, but cp is always from White's POV
      # Example: best_cp = -30, model_cp = +20 → margin = 20 - (-30) = 50 (engine better)
      return model_cp - best_cp    

  def stream_positions(self, max_positions: Optional[int] = None) -> Iterator[dict]:
    """Stream positions from compressed Lichess database.

    Args:
      max_positions: Maximum number of positions to read (None = unlimited).

    Yields:
      Dictionary with 'fen' and 'evals' keys.
    """
    if not os.path.exists(self.database_path):
      raise FileNotFoundError(
          f"Lichess database not found at {self.database_path}. "
          "Download from https://database.lichess.org/lichess_db_eval.jsonl.zst"
      )

    dctx = zstd.ZstdDecompressor()

    with open(self.database_path, 'rb') as compressed:
      with dctx.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        for i, line in enumerate(text_stream):
          if max_positions and i >= max_positions:
            break
          yield json.loads(line)

  def fen4_to_fen6(self, fen4: str) -> str:
    """Convert FEN-4 (pieces, color, castling, ep) to FEN-6.

    Args:
      fen4: FEN string with 4 fields (pieces, color, castling, ep).

    Returns:
      FEN string with 6 fields (halfmove and fullmove sampled from training distribution).

    Note:
      Samples halfmove and fullmove to match the base model's training distribution:
      - Halfmove: exponential-like distribution (42.5% are 0, mean 1.71)
      - Fullmove: typical range 13-28, mean 21.77, median 19
    """
    # Sample halfmove clock from exponential-like distribution matching training data
    # 42.5% are 0, 21.3% are 1, 11.9% are 2, etc.
    halfmove_weights = [42.5, 21.3, 11.9, 8.7, 5.2, 3.5, 2.1, 1.6, 1.1, 0.7, 0.5, 0.3, 0.2]
    halfmove_weights = np.array(halfmove_weights) / sum(halfmove_weights)
    halfmove = np.random.choice(len(halfmove_weights), p=halfmove_weights)

    # Sample fullmove number from approximate normal distribution
    # Mean 21.77, std ~10, clipped to reasonable range
    fullmove = max(3, min(100, int(np.random.normal(22, 10))))

    return f"{fen4} {halfmove} {fullmove}"

  def get_best_move_and_cp(self, evals: list) -> tuple[str, float]:
    """Extract best move and evaluation from Lichess evals.

    Args:
      evals: List of evaluation dictionaries from Lichess DB.

    Returns:
      Tuple of (best_move_uci, centipawn_evaluation).
    """
    # Use the evaluation with the most PVs (deepest analysis)
    best_eval = evals[-1]

    # First PV is the best line
    best_pv = best_eval['pvs'][0]

    # Extract first move from line (UCI format)
    line = best_pv['line']
    best_move = line.split()[0]

    # Get evaluation (cp or mate)
    if 'mate' in best_pv:
      # Mate in N moves -> convert to large cp value
      mate_in = best_pv['mate']
      cp = 20000 if mate_in > 0 else -20000
    else:
      cp = best_pv['cp']

    return best_move, cp

  def find_move_cp(self, evals: list, move_uci: str) -> Optional[float]:
    """Find centipawn evaluation for a specific move.

    Args:
      evals: List of evaluation dictionaries from Lichess DB.
      move_uci: UCI string of the move to find.

    Returns:
      Centipawn evaluation if move found in PVs, None otherwise.
    """
    # Use the evaluation with the most PVs
    best_eval = evals[-1]

    for pv in best_eval['pvs']:
      line = pv['line']
      first_move = line.split()[0]

      if first_move == move_uci:
        if 'mate' in pv:
          mate_in = pv['mate']
          return 20000 if mate_in > 0 else -20000
        else:
          return pv['cp']

    return None

  def predict_model_move(self, board: chess.Board) -> chess.Move:
    """Get model's predicted move for a position.

    This follows the exact logic from ActionValueEngine.analyse() in neural_engines.py.

    Args:
      board: Chess board position.

    Returns:
      Predicted move, vector containing win probabilities
    """
    # Get legal moves and tokenize actions (same as ActionValueEngine.analyse)
    sorted_legal_moves = list(board.legal_moves)
    legal_actions = np.array([utils.MOVE_TO_ACTION[m.uci()] for m in sorted_legal_moves], dtype=np.int32)
    legal_actions = np.expand_dims(legal_actions, axis=-1)

    # Tokenize the return buckets (dummy values)
    dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)

    # Tokenize the board
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    sequences = np.stack([tokenized_fen] * len(legal_actions))

    # Create sequences: [fen, action, return_bucket] for each legal action
    sequences = np.concatenate([sequences, legal_actions, dummy_return_buckets], axis=1)

    # Get log probabilities for return buckets for each action
    try:
      return_buckets_log_probs = self.predict_fn(sequences)[:, -1]  # [num_legal_actions, num_buckets]
    except Exception as e:
      print(f"ERROR in predict_fn: {e}")
      print(f"  FEN: {board.fen()}")
      print(f"  Sequences shape: {sequences.shape}")
      print(f"  Num legal moves: {len(sorted_legal_moves)}")
      raise

    # Convert log probs to actual probs and compute expected win probability
    return_buckets_probs = np.exp(return_buckets_log_probs)
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
    win_probs = np.inner(return_buckets_probs, return_buckets_values)

    # Return the move with highest win probability
    best_index = np.argmax(win_probs)
    return sorted_legal_moves[best_index], win_probs

  def get_best_line(self, evals: list) -> list[str]:
    """Extract best line (PV) from Lichess evals.

    Args:
      evals: List of evaluation dictionaries from Lichess DB.

    Returns:
      List of UCI move strings in the principal variation.
    """
    # Use the evaluation with the most PVs (deepest analysis)
    best_eval = evals[-1]

    # First PV is the best line
    best_pv = best_eval['pvs'][0]

    # Extract moves from line
    line = best_pv['line']
    return line.split()

  def get_all_pv_first_moves(self, evals: list) -> set[str]:
    """Extract first moves from all PVs in the deepest evaluation.

    Args:
      evals: List of evaluation dictionaries from Lichess DB.

    Returns:
      Set of UCI strings for the first move of each PV.
    """
    # Use the evaluation with the most PVs (deepest analysis)
    best_eval = evals[-1]

    first_moves = set()
    for pv in best_eval['pvs']:
      line = pv['line']
      first_move = line.split()[0]
      first_moves.add(first_move)

    return first_moves

  def generate_streaming_batches(
      self,
      positions_per_batch: int,
      batch_size: int,
      target_pairs: int,
  ) -> Iterator[tuple[list, list, list, dict]]:
    """Generate preference pairs in streaming batches (memory-efficient).

    Args:
      positions_per_batch: Number of Lichess positions to process per batch.
      batch_size: Mini-batch size for training.
      target_pairs: Target number of total pairs to generate.

    Yields:
      Tuples of (positions_batch, chosen_batch, rejected_batch, stats_dict).
    """
    total_positions_processed = 0
    total_pairs_generated = 0
    position_stream = self.stream_positions(max_positions=None)

    print(f"\nStreaming Lichess positions until {target_pairs} pairs generated...")

    while total_pairs_generated < target_pairs:
      # Process one batch of positions
      batch_preferences = []
      batch_positions_processed = 0
      batch_skipped_gameover = 0
      batch_skipped_no_eval = 0
      batch_pairs_from_lines = 0
      batch_skipped_cp_margin = 0   # NEW: count positions failing cp threshold

      for _ in range(positions_per_batch):
        try:
          position_data = next(position_stream)
        except StopIteration:
          print(f"\nReached end of Lichess database at {total_pairs_generated} pairs")
          if batch_preferences:
            # Yield remaining pairs
            for batch_data in self._format_batch(batch_preferences, batch_size, {}):
              yield batch_data
          return

        batch_positions_processed += 1
        total_positions_processed += 1

        # Progress logging every 1000 positions
        if batch_positions_processed % 1000 == 0:
          print(f"    Processing position {batch_positions_processed}/{positions_per_batch} "
                f"({len(batch_preferences)} pairs so far)...")

        # Get best line and all acceptable first moves from evaluations
        try:
          # best_line = self.get_best_line(position_data['evals'])
          acceptable_first_moves = self.get_all_pv_first_moves(position_data['evals'])
          # NEW: also get cp for engine-best
          best_move_uci, best_cp = self.get_best_move_and_cp(position_data['evals'])
          
        except (KeyError, IndexError):
          batch_skipped_no_eval += 1
          continue

        # Convert FEN and create board
        fen4 = position_data['fen']
        fen6 = self.fen4_to_fen6(fen4)
        board = chess.Board(fen6)

        # Skip if no legal moves
        if board.is_game_over():
          batch_skipped_gameover += 1
          continue

        # Check ONLY the initial position (where we have PV data)
        # Get model's prediction for the initial position
        try:
          model_move, win_probs = self.predict_model_move(board)
          model_move_uci = model_move.uci()
          
        except Exception as e:
          if batch_positions_processed <= 10:  # Debug first few positions
            print(f"    Error predicting move: {e}")
          continue
                
        best_index = np.argmax(win_probs)
        model_win_prob = win_probs[best_index]

        # Only create preference pair if model's move is NOT in any PV
        # best_first_move = best_line[0]  # First move of best PV
        # diff_cp = self.find_move_cp(position_data['evals'], best_first_move - \
        #   self.find_move_cp(position_data['evals'], model_move.uci()))
                
        if model_move_uci not in acceptable_first_moves:
          # NEW: try to get cp for model's move from evals
          model_cp = self.find_move_cp(position_data['evals'], model_move_uci)
          
          if model_cp is not None and best_cp is not None:
            margin = self._cp_margin_for_side(best_cp, model_cp, board)
          else:
            #Handle what to do if the model move is not in the PV. 
            white_prob = 1.0 / (1.0 + 10 ** (-best_cp / 400.0))
            
            if board.turn == chess.WHITE:
              engine_prob_side = white_prob
            else:
              engine_prob_side = 1.0 - white_prob
            
            # 2) Probability margin in side-to-move space
            prob_margin = engine_prob_side - model_win_prob        
            
            # 3) Scale to cp-like units WITHOUT forcing >= cp_margin
            margin = prob_margin * 800.0    # can be small or negative            
                        
          # Enforce cp-based quality threshold
          if margin < self.cp_margin:
            batch_skipped_cp_margin += 1
            continue            
            
          batch_preferences.append(PreferencePair(
              position=board.fen(),
              chosen_move=best_move_uci,  # Best move (first PV)
              rejected_move=model_move_uci,
              cp_diff=margin,              
          ))
          batch_pairs_from_lines += 1

        # Debug: Print first few comparisons
        if batch_positions_processed <= 5:
          print(f"    Pos {batch_positions_processed}: "
                f"Best={best_move_uci}, Model={model_move.uci()}"
                f"AllPVs={acceptable_first_moves}, InPVs={model_move.uci() in acceptable_first_moves}")

      # Stats for this batch
      total_pairs_generated += len(batch_preferences)
      stats = {
          'batch_positions': batch_positions_processed,
          'batch_pairs': len(batch_preferences),
          'batch_skipped_gameover': batch_skipped_gameover,
          'batch_skipped_no_eval': batch_skipped_no_eval,
          'batch_skipped_cp_margin': batch_skipped_cp_margin,  # NEW
          'total_positions': total_positions_processed,
          'total_pairs': total_pairs_generated,
      }

      print(f"  Batch: {batch_positions_processed} positions → {len(batch_preferences)} pairs | "
            f"Total: {total_positions_processed:,} positions, {total_pairs_generated:,} pairs")
      print(f"  Formatting {len(batch_preferences)} pairs into mini-batches...")

      if batch_preferences:
        # Convert to training format and yield mini-batches
        for batch_data in self._format_batch(batch_preferences, batch_size, stats):
          yield batch_data
      else:
        print(f"  Warning: No pairs generated from this batch")

      # Check if we've reached target
      if total_pairs_generated >= target_pairs:
        print(f"\nReached target of {target_pairs} pairs!")
        return
      else:
        print(f"  Need {target_pairs - total_pairs_generated} more pairs, continuing to next batch...")

  def _format_batch(
      self,
      preferences: list[PreferencePair],
      batch_size: int,      
      stats: Optional[dict] = None
  ) -> tuple[list, list, list, dict]:
    """Convert preferences to training format."""
    positions_list = []
    chosen_moves_list = []
    rejected_moves_list = []
    cp_diff_list = [] #Added
        
    for pref in preferences:
      board = chess.Board(pref.position)
      tokenized = tokenizer.tokenize(board.fen()).astype(np.uint32)

      positions_list.append(tokenized)
      chosen_moves_list.append(utils.MOVE_TO_ACTION[pref.chosen_move])
      rejected_moves_list.append(utils.MOVE_TO_ACTION[pref.rejected_move])
      cp_diff_list.append(pref.cp_diff) #Added
                
    # Shuffle for randomness
    indices = np.arange(len(preferences))
    np.random.shuffle(indices)

    # Create mini-batches
    for i in range(0, len(preferences), batch_size):
      batch_indices = indices[i:i+batch_size]

      positions_batch = [positions_list[idx] for idx in batch_indices]
      chosen_batch = [chosen_moves_list[idx] for idx in batch_indices]
      rejected_batch = [rejected_moves_list[idx] for idx in batch_indices]
      cp_diff_batch = [cp_diff_list[idx] for idx in batch_indices] #Added
                
      yield positions_batch, chosen_batch, rejected_batch, cp_diff_batch, stats or {}
