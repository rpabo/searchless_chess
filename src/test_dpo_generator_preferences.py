# tests/test_dpo_generator_preferences.py
import pytest
import numpy as np
import chess

from dpo_generator import DPOSelfPlayGenerator, GameTrajectory, PreferencePair
from searchless_chess.src import utils, tokenizer

# ----- Mocks -----

class MockRelScore:
    """Mimics chess.engine.PovScore behavior just enough for the generator."""
    def __init__(self, cp):
        # cp is 'centipawns from side-to-move POV at the time of analyse()'
        self._cp = cp

    def is_mate(self):
        return False

    def mate(self):
        return None

    def score(self):
        # generator divides by 100.0 to get pawns
        return int(self._cp * 100)

    @property
    def relative(self):
        # in generator, they read ['score'].relative then call .score()/is_mate()
        return self


class MockStockfishEngine:
    """
    A controllable Stockfish mock:
      - analyse(board) returns:
          On base position (no moves pushed yet): {'pv': [best_move]}
          After SF move: {'score': MockRelScore(cp_after_sf)}
          After LLM move: {'score': MockRelScore(cp_after_llm)}
    We also record the FENs analysed to ensure post-move evals happen.
    """
    def __init__(self, best_move_uci, cp_after_sf, cp_after_llm, record_fens=True):
        self.best_move = chess.Move.from_uci(best_move_uci)
        self.cp_after_sf = cp_after_sf
        self.cp_after_llm = cp_after_llm
        self.record_fens = record_fens
        self.fens = []

    def analyse(self, board):
        if self.record_fens:
            self.fens.append(board.fen())

        if len(board.move_stack) == 0:
            # "top-level" call: return pv with best move only
            return {'pv': [self.best_move]}
        # After at least one move: decide which move was pushed last
        last = board.move_stack[-1]
        if last == self.best_move:
            return {'score': MockRelScore(self.cp_after_sf)}
        else:
            return {'score': MockRelScore(self.cp_after_llm)}


class MockNeuralEngine:
    """Always plays a fixed move (LLM move)."""
    def __init__(self, move_uci):
        self._move = chess.Move.from_uci(move_uci)

    def play(self, board: chess.Board):
        return self._move


# ----- Helpers -----

def make_traj(fen: str, llm_move: str) -> GameTrajectory:
    return GameTrajectory(
        positions=[fen],
        moves=[llm_move],
        move_numbers=[1],
    )

def token_of(move_uci: str) -> int:
    return utils.MOVE_TO_ACTION[move_uci]


# ----- Tests -----

def test_preference_labeling_white_to_move(monkeypatch):
    """
    White to move. SF move leads to +0.6 (better for side-to-move),
    LLM move leads to +0.1. Expect chosen=SF, rejected=LLM and eval_margin≈0.5.
    Also verifies post-move eval calls.
    """
    # Position with white to move
    fen = chess.STARTING_FEN  # 'w' to move in standard start
    llm_move = 'e2e3'
    sf_move = 'e2e4'

    # Mock engine: after SF move -> +0.6, after LLM move -> +0.1
    mock_sf = MockStockfishEngine(best_move_uci=sf_move, cp_after_sf=0.6, cp_after_llm=0.1)

    # Monkeypatch constructor used in DPOSelfPlayGenerator.__init__
    from dpo_generator import stockfish_engine
    monkeypatch.setattr(stockfish_engine, 'StockfishEngine', lambda limit: mock_sf)

    gen = DPOSelfPlayGenerator(neural_engine=MockNeuralEngine(llm_move), eval_threshold=0.2)
    prefs = gen.create_preferences([make_traj(fen, llm_move)])

    # One preference expected
    assert len(prefs) == 1
    p: PreferencePair = prefs[0]
    
    assert p.chosen_move == sf_move
    assert p.rejected_move == llm_move
    assert p.chosen_action == token_of(sf_move)
    assert p.rejected_action == token_of(llm_move)
    
    # Margin should be (0.6 - 0.1) = 0.5
    assert abs(p.eval_margin - 0.5) < 1e-6

    # Verify we evaluated AFTER pushing each move (two extra FENs beyond base)
    # Base FEN first call + pushed-SF FEN + pushed-LLM FEN
    assert len(mock_sf.fens) >= 3  # base + after SF + after LLM


def test_preference_labeling_black_to_move_perspective(monkeypatch):
    """
    Black to move. From side-to-move POV (Black), better means more negative from White POV.
    We simulate:
      - After SF move: -0.7
      - After LLM move: -0.2
    For the *initial mover* (Black), SF is better by 0.5 pawns.
    The generator must map both post-move evals back to the initial side-to-move perspective
    and still label chosen=SF.
    """
    # Make a legal midgame FEN with Black to move (e.g., after 1.e4)
    board = chess.Board()
    board.push_san("e4")
    fen_black = board.fen()  # now 'b' to move

    llm_move = 'c7c5'   # model move
    sf_move  = 'e7e5'   # SF best

    # Mock: after SF move -> -0.7 (good for Black), after LLM move -> -0.2 (worse)
    mock_sf = MockStockfishEngine(best_move_uci=sf_move, cp_after_sf=-0.7, cp_after_llm=-0.2)

    from dpo_generator import stockfish_engine
    monkeypatch.setattr(stockfish_engine, 'StockfishEngine', lambda limit: mock_sf)

    gen = DPOSelfPlayGenerator(neural_engine=MockNeuralEngine(llm_move), eval_threshold=0.2)
    prefs = gen.create_preferences([make_traj(fen_black, llm_move)])

    assert len(prefs) == 1
    p: PreferencePair = prefs[0]
    assert p.chosen_move == sf_move
    assert p.rejected_move == llm_move
    assert p.chosen_action == token_of(sf_move)
    assert p.rejected_action == token_of(llm_move)

    # From initial side-to-move (Black) perspective, margin should be 0.5
    assert abs(p.eval_margin - 0.5) < 1e-6


def test_no_preference_when_below_threshold(monkeypatch):
    """
    If eval difference < threshold, no preference should be emitted.
    """
    fen = chess.STARTING_FEN
    llm_move = 'e2e3'
    sf_move  = 'e2e4'

    # Only 0.05 difference (below 0.2 threshold)
    mock_sf = MockStockfishEngine(best_move_uci=sf_move, cp_after_sf=0.55, cp_after_llm=0.50)

    from dpo_generator import stockfish_engine
    monkeypatch.setattr(stockfish_engine, 'StockfishEngine', lambda limit: mock_sf)

    gen = DPOSelfPlayGenerator(neural_engine=MockNeuralEngine(llm_move), eval_threshold=0.2)
    prefs = gen.create_preferences([make_traj(fen, llm_move)])

    assert prefs == []


def test_illegal_llm_move_is_skipped(monkeypatch):
    """
    If the LLM move is illegal in the given position, the position must be skipped.
    """
    fen = chess.STARTING_FEN
    llm_move = 'h1h3'   # illegal from the initial position (rook blocked)
    sf_move  = 'e2e4'

    mock_sf = MockStockfishEngine(best_move_uci=sf_move, cp_after_sf=0.2, cp_after_llm=0.0)

    from dpo_generator import stockfish_engine
    monkeypatch.setattr(stockfish_engine, 'StockfishEngine', lambda limit: mock_sf)

    gen = DPOSelfPlayGenerator(neural_engine=MockNeuralEngine(llm_move), eval_threshold=0.05)
    prefs = gen.create_preferences([make_traj(fen, llm_move)])

    assert prefs == []