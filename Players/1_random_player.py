import random
from typing import Tuple, List
from Source.board import Board

class RandomPlayer:
    """A player that makes completely random moves"""
    
    def __init__(self, name: str = "Random Player"):
        self.name = name
    
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a random move from available positions"""
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        return random.choice(valid_moves)
    
    def reset(self):
        """Reset any internal state (random player doesn't need this)"""
        pass