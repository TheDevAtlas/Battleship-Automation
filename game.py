from board import Board
from typing import Dict, Any

class Game:
    def __init__(self, player_strategy):
        self.player = player_strategy
        self.board = Board()
        self.move_count = 0
        self.game_history = []
    
    def setup_game(self):
        """Initialize a new game"""
        self.board = Board()
        self.board.place_ships_randomly()
        self.move_count = 0
        self.game_history = []
        self.player.reset()
    
    def play_game(self, show_moves: bool = False) -> Dict[str, Any]:
        """Play a complete game and return results"""
        self.setup_game()
        
        while not self.board.is_game_over():
            # Get move from player strategy
            row, col = self.player.make_move(self.board)
            
            # Make the move
            result = self.board.make_guess(row, col)
            self.move_count += 1
            
            # Record the move
            move_info = {
                'move': self.move_count,
                'position': (row, col),
                'result': result
            }
            self.game_history.append(move_info)
            
            if show_moves:
                if result == "sunk":
                    ship_id = self.board.grid[row][col]
                    ship_size = next(ship['size'] for ship in self.board.ships if ship['id'] == ship_id)
                    print(f"Move {self.move_count}: ({row}, {col}) - {result} (ship size {ship_size})")
                else:
                    print(f"Move {self.move_count}: ({row}, {col}) - {result}")
        
        return {
            'moves': self.move_count,
            'history': self.game_history,
            'player': self.player.name,
            'board_state': self.board.get_board_state()
        }
    
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics"""
        return {
            'moves': self.move_count,
            'hits': self.board.hit_count,
            'misses': self.move_count - self.board.hit_count,
            'game_over': self.board.is_game_over()
        }