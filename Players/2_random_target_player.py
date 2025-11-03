import random
from typing import Tuple, List, Set, Optional
from Source.board import Board

class RandomTargetPlayer:
    """A player that uses random moves for hunting, but switches to target mode when a hit is found"""
    
    def __init__(self, name: str = "Random Target Player"):
        self.name = name
        self.mode = "hunt"  # "hunt" or "target"
        self.target_stack = []  # Stack of coordinates to investigate after a hit
        self.current_ship_hits = []  # Current ship being targeted
        self.sunk_ship_coords = set()  # All coordinates of sunk ships
        self.ship_sizes = [5, 4, 3, 3, 2]  # Known ship sizes
        
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a move based on current strategy (hunt or target)"""
        if self.mode == "target" and self.target_stack:
            return self._target_move(board)
        else:
            return self._hunt_move(board)
    
    def _hunt_move(self, board: Board) -> Tuple[int, int]:
        """Make a random move during hunt phase"""
        self.mode = "hunt"
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Filter out coordinates around sunk ships to avoid wasting moves
        filtered_moves = []
        for move in valid_moves:
            if not self._is_adjacent_to_sunk_ship(move):
                filtered_moves.append(move)
        
        # If all moves are adjacent to sunk ships, use any valid move
        if not filtered_moves:
            filtered_moves = valid_moves
            
        row, col = random.choice(filtered_moves)
        
        # Check result of this move to switch to target mode if needed
        result = self._simulate_move_result(board, row, col)
        if result == "hit":
            self.mode = "target"
            self.current_ship_hits = [(row, col)]
            self._add_adjacent_targets(board, row, col)
        elif result == "sunk":
            # Ship was sunk with this hit, get the ship coordinates
            ship_id = board.grid[row][col]
            sunk_coords = board.get_sunk_ship_coords(ship_id)
            if sunk_coords:
                self.sunk_ship_coords.update(sunk_coords)
            
            # Check if there are any remaining unsunk hits on the board
            has_unsunk_hits = False
            for r in range(board.size):
                for c in range(board.size):
                    if board.hits[r][c] and (r, c) not in self.sunk_ship_coords:
                        has_unsunk_hits = True
                        break
                if has_unsunk_hits:
                    break
            
            # Only switch to hunt mode if there are no remaining unsunk hits
            if not has_unsunk_hits:
                self.mode = "hunt"
            self.current_ship_hits = []
            self.target_stack = []
        
        return (row, col)
    
    def _target_move(self, board: Board) -> Tuple[int, int]:
        """Make a targeted move when pursuing a hit ship"""
        if not self.target_stack:
            return self._hunt_move(board)
        
        row, col = self.target_stack.pop()
        
        # Check if this coordinate is still valid
        if board.guesses[row][col]:
            return self._target_move(board)  # Try next target
        
        result = self._simulate_move_result(board, row, col)
        
        if result == "hit":
            self.current_ship_hits.append((row, col))
            # Add new adjacent targets
            self._add_adjacent_targets(board, row, col)
            # If we have 2+ hits, focus on the line they form
            if len(self.current_ship_hits) >= 2:
                self._focus_on_ship_line(board)
        elif result == "sunk":
            # Ship was sunk
            ship_id = board.grid[row][col]
            sunk_coords = board.get_sunk_ship_coords(ship_id)
            if sunk_coords:
                self.sunk_ship_coords.update(sunk_coords)
            
            # Check if there are any remaining unsunk hits on the board
            has_unsunk_hits = False
            for r in range(board.size):
                for c in range(board.size):
                    if board.hits[r][c] and (r, c) not in self.sunk_ship_coords:
                        has_unsunk_hits = True
                        break
                if has_unsunk_hits:
                    break
            
            # Only switch to hunt mode if there are no remaining unsunk hits
            if not has_unsunk_hits:
                self.mode = "hunt"
            self.current_ship_hits = []
            self.target_stack = []
        
        return (row, col)
    
    def _simulate_move_result(self, board: Board, row: int, col: int) -> str:
        """Simulate what the result of a move would be without actually making it"""
        if board.guesses[row][col]:
            return 'already_guessed'
        
        if board.grid[row][col] != 0:  # Hit
            ship_id = board.grid[row][col]
            for ship in board.ships:
                if ship['id'] == ship_id:
                    # Check if this would sink the ship
                    if ship['hits'] + 1 == ship['size']:
                        return 'sunk'
                    else:
                        return 'hit'
        
        return 'miss'
    
    def _add_adjacent_targets(self, board: Board, row: int, col: int):
        """Add adjacent coordinates to target stack"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < board.size and 0 <= new_col < board.size and
                not board.guesses[new_row][new_col] and
                (new_row, new_col) not in self.sunk_ship_coords):
                self.target_stack.append((new_row, new_col))
    
    def _focus_on_ship_line(self, board: Board):
        """When we have multiple hits, focus on the line they form"""
        if len(self.current_ship_hits) < 2:
            return
        
        # Clear current target stack
        self.target_stack = []
        
        # Determine if ship is horizontal or vertical
        first_hit = self.current_ship_hits[0]
        second_hit = self.current_ship_hits[1]
        
        if first_hit[0] == second_hit[0]:  # Same row - horizontal ship
            row = first_hit[0]
            min_col = min(hit[1] for hit in self.current_ship_hits)
            max_col = max(hit[1] for hit in self.current_ship_hits)
            
            # Add targets at both ends of the line
            for col in [min_col - 1, max_col + 1]:
                if (0 <= col < board.size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    self.target_stack.append((row, col))
                    
        elif first_hit[1] == second_hit[1]:  # Same col - vertical ship
            col = first_hit[1]
            min_row = min(hit[0] for hit in self.current_ship_hits)
            max_row = max(hit[0] for hit in self.current_ship_hits)
            
            # Add targets at both ends of the line
            for row in [min_row - 1, max_row + 1]:
                if (0 <= row < board.size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    self.target_stack.append((row, col))
    
    def _is_adjacent_to_sunk_ship(self, coord: Tuple[int, int]) -> bool:
        """Check if a coordinate is adjacent to any sunk ship"""
        row, col = coord
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            adj_row, adj_col = row + dr, col + dc
            if (adj_row, adj_col) in self.sunk_ship_coords:
                return True
        return False
    
    def reset(self):
        """Reset all internal state for a new game"""
        self.mode = "hunt"
        self.target_stack = []
        self.current_ship_hits = []
        self.sunk_ship_coords = set()