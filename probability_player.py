import random
from typing import Tuple, List, Set, Optional, Dict
from board import Board
import numpy as np

class ProbabilityPlayer:
    """A player that uses probability mapping combined with hunt/target strategies.
    
    This player maintains a probability map of where ships are likely to be based on:
    1. Remaining ship sizes that haven't been sunk
    2. Valid positions where ships can still fit
    3. Known hits that haven't been completed into sunk ships
    4. Intelligent spacing to reduce maximum number of moves
    
    The probability map is updated after each move and guides both hunt and target phases.
    """
    
    def __init__(self, name: str = "Probability Player"):
        self.name = name
        self.mode = "hunt"  # "hunt" or "target"
        self.target_stack = []  # Stack of coordinates to investigate after a hit
        self.current_ship_hits = []  # Current ship being targeted
        self.sunk_ship_coords = set()  # All coordinates of sunk ships
        self.ship_sizes = [5, 4, 3, 3, 2]  # Known ship sizes
        self.sunk_ships = []  # Track which ships have been sunk (indices)
        self.board_size = 10
        self.probability_map = None
        
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a move based on probability mapping and current strategy"""
        # Update probability map based on current board state
        self._update_probability_map(board)
        
        if self.mode == "target" and self.target_stack:
            return self._target_move(board)
        else:
            return self._hunt_move(board)
    
    def _update_probability_map(self, board: Board):
        """Generate a probability map based on where ships can fit"""
        self.probability_map = np.zeros((self.board_size, self.board_size), dtype=float)
        
        # Get remaining ship sizes
        remaining_ships = []
        for i, size in enumerate(self.ship_sizes):
            if i not in self.sunk_ships:
                remaining_ships.append(size)
        
        if not remaining_ships:
            return
        
        # For each remaining ship, calculate where it could fit
        for ship_size in remaining_ships:
            # Try horizontal placements
            for row in range(self.board_size):
                for col in range(self.board_size - ship_size + 1):
                    if self._can_fit_ship_horizontal(board, row, col, ship_size):
                        # Add probability to all cells this ship would occupy
                        for c in range(col, col + ship_size):
                            self.probability_map[row][c] += 1.0
            
            # Try vertical placements
            for row in range(self.board_size - ship_size + 1):
                for col in range(self.board_size):
                    if self._can_fit_ship_vertical(board, row, col, ship_size):
                        # Add probability to all cells this ship would occupy
                        for r in range(row, row + ship_size):
                            self.probability_map[r][col] += 1.0
        
        # Boost probability around known hits that haven't been completed
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.hits[row][col] and (row, col) not in self.sunk_ship_coords:
                    # This is a hit that's part of an unsunk ship
                    # Boost adjacent cells
                    self._boost_adjacent_cells(row, col, board, boost_factor=5.0)
        
        # Zero out already guessed positions
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.guesses[row][col]:
                    self.probability_map[row][col] = 0
    
    def _can_fit_ship_horizontal(self, board: Board, row: int, col: int, size: int) -> bool:
        """Check if a ship of given size can fit horizontally at position"""
        for c in range(col, col + size):
            # Can't fit if already guessed and was a miss
            if board.guesses[row][c] and not board.hits[row][c]:
                return False
            # Can't fit if it's part of a sunk ship
            if (row, c) in self.sunk_ship_coords:
                return False
        
        # If there are any hits in this range, make sure they're not part of a sunk ship
        # and that they could be part of the same ship
        hits_in_range = []
        for c in range(col, col + size):
            if board.hits[row][c]:
                if (row, c) in self.sunk_ship_coords:
                    return False
                hits_in_range.append((row, c))
        
        # If there are hits, they should all be in the same row (which they are by construction)
        # and not separated by misses
        if len(hits_in_range) > 0:
            for c in range(col, col + size):
                if board.guesses[row][c] and not board.hits[row][c]:
                    # There's a miss in this range, can't fit
                    return False
        
        return True
    
    def _can_fit_ship_vertical(self, board: Board, row: int, col: int, size: int) -> bool:
        """Check if a ship of given size can fit vertically at position"""
        for r in range(row, row + size):
            # Can't fit if already guessed and was a miss
            if board.guesses[r][col] and not board.hits[r][col]:
                return False
            # Can't fit if it's part of a sunk ship
            if (r, col) in self.sunk_ship_coords:
                return False
        
        # If there are any hits in this range, make sure they're not part of a sunk ship
        hits_in_range = []
        for r in range(row, row + size):
            if board.hits[r][col]:
                if (r, col) in self.sunk_ship_coords:
                    return False
                hits_in_range.append((r, col))
        
        # If there are hits, check no misses separate them
        if len(hits_in_range) > 0:
            for r in range(row, row + size):
                if board.guesses[r][col] and not board.hits[r][col]:
                    return False
        
        return True
    
    def _boost_adjacent_cells(self, row: int, col: int, board: Board, boost_factor: float = 5.0):
        """Boost probability of cells adjacent to a known hit"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.board_size and 0 <= new_col < self.board_size and
                not board.guesses[new_row][new_col]):
                self.probability_map[new_row][new_col] += boost_factor
    
    def _get_smallest_remaining_ship(self) -> int:
        """Get the size of the smallest ship that hasn't been sunk yet"""
        remaining_sizes = []
        for i, size in enumerate(self.ship_sizes):
            if i not in self.sunk_ships:
                remaining_sizes.append(size)
        
        return min(remaining_sizes) if remaining_sizes else 2
    
    def _generate_spaced_candidates(self, board: Board, spacing: int) -> List[Tuple[int, int]]:
        """Generate spaced grid candidates for efficient hunting"""
        valid_moves = board.get_valid_moves()
        spaced_moves = []
        
        # Use spacing pattern to reduce search space
        for move in valid_moves:
            row, col = move
            if (row + col) % spacing == 0:
                if not self._is_adjacent_to_sunk_ship(move):
                    spaced_moves.append(move)
        
        # Fallback if no spaced moves available
        if not spaced_moves:
            for move in valid_moves:
                if not self._is_adjacent_to_sunk_ship(move):
                    spaced_moves.append(move)
        
        if not spaced_moves:
            spaced_moves = valid_moves
            
        return spaced_moves
    
    def _hunt_move(self, board: Board) -> Tuple[int, int]:
        """Make a probability-guided hunt move"""
        self.mode = "hunt"
        
        # Get spacing based on smallest remaining ship
        smallest_ship = self._get_smallest_remaining_ship()
        spaced_candidates = self._generate_spaced_candidates(board, smallest_ship)
        
        if not spaced_candidates:
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            spaced_candidates = valid_moves
        
        # Among spaced candidates, choose the one with highest probability
        best_move = None
        best_prob = -1
        
        for move in spaced_candidates:
            row, col = move
            prob = self.probability_map[row][col]
            if prob > best_prob:
                best_prob = prob
                best_move = move
        
        # If all probabilities are equal (or zero), choose randomly
        if best_prob == 0 or best_move is None:
            best_move = random.choice(spaced_candidates)
        else:
            # Add some randomness: choose from top 20% of moves
            high_prob_moves = []
            threshold = best_prob * 0.8  # Within 80% of best probability
            for move in spaced_candidates:
                row, col = move
                if self.probability_map[row][col] >= threshold:
                    high_prob_moves.append(move)
            
            if high_prob_moves:
                best_move = random.choice(high_prob_moves)
        
        row, col = best_move
        
        # Check result of this move to switch to target mode if needed
        result = self._simulate_move_result(board, row, col)
        if result == "hit":
            self.mode = "target"
            self.current_ship_hits = [(row, col)]
            self._add_adjacent_targets(board, row, col)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
        return (row, col)
    
    def _target_move(self, board: Board) -> Tuple[int, int]:
        """Make a probability-guided target move when pursuing a hit ship"""
        if not self.target_stack:
            return self._hunt_move(board)
        
        # Among targets in the stack, choose the one with highest probability
        best_target = None
        best_prob = -1
        best_index = -1
        
        for i, target in enumerate(self.target_stack):
            row, col = target
            if board.guesses[row][col]:
                continue  # Skip already guessed positions
            
            prob = self.probability_map[row][col]
            if prob > best_prob:
                best_prob = prob
                best_target = target
                best_index = i
        
        if best_target is None or best_index == -1:
            # All targets already guessed, switch back to hunt
            self.target_stack = []
            return self._hunt_move(board)
        
        # Remove the chosen target from stack
        self.target_stack.pop(best_index)
        row, col = best_target
        
        result = self._simulate_move_result(board, row, col)
        
        if result == "hit":
            self.current_ship_hits.append((row, col))
            # Add new adjacent targets
            self._add_adjacent_targets(board, row, col)
            # If we have 2+ hits, focus on the line they form
            if len(self.current_ship_hits) >= 2:
                self._focus_on_ship_line(board)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        elif result == "miss":
            # Continue with remaining targets
            pass
        
        return (row, col)
    
    def _handle_sunk_ship(self, board: Board, ship_id: int):
        """Handle when a ship is sunk"""
        sunk_coords = board.get_sunk_ship_coords(ship_id)
        if sunk_coords:
            self.sunk_ship_coords.update(sunk_coords)
            
            # Determine which ship size was sunk and mark it
            ship_size = len(sunk_coords)
            for i, size in enumerate(self.ship_sizes):
                if size == ship_size and i not in self.sunk_ships:
                    self.sunk_ships.append(i)
                    break
        
        self.mode = "hunt"
        self.current_ship_hits = []
        self.target_stack = []
    
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
            if (0 <= new_row < self.board_size and 0 <= new_col < self.board_size and
                not board.guesses[new_row][new_col] and
                (new_row, new_col) not in self.sunk_ship_coords and
                (new_row, new_col) not in self.target_stack):
                self.target_stack.append((new_row, new_col))
    
    def _focus_on_ship_line(self, board: Board):
        """When we have multiple hits, focus on the line they form"""
        if len(self.current_ship_hits) < 2:
            return
        
        # Determine if ship is horizontal or vertical
        first_hit = self.current_ship_hits[0]
        second_hit = self.current_ship_hits[1]
        
        # Filter out targets that don't align with the ship direction
        new_target_stack = []
        
        if first_hit[0] == second_hit[0]:  # Same row - horizontal ship
            row = first_hit[0]
            min_col = min(hit[1] for hit in self.current_ship_hits)
            max_col = max(hit[1] for hit in self.current_ship_hits)
            
            # Prioritize targets at both ends of the line
            for col in [min_col - 1, max_col + 1]:
                if (0 <= col < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_target_stack.append((row, col))
            
            # Keep other targets in same row
            for target in self.target_stack:
                if target[0] == row and target not in new_target_stack:
                    new_target_stack.append(target)
                    
        elif first_hit[1] == second_hit[1]:  # Same col - vertical ship
            col = first_hit[1]
            min_row = min(hit[0] for hit in self.current_ship_hits)
            max_row = max(hit[0] for hit in self.current_ship_hits)
            
            # Prioritize targets at both ends of the line
            for row in [min_row - 1, max_row + 1]:
                if (0 <= row < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_target_stack.append((row, col))
            
            # Keep other targets in same column
            for target in self.target_stack:
                if target[1] == col and target not in new_target_stack:
                    new_target_stack.append(target)
        else:
            # Hits are not aligned yet, keep all targets
            return
        
        self.target_stack = new_target_stack
    
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
        self.sunk_ships = []
        self.probability_map = None
