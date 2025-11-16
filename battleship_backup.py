import asyncio
import json
import random
from typing import List, Tuple, Set, Optional
import statistics
import numpy as np

class BattleshipGame:
    """Represents a single battleship game instance with vectorized operations."""
    
    def __init__(self):
        # Use NumPy array instead of nested lists for faster operations
        self.board = np.zeros((10, 10), dtype=np.int8)
        self.ship_positions: Set[Tuple[int, int]] = set()
        self.guessed_positions: Set[Tuple[int, int]] = set()
        self.moves = 0
        self.hits = 0
        self.misses = 0
        self.ships = [5, 4, 3, 3, 2]  # Standard battleship ships
        self.ship_tiles: List[Set[Tuple[int, int]]] = []  # Track tiles for each ship
        
    def reset(self):
        """Reset the game state without creating new objects (avoid deep copy)."""
        self.board[:] = 0  # In-place reset, much faster than creating new array
        self.ship_positions.clear()
        self.guessed_positions.clear()
        self.ship_tiles.clear()
        self.moves = 0
        self.hits = 0
        self.misses = 0
          def setup_board(self):
        """Place ships randomly on the board."""
        self.ship_tiles = []  # Reset ship tracking
        for ship_size in self.ships:
            placed = False
            attempts = 0
            
            while not placed and attempts < 1000:
                horizontal = random.choice([True, False])
                row = random.randint(0, 9)
                col = random.randint(0, 9)
                
                if self._can_place_ship(row, col, ship_size, horizontal):
                    ship_positions = self._place_ship(row, col, ship_size, horizontal)
                    self.ship_tiles.append(ship_positions)
                    placed = True
                attempts += 1
                
            if not placed:
                # If we couldn't place a ship, reset and try again
                self.reset()
                self.setup_board()
                return
    
    def _can_place_ship(self, row: int, col: int, size: int, horizontal: bool) -> bool:
        """Check if a ship can be placed at the given position."""
        if horizontal:
            if col + size > 10:
                return False
            for i in range(size):
                if (row, col + i) in self.ship_positions:
                    return False
        else:
            if row + size > 10:
                return False
            for i in range(size):
                if (row + i, col) in self.ship_positions:
                    return False
        return True
    
    def _place_ship(self, row: int, col: int, size: int, horizontal: bool) -> Set[Tuple[int, int]]:
        """Place a ship on the board and return its positions."""
        ship_positions = set()
        if horizontal:
            for i in range(size):
                pos = (row, col + i)
                self.ship_positions.add(pos)
                ship_positions.add(pos)
        else:
            for i in range(size):
                pos = (row + i, col)
                self.ship_positions.add(pos)
                ship_positions.add(pos)
        return ship_positions
    
    def make_guess(self, row: int, col: int) -> tuple:
        """Make a guess at a position. Returns (result, sunk_ship_positions).
        result: 'hit', 'miss', or 'invalid'
        sunk_ship_positions: set of positions if a ship was sunk, None otherwise
        """
        if (row, col) in self.guessed_positions:
            return 'invalid', None
        
        self.guessed_positions.add((row, col))
        self.moves += 1
          if (row, col) in self.ship_positions:
            self.board[row, col] = 2  # Hit (NumPy array indexing)
            self.hits += 1
            
            # Check if this hit sank a ship
            for ship in self.ship_tiles:
                if (row, col) in ship:
                    # Check if all positions of this ship have been hit
                    if ship.issubset(self.guessed_positions):
                        return 'hit', ship
                    break            
            return 'hit', None
        else:
            self.board[row, col] = 1  # Miss (NumPy array indexing)
            self.misses += 1
            return 'miss', None
    
    def is_game_over(self) -> bool:
        """Check if all ships have been sunk."""
        return self.hits == sum(self.ships)
        return self.hits == sum(self.ships)    def get_board_state(self) -> List[List[int]]:
        """Get the current board state as a list of lists for compatibility."""
        return self.board.tolist()
    
    def get_board_state_with_sunk(self) -> List[List[int]]:
        """Get the current board state with sunk ships marked as state 3."""
        # Use copy() instead of nested list comprehension (faster)
        board_copy = self.board.copy()
        
        # Mark sunk ship positions
        for ship in self.ship_tiles:
            if ship.issubset(self.guessed_positions):
                # This ship is sunk, mark all its positions as state 3
                for row, col in ship:
                    board_copy[row, col] = 3
        
        return board_copy.tolist()


class RandomAgent:
    """Agent that makes random guesses."""
    
    def __init__(self):
        self.available_positions = [(r, c) for r in range(10) for c in range(10)]
        random.shuffle(self.available_positions)
        self.position_index = 0
    
    def get_move(self) -> Tuple[int, int]:
        """Get the next random move."""
        if self.position_index >= len(self.available_positions):
            # Should never happen in a proper game
            return (random.randint(0, 9), random.randint(0, 9))
        
        move = self.available_positions[self.position_index]
        self.position_index += 1
        return move
    
    def update(self, row: int, col: int, result: str, sunk_ship: Optional[Set[Tuple[int, int]]] = None):
        """Update agent with the result of the last move (not used by random agent)."""
        pass


class HuntAndTargetAgent:
    """Agent that hunts randomly, then targets adjacent cells when it gets a hit."""
    
    def __init__(self):
        self.available_positions = [(r, c) for r in range(10) for c in range(10)]
        random.shuffle(self.available_positions)
        self.position_index = 0
          # Tracking hits and targets
        self.hits: Set[Tuple[int, int]] = set()
        self.misses: Set[Tuple[int, int]] = set()
        self.all_guessed: Set[Tuple[int, int]] = set()  # All positions we've guessed
        self.targets: List[Tuple[int, int]] = []  # Queue of positions to target
        self.current_ship_hits: Set[Tuple[int, int]] = set()  # Hits on the current ship being targeted
    def get_move(self) -> Tuple[int, int]:
        """Get the next move using hunt and target strategy."""
        # If we have targets to pursue, target mode
        if self.targets:
            return self.targets.pop(0)
        
        # Otherwise, hunt mode - pick next random position
        while self.position_index < len(self.available_positions):
            move = self.available_positions[self.position_index]
            self.position_index += 1
            
            # Skip positions we've already guessed
            if move not in self.all_guessed:
                return move
        
        # Fallback (should never happen in a proper game)
        return (random.randint(0, 9), random.randint(0, 9))
    
    def update(self, row: int, col: int, result: str, sunk_ship: Optional[Set[Tuple[int, int]]] = None):
        """Update agent with the result of the last move."""
        # Track all guessed positions
        self.all_guessed.add((row, col))
        
        if result == 'hit':
            self.hits.add((row, col))
            self.current_ship_hits.add((row, col))
            
            # If a ship was sunk, remove its positions from our tracking
            if sunk_ship is not None:
                # Remove sunk ship positions from current tracking
                self.current_ship_hits -= sunk_ship
                
                # Remove any targets that are adjacent to or part of the sunk ship
                self.targets = [t for t in self.targets if not self._is_adjacent_or_in_ship(t, sunk_ship)]
                
                # If we cleared all targets, we're back in hunt mode automatically
            else:
                # Ship not sunk yet, add adjacent cells to targets
                self._add_adjacent_targets(row, col)
        elif result == 'miss':
            self.misses.add((row, col))
    
    def _add_adjacent_targets(self, row: int, col: int):
        """Add adjacent cells to the target list."""
        # Check all 4 adjacent cells (up, down, left, right)
        adjacent = [
            (row - 1, col),  # Up
            (row + 1, col),  # Down
            (row, col - 1),  # Left
            (row, col + 1)   # Right
        ]
        
        for adj_row, adj_col in adjacent:
            # Check if position is valid and not already guessed or targeted
            if (0 <= adj_row < 10 and 0 <= adj_col < 10 and
                (adj_row, adj_col) not in self.all_guessed and
                (adj_row, adj_col) not in self.targets):
                
                # If we have multiple hits on current ship, prioritize inline targets
                if len(self.current_ship_hits) > 1:
                    # Check if this position is inline with existing hits
                    if self._is_inline_with_hits(adj_row, adj_col):
                        self.targets.insert(0, (adj_row, adj_col))  # Add to front
                    else:
                        self.targets.append((adj_row, adj_col))  # Add to back
                else:
                    self.targets.append((adj_row, adj_col))
    
    def _is_inline_with_hits(self, row: int, col: int) -> bool:
        """Check if a position is inline with current ship hits."""
        if len(self.current_ship_hits) < 2:
            return True
        
        # Get all current hits as a list
        hits_list = list(self.current_ship_hits)
        
        # Check if hits are horizontal or vertical
        rows = [h[0] for h in hits_list]
        cols = [h[1] for h in hits_list]
        
        # If all hits are in same row (horizontal ship)
        if len(set(rows)) == 1:
            return row == rows[0]
        
        # If all hits are in same column (vertical ship)
        if len(set(cols)) == 1:
            return col == cols[0]
        
        return True
    
    def _is_adjacent_or_in_ship(self, pos: Tuple[int, int], sunk_ship: Set[Tuple[int, int]]) -> bool:
        """Check if a position is in the sunk ship or adjacent to it."""
        row, col = pos
        
        # Check if position is in the sunk ship
        if pos in sunk_ship:
            return True
        
        # Check if position is adjacent to any tile in the sunk ship
        for ship_row, ship_col in sunk_ship:
            # Check all 4 adjacent positions
            if (abs(row - ship_row) == 1 and col == ship_col) or \
               (abs(col - ship_col) == 1 and row == ship_row):
                return True
        
        return False


class ParityHuntAgent:
    """Agent that uses parity-based hunting to eliminate squares based on smallest remaining ship."""
    
    def __init__(self):
        self.available_positions = [(r, c) for r in range(10) for c in range(10)]
        random.shuffle(self.available_positions)
        self.position_index = 0
        
        # Tracking hits and targets
        self.hits: Set[Tuple[int, int]] = set()
        self.misses: Set[Tuple[int, int]] = set()
        self.all_guessed: Set[Tuple[int, int]] = set()
        self.targets: List[Tuple[int, int]] = []  # Queue of positions to target
        self.current_ship_hits: Set[Tuple[int, int]] = set()
        
        # Track remaining ships (we know these: 5, 4, 3, 3, 2)
        self.remaining_ships = [5, 4, 3, 3, 2]
        
        # Eliminated squares (not valid for current parity pattern)
        self.eliminated_squares: Set[Tuple[int, int]] = set()
        self._update_eliminated_squares()
    
    def _update_eliminated_squares(self):
        """Update the set of eliminated squares based on smallest remaining ship."""
        self.eliminated_squares.clear()
        
        if not self.remaining_ships:
            return
        
        smallest_ship = min(self.remaining_ships)
        
        # Mark squares that can't contain the smallest ship
        for row in range(10):
            for col in range(10):
                # If (row + col) is not divisible by smallest_ship, eliminate it
                if (row + col) % smallest_ship != 0:
                    self.eliminated_squares.add((row, col))
    def get_eliminated_squares(self) -> List[Tuple[int, int]]:
        """Get the current list of eliminated squares for visual display.
        Only returns squares that are currently eliminated AND not yet guessed.
        """
        # Return only the squares that are eliminated under the current parity
        # and haven't been guessed yet. This ensures that when parity changes,
        # previously eliminated squares become white again if they're now valid.
        return list(self.eliminated_squares - self.all_guessed)
    
    def get_move(self) -> Tuple[int, int]:
        """Get the next move using parity hunt and target strategy."""
        # If we have targets to pursue, target mode
        if self.targets:
            return self.targets.pop(0)
        
        # Otherwise, hunt mode - pick next valid parity position
        smallest_ship = min(self.remaining_ships) if self.remaining_ships else 1
        
        # Try to find a valid move that matches the parity pattern
        while self.position_index < len(self.available_positions):
            move = self.available_positions[self.position_index]
            self.position_index += 1
            
            # Skip positions we've already guessed
            if move in self.all_guessed:
                continue
            
            row, col = move
            # Check if position matches parity pattern
            if (row + col) % smallest_ship == 0:
                return move
        
        # Fallback: if we've exhausted parity positions, pick any remaining position
        for row in range(10):
            for col in range(10):
                if (row, col) not in self.all_guessed:
                    return (row, col)
        
        # Should never reach here in a proper game
        return (random.randint(0, 9), random.randint(0, 9))
    
    def update(self, row: int, col: int, result: str, sunk_ship: Optional[Set[Tuple[int, int]]] = None):
        """Update agent with the result of the last move."""
        # Track all guessed positions
        self.all_guessed.add((row, col))
        
        if result == 'hit':
            self.hits.add((row, col))
            self.current_ship_hits.add((row, col))
            
            # If a ship was sunk, remove it from remaining ships
            if sunk_ship is not None:
                ship_size = len(sunk_ship)
                if ship_size in self.remaining_ships:
                    self.remaining_ships.remove(ship_size)
                    # Update eliminated squares based on new smallest ship
                    self._update_eliminated_squares()
                
                # Remove sunk ship positions from current tracking
                self.current_ship_hits -= sunk_ship
                
                # Remove any targets that are adjacent to or part of the sunk ship
                self.targets = [t for t in self.targets if not self._is_adjacent_or_in_ship(t, sunk_ship)]
            else:
                # Ship not sunk yet, add adjacent cells to targets
                self._add_adjacent_targets(row, col)
        elif result == 'miss':
            self.misses.add((row, col))
    
    def _add_adjacent_targets(self, row: int, col: int):
        """Add adjacent cells to the target list."""
        # Check all 4 adjacent cells (up, down, left, right)
        adjacent = [
            (row - 1, col),  # Up
            (row + 1, col),  # Down
            (row, col - 1),  # Left
            (row, col + 1)   # Right
        ]
        
        for adj_row, adj_col in adjacent:
            # Check if position is valid and not already guessed or targeted
            if (0 <= adj_row < 10 and 0 <= adj_col < 10 and
                (adj_row, adj_col) not in self.all_guessed and
                (adj_row, adj_col) not in self.targets):
                
                # If we have multiple hits on current ship, prioritize inline targets
                if len(self.current_ship_hits) > 1:
                    # Check if this position is inline with existing hits
                    if self._is_inline_with_hits(adj_row, adj_col):
                        self.targets.insert(0, (adj_row, adj_col))  # Add to front
                    else:
                        self.targets.append((adj_row, adj_col))  # Add to back
                else:
                    self.targets.append((adj_row, adj_col))
    
    def _is_inline_with_hits(self, row: int, col: int) -> bool:
        """Check if a position is inline with current ship hits."""
        if len(self.current_ship_hits) < 2:
            return True
        
        # Get all current hits as a list
        hits_list = list(self.current_ship_hits)
        
        # Check if hits are horizontal or vertical
        rows = [h[0] for h in hits_list]
        cols = [h[1] for h in hits_list]
        
        # If all hits are in same row (horizontal ship)
        if len(set(rows)) == 1:
            return row == rows[0]
        
        # If all hits are in same column (vertical ship)
        if len(set(cols)) == 1:
            return col == cols[0]
        
        return True
    
    def _is_adjacent_or_in_ship(self, pos: Tuple[int, int], sunk_ship: Set[Tuple[int, int]]) -> bool:
        """Check if a position is in the sunk ship or adjacent to it."""
        row, col = pos
        
        # Check if position is in the sunk ship
        if pos in sunk_ship:
            return True
        
        # Check if position is adjacent to any tile in the sunk ship
        for ship_row, ship_col in sunk_ship:
            # Check all 4 adjacent positions
            if (abs(row - ship_row) == 1 and col == ship_col) or \
               (abs(col - ship_col) == 1 and row == ship_row):
                return True
        
        return False


class ProbabilityAgent:
    """Agent that uses probability heat maps to make optimal guesses."""
    
    def __init__(self):
        # Tracking hits and targets
        self.hits: Set[Tuple[int, int]] = set()
        self.misses: Set[Tuple[int, int]] = set()
        self.all_guessed: Set[Tuple[int, int]] = set()
        
        # Track remaining ships (we know these: 5, 4, 3, 3, 2)
        self.remaining_ships = [5, 4, 3, 3, 2]
        
        # Probability heat map
        self.prob_map = np.zeros((10, 10))
        
        # Track unsunk hits (hits that are part of ships we haven't sunk yet)
        self.unsunk_hits: Set[Tuple[int, int]] = set()
        
        # Shot map: 0 = unknown, 1 = miss, 2 = hit
        self.shot_map = np.zeros((10, 10), dtype=int)
        
        # Generate initial probability map
        self._generate_prob_map()
    
    def _generate_prob_map(self):
        """Generate probability heat map based on remaining ships and known information."""
        self.prob_map = np.zeros((10, 10))
        
        # For each remaining ship size, calculate probabilities
        for ship_size in self.remaining_ships:
            # Check horizontal placements
            for row in range(10):
                for col in range(10 - ship_size + 1):
                    # Check if ship can fit here horizontally
                    segment = [(row, col + i) for i in range(ship_size)]
                    if self._can_place_ship(segment):
                        # Add probability to all squares in this potential placement
                        for r, c in segment:
                            self.prob_map[r, c] += 1
            
            # Check vertical placements
            for row in range(10 - ship_size + 1):
                for col in range(10):
                    # Check if ship can fit here vertically
                    segment = [(row + i, col) for i in range(ship_size)]
                    if self._can_place_ship(segment):
                        # Add probability to all squares in this potential placement
                        for r, c in segment:
                            self.prob_map[r, c] += 1
        
        # Apply bonuses for unsunk hits
        if self.unsunk_hits:
            self._apply_hit_bonuses()
    
    def _can_place_ship(self, segment: List[Tuple[int, int]]) -> bool:
        """Check if a ship segment can be placed (doesn't overlap misses or sunk ships)."""
        for row, col in segment:
            # Can't place on a miss
            if (row, col) in self.misses:
                return False
            # Can't place on a position we already guessed (unless it's an unsunk hit)
            if (row, col) in self.all_guessed and (row, col) not in self.unsunk_hits:
                return False
        
        # If there are unsunk hits, the segment must contain at least one hit
        if self.unsunk_hits:
            has_hit = any((r, c) in self.unsunk_hits for r, c in segment)
            if not has_hit:
                return False
        
        return True
    
    def _apply_hit_bonuses(self):
        """Apply artificial weight to squares adjacent to unsunk hits."""
        for row, col in self.unsunk_hits:
            # Check if we have multiple hits in a line (horizontal or vertical)
            direction = self._get_hit_direction()
            
            if direction == 'horizontal':
                # Boost horizontal neighbors
                if col > 0 and (row, col - 1) not in self.all_guessed:
                    self.prob_map[row, col - 1] += 50
                if col < 9 and (row, col + 1) not in self.all_guessed:
                    self.prob_map[row, col + 1] += 50
            elif direction == 'vertical':
                # Boost vertical neighbors
                if row > 0 and (row - 1, col) not in self.all_guessed:
                    self.prob_map[row - 1, col] += 50
                if row < 9 and (row + 1, col) not in self.all_guessed:
                    self.prob_map[row + 1, col] += 50
            else:
                # No clear direction, boost all adjacent squares
                adjacent = [
                    (row - 1, col),  # Up
                    (row + 1, col),  # Down
                    (row, col - 1),  # Left
                    (row, col + 1)   # Right
                ]
                for adj_row, adj_col in adjacent:
                    if (0 <= adj_row < 10 and 0 <= adj_col < 10 and
                        (adj_row, adj_col) not in self.all_guessed):
                        self.prob_map[adj_row, adj_col] += 50
    
    def _get_hit_direction(self) -> str:
        """Determine if unsunk hits are horizontal, vertical, or undetermined."""
        if len(self.unsunk_hits) < 2:
            return 'none'
        
        hits_list = list(self.unsunk_hits)
        rows = [h[0] for h in hits_list]
        cols = [h[1] for h in hits_list]
        
        # If all hits are in same row (horizontal)
        if len(set(rows)) == 1:
            return 'horizontal'
        
        # If all hits are in same column (vertical)
        if len(set(cols)) == 1:
            return 'vertical'
        
        return 'none'
    
    def get_move(self) -> Tuple[int, int]:
        """Get the next move by selecting the square with highest probability."""
        # Regenerate probability map with current information
        self._generate_prob_map()
        
        # Find the square with the highest probability that we haven't guessed
        max_prob = -1
        best_move = None
        
        for row in range(10):
            for col in range(10):
                if (row, col) not in self.all_guessed:
                    if self.prob_map[row, col] > max_prob:
                        max_prob = self.prob_map[row, col]
                        best_move = (row, col)
        
        # Fallback (should never happen in a proper game)
        if best_move is None:
            for row in range(10):
                for col in range(10):
                    if (row, col) not in self.all_guessed:
                        return (row, col)
            return (random.randint(0, 9), random.randint(0, 9))
        
        return best_move
    
    def update(self, row: int, col: int, result: str, sunk_ship: Optional[Set[Tuple[int, int]]] = None):
        """Update agent with the result of the last move."""
        # Track all guessed positions
        self.all_guessed.add((row, col))
        
        if result == 'hit':
            self.hits.add((row, col))
            self.unsunk_hits.add((row, col))
            self.shot_map[row, col] = 2
            
            # If a ship was sunk, remove it from remaining ships
            if sunk_ship is not None:
                ship_size = len(sunk_ship)
                if ship_size in self.remaining_ships:
                    self.remaining_ships.remove(ship_size)
                
                # Remove sunk ship positions from unsunk hits
                self.unsunk_hits -= sunk_ship
        elif result == 'miss':
            self.misses.add((row, col))
            self.shot_map[row, col] = 1
    
    def get_prob_map(self) -> np.ndarray:
        """Get the current probability map for visualization."""
        return self.prob_map.copy()


class GameStatistics:
    """Track statistics across multiple games."""
    
    def __init__(self):
        self.game_moves: List[int] = []
        self.total_games = 0
    
    def add_game(self, moves: int):
        """Add a completed game's statistics."""
        self.game_moves.append(moves)
        self.total_games += 1
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.game_moves:
            return {
                'total_games': 0,
                'average_moves': 0,
                'best_game': 0,
                'worst_game': 0,
                'median_moves': 0
            }
        
        return {
            'total_games': self.total_games,
            'average_moves': statistics.mean(self.game_moves),
            'best_game': min(self.game_moves),
            'worst_game': max(self.game_moves),
            'median_moves': statistics.median(self.game_moves)
        }
    
    def print_summary(self):
        """Print statistics to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("GAME STATISTICS")
        print("=" * 50)
        print(f"Total Games Played: {summary['total_games']}")
        print(f"Average Moves: {summary['average_moves']:.2f}")
        print(f"Best Game (fewest moves): {summary['best_game']}")
        print(f"Worst Game (most moves): {summary['worst_game']}")
        print(f"Median Moves: {summary['median_moves']:.1f}")
        print("=" * 50)
          # Print distribution
        if self.game_moves:
            print("\nMove Distribution:")
            ranges = [(0, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 100)]
            for start, end in ranges:
                count = sum(1 for m in self.game_moves if start <= m <= end)
                if count > 0:
                    pct = (count / self.total_games) * 100
                    print(f"  {start}-{end} moves: {count} games ({pct:.1f}%)")
