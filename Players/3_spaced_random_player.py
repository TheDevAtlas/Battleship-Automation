import random
from typing import Tuple, List, Set, Optional
from Source.board import Board

class SpacedRandomPlayer:
    """A player that uses spacing strategies to reduce guesses while maintaining randomness.
    
    Key insight: For a ship of size N, we only need to check every Nth square in a grid pattern
    to guarantee we'll hit it. This dramatically reduces the search space while staying random.
    
    For example:
    - Size 2 ship: Check every 2nd square (50% reduction)
    - Size 3 ship: Check every 3rd square (67% reduction)  
    - Size 4 ship: Check every 4th square (75% reduction)
    - Size 5 ship: Check every 5th square (80% reduction)
    """
    
    def __init__(self, name: str = "Spaced Random Player"):
        self.name = name
        self.mode = "hunt"  # "hunt" or "target"
        self.target_stack = []  # Stack of coordinates to investigate after a hit
        self.current_ship_hits = []  # Current ship being targeted
        self.sunk_ship_coords = set()  # All coordinates of sunk ships
        self.ship_sizes = [5, 4, 3, 3, 2]  # Known ship sizes
        self.sunk_ships = []  # Track which ships have been sunk
        
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a move based on current strategy (hunt or target)"""
        # Sync state first so we don't drop target mode if unsunk hits remain
        self._sync_from_board(board)
        if self.mode == "target" and self.target_stack:
            return self._target_move(board)
        else:
            return self._hunt_move(board)
    
    def _get_smallest_remaining_ship(self) -> int:
        """Get the size of the smallest ship that hasn't been sunk yet"""
        remaining_sizes = []
        for i, size in enumerate(self.ship_sizes):
            if i not in self.sunk_ships:
                remaining_sizes.append(size)
        
        return min(remaining_sizes) if remaining_sizes else 2
    
    def _generate_spaced_grid(self, board: Board, spacing: int) -> List[Tuple[int, int]]:
        """Generate a spaced grid pattern for hunting
        
        For spacing N, we check positions where (row + col) % N == 0
        This aligns all checks on diagonals parallel to the main diagonal,
        ensuring we hit any ship of size N while minimizing overlap.
        
        Example for spacing=2 on 10x10 board:
        X . X . X . X . X .
        . X . X . X . X . X
        X . X . X . X . X .
        (where X marks valid hunt positions)
        """
        valid_moves = board.get_valid_moves()
        spaced_moves = []
        
        # Use a single offset (0) to align all checks on the same diagonal pattern
        # This ensures efficient coverage without redundancy
        for move in valid_moves:
            row, col = move
            if (row + col) % spacing == 0:
                if not self._is_adjacent_to_sunk_ship(move):
                    spaced_moves.append(move)
        
        # If no spaced moves available (e.g., all adjacent to sunk ships),
        # fall back to any valid move not adjacent to sunk ships
        if not spaced_moves:
            for move in valid_moves:
                if not self._is_adjacent_to_sunk_ship(move):
                    spaced_moves.append(move)
        
        # If still no moves, use any valid move
        if not spaced_moves:
            spaced_moves = valid_moves
            
        return spaced_moves
    
    def _hunt_move(self, board: Board) -> Tuple[int, int]:
        """Make a spaced random move during hunt phase"""
        self.mode = "hunt"
        
        # Determine spacing based on smallest remaining ship
        smallest_ship = self._get_smallest_remaining_ship()
        
        # Generate spaced grid candidates
        spaced_candidates = self._generate_spaced_grid(board, smallest_ship)
        
        if not spaced_candidates:
            # Fallback to any valid move
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            row, col = random.choice(valid_moves)
        else:
            row, col = random.choice(spaced_candidates)
        
        # Check result of this move to switch to target mode if needed
        result = self._simulate_move_result(board, row, col)
        if result == "hit":
            self.mode = "target"
            self.current_ship_hits = [(row, col)]
            self._add_adjacent_targets(board, row, col)
        elif result == "sunk":
            # Ship was sunk with this hit, track it
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
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
            self._handle_sunk_ship(board, ship_id)
        
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
        # Rebuild targets from any other unsunk hits; otherwise go to hunt mode
        self._rebuild_targets_from_unsunk_hits(board)
    
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
        self.sunk_ships = []

    # ---------------------- helpers to maintain target mode correctly ----------------------
    def _sync_from_board(self, board: Board):
        """Refresh sunk coords and rebuild targets if needed before choosing a move."""
        self._refresh_sunk_coords(board)
        if not self.target_stack:
            self._rebuild_targets_from_unsunk_hits(board)

    def _refresh_sunk_coords(self, board: Board):
        sunk = set()
        for ship in board.ships:
            if ship.get('sunk', False):
                for coord in ship.get('coords', []):
                    sunk.add(tuple(coord))
        self.sunk_ship_coords = sunk

    def _rebuild_targets_from_unsunk_hits(self, board: Board):
        unsunk_hits = []
        for r in range(board.size):
            for c in range(board.size):
                if board.hits[r][c] and (r, c) not in self.sunk_ship_coords:
                    unsunk_hits.append((r, c))
        if unsunk_hits:
            self.mode = "target"
            self.current_ship_hits = list(unsunk_hits)
            self.target_stack = []
            for hr, hc in unsunk_hits:
                self._add_adjacent_targets(board, hr, hc)
            # Deduplicate and drop already-guessed
            seen = set()
            deduped = []
            for t in self.target_stack:
                if t not in seen and not board.guesses[t[0]][t[1]] and t not in self.sunk_ship_coords:
                    seen.add(t)
                    deduped.append(t)
            self.target_stack = deduped
        else:
            self.mode = "hunt"
            self.current_ship_hits = []
