import random
import numpy as np
from typing import Tuple, List, Set, Optional, Dict
from Source.board import Board
from collections import defaultdict

class OptimizedMonteCarloPlayer:
    """A Monte Carlo player using true random board sampling.
    
    This player uses the same Monte Carlo occupancy sampling algorithm as the JavaScript HTML version:
    1. Randomly samples valid complete board configurations
    2. Counts how often each cell is occupied across all successful samples
    3. Uses these occupancy counts to guide move selection
    4. Respects known misses and avoids overlapping ships in samples
    
    The algorithm:
    - For each simulation, attempts to place all remaining ships randomly
    - Ships are placed in order from largest to smallest for better success rate
    - Only placements that avoid known misses and don't overlap are accepted
    - Failed placements are rejected and don't contribute to the heat map
    - Final probability map shows relative occupancy frequency across samples
    """
    
    def __init__(self, name: str = "Optimized Monte Carlo Player", num_simulations: int = 300):
        self.name = name
        self.mode = "hunt"
        self.target_stack = []
        self.current_ship_hits = []
        self.sunk_ship_coords = set()
        self.ship_sizes = [5, 4, 3, 3, 2]
        self.sunk_ships = []
        self.board_size = 10
        self.num_simulations = num_simulations
        
        # Advanced tracking
        self.known_misses = set()
        self.known_hits = set()
        self.probability_map = None
        self.entropy_map = None
        
        # Cache for entropy calculations
        self._entropy_cache = {}
        
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a move based on Monte Carlo sampling and current strategy"""
        # Keep sunk coords and known hits in sync before scoring moves
        self._refresh_sunk_coords(board)
        self._update_state(board)
        
        # ALWAYS run Monte Carlo simulations to generate probability map for GUI display
        heat_map = self._monte_carlo_simulations_optimized(board)
        self.probability_map = heat_map
        
        # If there are unsunk hits but our target stack is empty, rebuild it now
        if not self.target_stack and self.known_hits:
            self.mode = "target"
            self.current_ship_hits = list(self.known_hits)
            self.target_stack = []
            for hr, hc in self.known_hits:
                self._add_intelligent_targets(board, hr, hc)

        if self.mode == "target" and self.target_stack:
            return self._target_move(board)
        else:
            return self._hunt_move(board)
    
    def _update_state(self, board: Board):
        """Update internal tracking of board state - optimized with vectorization"""
        # Convert to numpy arrays for faster processing
        guesses_array = np.array(board.guesses, dtype=bool)
        hits_array = np.array(board.hits, dtype=bool)
        
        # Vectorized operations to find misses and hits
        miss_positions = np.argwhere(guesses_array & ~hits_array)
        hit_positions = np.argwhere(guesses_array & hits_array)
        
        # Update sets efficiently
        self.known_misses = set(map(tuple, miss_positions))
        
        # Only include hits that aren't in sunk ships
        self.known_hits = set()
        for pos in map(tuple, hit_positions):
            if pos not in self.sunk_ship_coords:
                self.known_hits.add(pos)
    
    def _hunt_move(self, board: Board) -> Tuple[int, int]:
        """Make a Monte Carlo-guided hunt move with optimized simulations"""
        self.mode = "hunt"
        
        # Probability map already generated in make_move(), use it directly
        heat_map = self.probability_map
        
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Apply spacing strategy
        smallest_ship = self._get_smallest_remaining_ship()
        spaced_moves = self._generate_spaced_candidates(valid_moves, smallest_ship)
        candidate_moves = spaced_moves if spaced_moves else valid_moves
        
        # Enforce choosing only cells with non-zero Monte Carlo occupancy if any exist
        positive_moves = [m for m in candidate_moves if heat_map[m[0], m[1]] > 0]
        if positive_moves:
            candidate_moves = positive_moves
        else:
            any_positive = [(r, c) for r in range(self.board_size) for c in range(self.board_size)
                            if not board.guesses[r][c] and heat_map[r, c] > 0]
            if any_positive:
                candidate_moves = any_positive
        
        # Vectorized scoring of all candidates at once
        entropy_scores = self._calculate_entropy_batch(board, candidate_moves)
        
        # Combine heat map and entropy scores
        combined_scores = {}
        for move in candidate_moves:
            row, col = move
            heat_score = heat_map[row, col]
            entropy_score = entropy_scores.get(move, 0.0)
            combined_scores[move] = 0.6 * heat_score + 0.4 * entropy_score
        
        # Select best move with controlled randomness
        if combined_scores:
            max_score = max(combined_scores.values())
            if max_score > 0:
                threshold = max_score * 0.85
                top_moves = [move for move, score in combined_scores.items() if score >= threshold]
                best_move = random.choice(top_moves) if top_moves else max(combined_scores, key=combined_scores.get)
            else:
                best_move = random.choice(candidate_moves)
        else:
            best_move = random.choice(candidate_moves)
        
        row, col = best_move
        
        # Handle result
        result = self._simulate_move_result(board, row, col)
        if result == "hit":
            self.mode = "target"
            self.current_ship_hits = [(row, col)]
            self._add_intelligent_targets(board, row, col)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
        return (row, col)
    
    def _monte_carlo_simulations_optimized(self, board: Board) -> np.ndarray:
        """Run Monte Carlo simulations using true random board sampling (JavaScript HTML logic)"""
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        
        if not remaining_ships:
            return np.zeros((self.board_size, self.board_size), dtype=float)
        
        # Build misses set from board state
        misses = set()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.guesses[row][col] and not board.hits[row][col]:
                    misses.add((row, col))
        
        # Add sunk ship coordinates to misses (they block future placements)
        misses.update(self.sunk_ship_coords)
        
        # Run Monte Carlo sampling
        occupancy_map = np.zeros((self.board_size, self.board_size), dtype=int)
        accepted_samples = 0
        
        for _ in range(self.num_simulations):
            placement = self._sample_board_consistent_with(remaining_ships, misses)
            if placement is not None:
                accepted_samples += 1
                # Count occupancy for each cell in this placement
                for ship_cells in placement:
                    for row, col in ship_cells:
                        occupancy_map[row, col] += 1
        
        # Normalize to get probability map
        if accepted_samples > 0:
            heat_map = occupancy_map.astype(float) / accepted_samples
        else:
            heat_map = occupancy_map.astype(float)
        
        return heat_map
    
    def _sample_board_consistent_with(self, ships: List[int], misses: Set[Tuple[int, int]]) -> Optional[List[List[Tuple[int, int]]]]:
        """Sample a complete valid board configuration (JavaScript HTML logic).
        
        Returns a list of ship placements (each ship is a list of (row, col) coordinates),
        or None if placement fails.
        """
        # Create temporary board to track occupancy
        board = np.zeros((self.board_size, self.board_size), dtype=bool)
        
        # Sort ships largest to smallest for better placement success
        ordered_ships = sorted(ships, reverse=True)
        placements = []
        
        for ship_size in ordered_ships:
            placed = False
            max_attempts = 1024
            
            for attempt in range(max_attempts):
                # Random position and orientation
                horizontal = random.choice([True, False])
                
                if horizontal:
                    row = random.randint(0, self.board_size - 1)
                    col = random.randint(0, self.board_size - ship_size)
                    
                    # Check if can place (not in misses, not overlapping)
                    can_place = True
                    cells = []
                    for c in range(col, col + ship_size):
                        if (row, c) in misses or board[row, c]:
                            can_place = False
                            break
                        cells.append((row, c))
                    
                    if can_place:
                        # Place the ship
                        for r, c in cells:
                            board[r, c] = True
                        placements.append(cells)
                        placed = True
                        break
                else:
                    row = random.randint(0, self.board_size - ship_size)
                    col = random.randint(0, self.board_size - 1)
                    
                    # Check if can place (not in misses, not overlapping)
                    can_place = True
                    cells = []
                    for r in range(row, row + ship_size):
                        if (r, col) in misses or board[r, col]:
                            can_place = False
                            break
                        cells.append((r, col))
                    
                    if can_place:
                        # Place the ship
                        for r, c in cells:
                            board[r, c] = True
                        placements.append(cells)
                        placed = True
                        break
            
            if not placed:
                # Failed to place this ship, reject this sample
                return None
        
        return placements
    
    def _calculate_entropy_batch(self, board: Board, moves: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Calculate entropy for multiple moves in batch - optimized"""
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        
        if not remaining_ships:
            return {move: 0.0 for move in moves}
        
        # Build misses set
        misses = set()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.guesses[row][col] and not board.hits[row][col]:
                    misses.add((row, col))
        misses.update(self.sunk_ship_coords)
        
        entropy_scores = {}
        
        for move in moves:
            row, col = move
            entropy = 0.0
            
            # Check configurations for each remaining ship
            for ship_size in remaining_ships:
                # Horizontal configurations
                for start_col in range(max(0, col - ship_size + 1), min(self.board_size - ship_size + 1, col + 1)):
                    if self._can_place_ship(row, start_col, ship_size, True, misses):
                        entropy += 1.0
                
                # Vertical configurations
                for start_row in range(max(0, row - ship_size + 1), min(self.board_size - ship_size + 1, row + 1)):
                    if self._can_place_ship(start_row, col, ship_size, False, misses):
                        entropy += 1.0
            
            entropy_scores[move] = entropy
        
        return entropy_scores
    
    def _can_place_ship(self, row: int, col: int, size: int, horizontal: bool, misses: Set[Tuple[int, int]]) -> bool:
        """Check if ship can fit at given position"""
        if horizontal:
            if col + size > self.board_size:
                return False
            for c in range(col, col + size):
                if (row, c) in misses:
                    return False
        else:
            if row + size > self.board_size:
                return False
            for r in range(row, row + size):
                if (r, col) in misses:
                    return False
        
        return True
    
    def _target_move(self, board: Board) -> Tuple[int, int]:
        """Make an intelligent target move - uses full Monte Carlo probability map"""
        if not self.target_stack:
            return self._hunt_move(board)
        
        # Probability map already generated in make_move() with full Monte Carlo simulations
        # Now we combine it with target scoring for better decisions
        
        best_target = None
        best_combined_score = -float('inf')
        best_index = -1
        
        for i, target in enumerate(self.target_stack):
            row, col = target
            
            if board.guesses[row][col]:
                continue
            
            # Get Monte Carlo probability for this position
            mc_probability = self.probability_map[row][col] if self.probability_map is not None else 0.0
            # Skip cells with zero MC probability if there are other options
            if mc_probability <= 0:
                continue
            
            # Get tactical target score
            target_score = self._score_target_move(board, target)
            
            # Combine both scores: 40% Monte Carlo probability, 60% tactical targeting
            combined_score = 0.4 * mc_probability + 0.6 * target_score
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_target = target
                best_index = i
        
        if best_target is None:
            self.target_stack = []
            return self._hunt_move(board)
        
        self.target_stack.pop(best_index)
        row, col = best_target
        
        result = self._simulate_move_result(board, row, col)
        
        if result == "hit":
            self.current_ship_hits.append((row, col))
            self._add_intelligent_targets(board, row, col)
            
            if len(self.current_ship_hits) >= 2:
                self._focus_on_ship_line(board)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
        return (row, col)
    
    def _score_target_move(self, board: Board, target: Tuple[int, int]) -> float:
        """Score a target move - same logic as original"""
        row, col = target
        score = 0.0
        
        if len(self.current_ship_hits) >= 1:
            for hit_row, hit_col in self.current_ship_hits:
                if hit_row == row or hit_col == col:
                    score += 10.0
                    
                    if abs(hit_row - row) + abs(hit_col - col) == 1:
                        score += 20.0
        
        if len(self.current_ship_hits) >= 2:
            if self._extends_hit_line(target):
                score += 30.0
        
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        for ship_size in remaining_ships:
            if self._could_form_ship(board, target, ship_size):
                score += 5.0
        
        adjacent_unknowns = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.board_size and 0 <= new_col < self.board_size and
                not board.guesses[new_row][new_col]):
                adjacent_unknowns += 1
        score += adjacent_unknowns * 2.0
        
        return score
    
    def _extends_hit_line(self, target: Tuple[int, int]) -> bool:
        """Check if target extends the line formed by current hits"""
        if len(self.current_ship_hits) < 2:
            return False
        
        row, col = target
        hit_rows = [h[0] for h in self.current_ship_hits]
        hit_cols = [h[1] for h in self.current_ship_hits]
        
        if len(set(hit_rows)) == 1 and row == hit_rows[0]:
            min_col = min(hit_cols)
            max_col = max(hit_cols)
            if col == min_col - 1 or col == max_col + 1:
                return True
        
        if len(set(hit_cols)) == 1 and col == hit_cols[0]:
            min_row = min(hit_rows)
            max_row = max(hit_rows)
            if row == min_row - 1 or row == max_row + 1:
                return True
        
        return False
    
    def _could_form_ship(self, board: Board, target: Tuple[int, int], ship_size: int) -> bool:
        """Check if a ship of given size could include both target and current hits"""
        row, col = target
        all_points = self.current_ship_hits + [target]
        
        if len(all_points) > ship_size:
            return False
        
        if len(set(p[0] for p in all_points)) == 1:
            cols = [p[1] for p in all_points]
            span = max(cols) - min(cols) + 1
            if span <= ship_size:
                return True
        
        if len(set(p[1] for p in all_points)) == 1:
            rows = [p[0] for p in all_points]
            span = max(rows) - min(rows) + 1
            if span <= ship_size:
                return True
        
        return False
    
    def _add_intelligent_targets(self, board: Board, row: int, col: int):
        """Add targets intelligently based on context"""
        if len(self.current_ship_hits) <= 1:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.board_size and 0 <= new_col < self.board_size and
                    not board.guesses[new_row][new_col] and
                    (new_row, new_col) not in self.sunk_ship_coords and
                    (new_row, new_col) not in self.target_stack):
                    self.target_stack.append((new_row, new_col))
        else:
            self._focus_on_ship_line(board)
    
    def _focus_on_ship_line(self, board: Board):
        """Focus targets on the line formed by hits"""
        if len(self.current_ship_hits) < 2:
            return
        
        first_hit = self.current_ship_hits[0]
        second_hit = self.current_ship_hits[1]
        new_targets = []
        
        if first_hit[0] == second_hit[0]:
            row = first_hit[0]
            cols = [h[1] for h in self.current_ship_hits]
            min_col = min(cols)
            max_col = max(cols)
            
            for col in [min_col - 1, max_col + 1]:
                if (0 <= col < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
            
            for col in range(min_col, max_col + 1):
                if (not board.guesses[row][col] and
                    (row, col) not in self.current_ship_hits and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
        
        elif first_hit[1] == second_hit[1]:
            col = first_hit[1]
            rows = [h[0] for h in self.current_ship_hits]
            min_row = min(rows)
            max_row = max(rows)
            
            for row in [min_row - 1, max_row + 1]:
                if (0 <= row < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
            
            for row in range(min_row, max_row + 1):
                if (not board.guesses[row][col] and
                    (row, col) not in self.current_ship_hits and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
        
        self.target_stack = new_targets
    
    def _get_smallest_remaining_ship(self) -> int:
        """Get the size of the smallest ship that hasn't been sunk yet"""
        remaining_sizes = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        return min(remaining_sizes) if remaining_sizes else 2
    
    def _generate_spaced_candidates(self, valid_moves: List[Tuple[int, int]], spacing: int) -> List[Tuple[int, int]]:
        """Generate spaced grid candidates for efficient hunting"""
        spaced_moves = [move for move in valid_moves 
                       if (move[0] + move[1]) % spacing == 0 and not self._is_adjacent_to_sunk_ship(move)]
        
        if not spaced_moves:
            spaced_moves = [move for move in valid_moves if not self._is_adjacent_to_sunk_ship(move)]
        
        return spaced_moves
    
    def _handle_sunk_ship(self, board: Board, ship_id: int):
        """Handle when a ship is sunk"""
        sunk_coords = board.get_sunk_ship_coords(ship_id)
        if sunk_coords:
            self.sunk_ship_coords.update(sunk_coords)
            
            for coord in sunk_coords:
                if coord in self.known_hits:
                    self.known_hits.remove(coord)
            
            ship_size = len(sunk_coords)
            for i, size in enumerate(self.ship_sizes):
                if size == ship_size and i not in self.sunk_ships:
                    self.sunk_ships.append(i)
                    break
        
        # Remove sunk ship coordinates from current tracking
        self.current_ship_hits = [hit for hit in self.current_ship_hits if hit not in self.sunk_ship_coords]
        self.target_stack = [target for target in self.target_stack if target not in self.sunk_ship_coords]
        
        # If there are still unsunk hits, stay in target mode and rebuild target stack
        if self.known_hits:
            self.mode = "target"
            # Rebuild current ship hits from known unsunk hits
            self.current_ship_hits = list(self.known_hits)
            self.target_stack = []
            
            # Add adjacent cells of all unsunk hits to target stack
            for hit_coord in self.known_hits:
                self._add_intelligent_targets(board, hit_coord[0], hit_coord[1])
        else:
            # No unsunk hits remaining, switch to hunt mode
            self.mode = "hunt"
            self.current_ship_hits = []
            self.target_stack = []
    
    def _simulate_move_result(self, board: Board, row: int, col: int) -> str:
        """Simulate what the result of a move would be"""
        if board.guesses[row][col]:
            return 'already_guessed'
        
        if board.grid[row][col] != 0:
            ship_id = board.grid[row][col]
            for ship in board.ships:
                if ship['id'] == ship_id:
                    if ship['hits'] + 1 == ship['size']:
                        return 'sunk'
                    else:
                        return 'hit'
        
        return 'miss'
    
    def _is_adjacent_to_sunk_ship(self, coord: Tuple[int, int]) -> bool:
        """Check if a coordinate is adjacent to any sunk ship"""
        row, col = coord
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
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
        self.known_misses = set()
        self.known_hits = set()
        self.probability_map = None
        self.entropy_map = None
        self._entropy_cache.clear()

    # ---------------------- helpers to maintain target mode correctly ----------------------
    def _refresh_sunk_coords(self, board: Board):
        sunk = set()
        for ship in board.ships:
            if ship.get('sunk', False):
                for coord in ship.get('coords', []):
                    sunk.add(tuple(coord))
        self.sunk_ship_coords = sunk

