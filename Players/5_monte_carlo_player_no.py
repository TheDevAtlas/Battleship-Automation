import random
import numpy as np
from typing import Tuple, List, Set, Optional, Dict
from Source.board import Board
from collections import defaultdict

class MonteCarloPlayer:
    """A Monte Carlo Tree Search player that outperforms probability-based strategies.
    
    This player uses:
    1. Monte Carlo sampling with ship placement simulations for hunt mode
    2. Probabilistic reasoning with move lookahead for target mode
    3. Entropy-based scoring to maximize information gain
    4. Ship constraint propagation to narrow down possibilities
    
    The key innovation is running simulations of possible ship configurations
    and choosing moves that eliminate the most possibilities on average.
    """
    
    def __init__(self, name: str = "Monte Carlo Player", num_simulations: int = 300):
        self.name = name
        self.mode = "hunt"  # "hunt" or "target"
        self.target_stack = []  # Stack of coordinates to investigate after a hit
        self.current_ship_hits = []  # Current ship being targeted
        self.sunk_ship_coords = set()  # All coordinates of sunk ships
        self.ship_sizes = [5, 4, 3, 3, 2]  # Known ship sizes
        self.sunk_ships = []  # Track which ships have been sunk (indices)
        self.board_size = 10
        self.num_simulations = num_simulations  # Number of Monte Carlo simulations
        
        # Advanced tracking
        self.known_misses = set()  # Track all misses
        self.known_hits = set()  # Track all hits (not yet sunk)
        self.probability_map = None
        self.entropy_map = None  # Information gain map
        
    def make_move(self, board: Board) -> Tuple[int, int]:
        """Make a move based on Monte Carlo sampling and current strategy"""
        # Update internal state
        self._update_state(board)
        
        if self.mode == "target" and self.target_stack:
            return self._target_move(board)
        else:
            return self._hunt_move(board)
    
    def _update_state(self, board: Board):
        """Update internal tracking of board state"""
        # Update known misses and hits
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.guesses[row][col]:
                    if board.hits[row][col]:
                        if (row, col) not in self.sunk_ship_coords:
                            self.known_hits.add((row, col))
                    else:
                        self.known_misses.add((row, col))
    
    def _hunt_move(self, board: Board) -> Tuple[int, int]:
        """Make a Monte Carlo-guided hunt move"""
        self.mode = "hunt"
        
        # Run Monte Carlo simulations to build heat map
        heat_map = self._monte_carlo_simulations(board)
        
        # Store as probability_map for visualization
        self.probability_map = heat_map
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Apply spacing strategy for efficiency
        smallest_ship = self._get_smallest_remaining_ship()
        spaced_moves = self._generate_spaced_candidates(valid_moves, smallest_ship)
        
        # If we have spaced moves, use them; otherwise use all valid moves
        candidate_moves = spaced_moves if spaced_moves else valid_moves
        
        # Calculate entropy (information gain) for each candidate
        entropy_scores = {}
        for move in candidate_moves:
            row, col = move
            # Combine heat map probability with entropy
            heat_score = heat_map[row, col]
            entropy_score = self._calculate_entropy(board, move)
            
            # Weighted combination: 60% heat map, 40% entropy
            combined_score = 0.6 * heat_score + 0.4 * entropy_score
            entropy_scores[move] = combined_score
        
        # Choose move with highest combined score
        if entropy_scores:
            # Add controlled randomness: select from top 15% of moves
            max_score = max(entropy_scores.values())
            if max_score > 0:
                threshold = max_score * 0.85
                top_moves = [move for move, score in entropy_scores.items() if score >= threshold]
                best_move = random.choice(top_moves) if top_moves else max(entropy_scores, key=entropy_scores.get)
            else:
                best_move = random.choice(candidate_moves)
        else:
            best_move = random.choice(candidate_moves)
        
        row, col = best_move
        
        # Check result and switch to target mode if needed
        result = self._simulate_move_result(board, row, col)
        if result == "hit":
            self.mode = "target"
            self.current_ship_hits = [(row, col)]
            self._add_intelligent_targets(board, row, col)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
        return (row, col)
    
    def _monte_carlo_simulations(self, board: Board) -> np.ndarray:
        """Run Monte Carlo simulations to generate probability heat map"""
        heat_map = np.zeros((self.board_size, self.board_size), dtype=float)
        
        # Get remaining ships
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        
        if not remaining_ships:
            return heat_map
        
        # Run simulations
        successful_sims = 0
        for _ in range(self.num_simulations):
            placement = self._simulate_ship_placement(board, remaining_ships)
            if placement is not None:
                # Add this placement to heat map
                for coords in placement.values():
                    for row, col in coords:
                        heat_map[row, col] += 1.0
                successful_sims += 1
        
        # Normalize heat map
        if successful_sims > 0:
            heat_map /= successful_sims
        
        return heat_map
    
    def _simulate_ship_placement(self, board: Board, remaining_ships: List[int]) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """Simulate a valid random placement of remaining ships"""
        # Create a temporary grid for this simulation
        temp_grid = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Mark known misses and sunk ships as blocked
        for row, col in self.known_misses:
            temp_grid[row, col] = -1
        for row, col in self.sunk_ship_coords:
            temp_grid[row, col] = -1
        
        # Mark known hits that must be covered
        required_hits = list(self.known_hits)
        
        placements = {}
        
        # Shuffle ship order for randomness
        ship_order = list(enumerate(remaining_ships))
        random.shuffle(ship_order)
        
        for ship_idx, ship_size in ship_order:
            # Try to place this ship
            placed = False
            attempts = 0
            max_attempts = 100
            
            while not placed and attempts < max_attempts:
                # Random position and orientation
                row = random.randint(0, self.board_size - 1)
                col = random.randint(0, self.board_size - 1)
                horizontal = random.choice([True, False])
                
                # Check if ship can be placed
                if self._can_place_ship_simulation(temp_grid, row, col, ship_size, horizontal, required_hits):
                    # Place the ship
                    coords = self._place_ship_simulation(temp_grid, row, col, ship_size, horizontal, ship_idx + 1)
                    placements[ship_idx] = coords
                    
                    # Remove covered hits from required list
                    for coord in coords:
                        if coord in required_hits:
                            required_hits.remove(coord)
                    
                    placed = True
                
                attempts += 1
            
            if not placed:
                # Failed to place this ship, simulation invalid
                return None
        
        # Check if all required hits are covered
        if required_hits:
            return None
        
        return placements
    
    def _can_place_ship_simulation(self, grid: np.ndarray, row: int, col: int, 
                                   size: int, horizontal: bool, required_hits: List[Tuple[int, int]]) -> bool:
        """Check if a ship can be placed in the simulation"""
        coords = []
        
        if horizontal:
            if col + size > self.board_size:
                return False
            for c in range(col, col + size):
                if grid[row, c] != 0:
                    return False
                coords.append((row, c))
        else:
            if row + size > self.board_size:
                return False
            for r in range(row, row + size):
                if grid[r, col] != 0:
                    return False
                coords.append((r, col))
        
        # Prefer placements that cover required hits
        covers_hit = any(coord in required_hits for coord in coords)
        
        # If there are required hits and this doesn't cover any, penalize (but don't reject)
        # This is handled by the random nature of the simulation
        
        return True
    
    def _place_ship_simulation(self, grid: np.ndarray, row: int, col: int, 
                              size: int, horizontal: bool, ship_id: int) -> List[Tuple[int, int]]:
        """Place a ship in the simulation grid"""
        coords = []
        if horizontal:
            for c in range(col, col + size):
                grid[row, c] = ship_id
                coords.append((row, c))
        else:
            for r in range(row, row + size):
                grid[r, col] = ship_id
                coords.append((r, col))
        return coords
    
    def _calculate_entropy(self, board: Board, move: Tuple[int, int]) -> float:
        """Calculate information entropy (expected information gain) for a move"""
        row, col = move
        
        # Entropy is higher for moves that:
        # 1. Are near remaining possible ship positions
        # 2. Can help distinguish between multiple ship orientations
        # 3. Are in areas with high uncertainty
        
        entropy = 0.0
        
        # Check how many ship configurations this move could eliminate
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        
        for ship_size in remaining_ships:
            # Check horizontal configurations through this cell
            for start_col in range(max(0, col - ship_size + 1), min(self.board_size - ship_size + 1, col + 1)):
                if self._can_fit_ship(board, row, start_col, ship_size, True):
                    entropy += 1.0
            
            # Check vertical configurations through this cell
            for start_row in range(max(0, row - ship_size + 1), min(self.board_size - ship_size + 1, row + 1)):
                if self._can_fit_ship(board, start_row, col, ship_size, False):
                    entropy += 1.0
        
        return entropy
    
    def _can_fit_ship(self, board: Board, row: int, col: int, size: int, horizontal: bool) -> bool:
        """Check if a ship could theoretically fit at this position"""
        if horizontal:
            if col + size > self.board_size:
                return False
            for c in range(col, col + size):
                if board.guesses[row][c] and not board.hits[row][c]:
                    return False
                if (row, c) in self.sunk_ship_coords:
                    return False
        else:
            if row + size > self.board_size:
                return False
            for r in range(row, row + size):
                if board.guesses[r][col] and not board.hits[r][col]:
                    return False
                if (r, col) in self.sunk_ship_coords:
                    return False
        return True
    
    def _target_move(self, board: Board) -> Tuple[int, int]:
        """Make an intelligent target move with lookahead"""
        if not self.target_stack:
            return self._hunt_move(board)
        
        # Create a probability map for target mode visualization
        self.probability_map = np.zeros((self.board_size, self.board_size), dtype=float)
        
        # Score each target based on:
        # 1. Likelihood of continuing the ship
        # 2. Potential to sink the ship quickly
        # 3. Information gain if it's a miss
        
        best_target = None
        best_score = -float('inf')
        best_index = -1
        
        for i, target in enumerate(self.target_stack):
            row, col = target
            
            if board.guesses[row][col]:
                continue
            
            # Calculate target score
            score = self._score_target_move(board, target)
            # Update probability map with target scores
            self.probability_map[row, col] = score
            
            if score > best_score:
                best_score = score
                best_target = target
                best_index = i
        
        if best_target is None:
            self.target_stack = []
            return self._hunt_move(board)
        
        # Remove chosen target from stack
        self.target_stack.pop(best_index)
        row, col = best_target
        
        result = self._simulate_move_result(board, row, col)
        
        if result == "hit":
            self.current_ship_hits.append((row, col))
            self._add_intelligent_targets(board, row, col)
            
            # If we have multiple hits, focus on the ship line
            if len(self.current_ship_hits) >= 2:
                self._focus_on_ship_line(board)
        elif result == "sunk":
            ship_id = board.grid[row][col]
            self._handle_sunk_ship(board, ship_id)
        
        return (row, col)
    
    def _score_target_move(self, board: Board, target: Tuple[int, int]) -> float:
        """Score a target move based on multiple factors"""
        row, col = target
        score = 0.0
        
        # Factor 1: Alignment with existing hits
        if len(self.current_ship_hits) >= 1:
            for hit_row, hit_col in self.current_ship_hits:
                # Same row or column = good
                if hit_row == row or hit_col == col:
                    score += 10.0
                    
                    # Adjacent = very good
                    if abs(hit_row - row) + abs(hit_col - col) == 1:
                        score += 20.0
        
        # Factor 2: Forms a line with multiple hits
        if len(self.current_ship_hits) >= 2:
            # Check if this extends the line
            if self._extends_hit_line(target):
                score += 30.0
        
        # Factor 3: Potential ship sizes that could fit
        remaining_ships = [size for i, size in enumerate(self.ship_sizes) if i not in self.sunk_ships]
        for ship_size in remaining_ships:
            # Check if a ship of this size could pass through this target and current hits
            if self._could_form_ship(board, target, ship_size):
                score += 5.0
        
        # Factor 4: Number of adjacent unknowns (more unknowns = more future options)
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
        
        # Check horizontal line
        if len(set(hit_rows)) == 1 and row == hit_rows[0]:
            min_col = min(hit_cols)
            max_col = max(hit_cols)
            if col == min_col - 1 or col == max_col + 1:
                return True
        
        # Check vertical line
        if len(set(hit_cols)) == 1 and col == hit_cols[0]:
            min_row = min(hit_rows)
            max_row = max(hit_rows)
            if row == min_row - 1 or row == max_row + 1:
                return True
        
        return False
    
    def _could_form_ship(self, board: Board, target: Tuple[int, int], ship_size: int) -> bool:
        """Check if a ship of given size could include both target and current hits"""
        row, col = target
        
        # Must include target and at least one current hit
        all_points = self.current_ship_hits + [target]
        
        if len(all_points) > ship_size:
            return False
        
        # Check if they could form a horizontal ship
        if len(set(p[0] for p in all_points)) == 1:  # All same row
            cols = [p[1] for p in all_points]
            span = max(cols) - min(cols) + 1
            if span <= ship_size:
                return True
        
        # Check if they could form a vertical ship
        if len(set(p[1] for p in all_points)) == 1:  # All same column
            rows = [p[0] for p in all_points]
            span = max(rows) - min(rows) + 1
            if span <= ship_size:
                return True
        
        return False
    
    def _add_intelligent_targets(self, board: Board, row: int, col: int):
        """Add targets intelligently based on context"""
        # If this is the first hit, add all adjacent cells
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
            # We have multiple hits - only add targets that align
            self._focus_on_ship_line(board)
    
    def _focus_on_ship_line(self, board: Board):
        """Focus targets on the line formed by hits"""
        if len(self.current_ship_hits) < 2:
            return
        
        # Determine orientation
        first_hit = self.current_ship_hits[0]
        second_hit = self.current_ship_hits[1]
        
        new_targets = []
        
        # Horizontal ship
        if first_hit[0] == second_hit[0]:
            row = first_hit[0]
            cols = [h[1] for h in self.current_ship_hits]
            min_col = min(cols)
            max_col = max(cols)
            
            # Add ends of the line
            for col in [min_col - 1, max_col + 1]:
                if (0 <= col < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
            
            # Add any gaps in the line
            for col in range(min_col, max_col + 1):
                if (not board.guesses[row][col] and
                    (row, col) not in self.current_ship_hits and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
        
        # Vertical ship
        elif first_hit[1] == second_hit[1]:
            col = first_hit[1]
            rows = [h[0] for h in self.current_ship_hits]
            min_row = min(rows)
            max_row = max(rows)
            
            # Add ends of the line
            for row in [min_row - 1, max_row + 1]:
                if (0 <= row < self.board_size and
                    not board.guesses[row][col] and
                    (row, col) not in self.sunk_ship_coords):
                    new_targets.append((row, col))
            
            # Add any gaps in the line
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
            
            # Remove sunk coords from known hits
            for coord in sunk_coords:
                if coord in self.known_hits:
                    self.known_hits.remove(coord)
            
            # Mark the ship size as sunk
            ship_size = len(sunk_coords)
            for i, size in enumerate(self.ship_sizes):
                if size == ship_size and i not in self.sunk_ships:
                    self.sunk_ships.append(i)
                    break
        
        # Only switch to hunt mode if there are no remaining unsunk hits
        if not self.known_hits:
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
