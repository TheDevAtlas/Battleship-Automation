import random
from typing import List, Tuple, Optional

class Board:
    def __init__(self):
        self.size = 10
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.guesses = [[False for _ in range(self.size)] for _ in range(self.size)]
        self.hits = [[False for _ in range(self.size)] for _ in range(self.size)]
        self.ships = []
        self.ship_sizes = [5, 4, 3, 3, 2]  # Carrier, Battleship, Cruiser, Submarine, Destroyer
        self.total_ship_cells = sum(self.ship_sizes)
        self.hit_count = 0
        
    def place_ships_randomly(self):
        """Randomly place all ships on the board"""
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.ships = []
        
        for ship_id, ship_size in enumerate(self.ship_sizes, 1):
            placed = False
            attempts = 0
            max_attempts = 1000
            
            while not placed and attempts < max_attempts:
                # Random position and orientation
                row = random.randint(0, self.size - 1)
                col = random.randint(0, self.size - 1)
                horizontal = random.choice([True, False])
                
                if self._can_place_ship(row, col, ship_size, horizontal):
                    self._place_ship(row, col, ship_size, horizontal, ship_id)
                    placed = True
                
                attempts += 1
            
            if not placed:
                # If we can't place a ship after many attempts, restart
                return self.place_ships_randomly()
    
    def _can_place_ship(self, row: int, col: int, size: int, horizontal: bool) -> bool:
        """Check if a ship can be placed at the given position"""
        if horizontal:
            if col + size > self.size:
                return False
            for c in range(col, col + size):
                if self.grid[row][c] != 0:
                    return False
        else:
            if row + size > self.size:
                return False
            for r in range(row, row + size):
                if self.grid[r][col] != 0:
                    return False
        return True
    
    def _place_ship(self, row: int, col: int, size: int, horizontal: bool, ship_id: int):
        """Place a ship on the board"""
        ship_coords = []
        if horizontal:
            for c in range(col, col + size):
                self.grid[row][c] = ship_id
                ship_coords.append((row, c))
        else:
            for r in range(row, row + size):
                self.grid[r][col] = ship_id
                ship_coords.append((r, col))
        
        self.ships.append({
            'id': ship_id,
            'size': size,
            'coords': ship_coords,
            'hits': 0,
            'sunk': False
        })
    
    def make_guess(self, row: int, col: int) -> str:
        """Make a guess at the given coordinates. Returns 'hit', 'miss', 'sunk', or 'already_guessed'"""
        if self.guesses[row][col]:
            return 'already_guessed'
        
        self.guesses[row][col] = True
        
        if self.grid[row][col] != 0:  # Hit
            self.hits[row][col] = True
            self.hit_count += 1
            
            # Update ship hit count
            ship_id = self.grid[row][col]
            for ship in self.ships:
                if ship['id'] == ship_id:
                    ship['hits'] += 1
                    # Check if ship is sunk
                    if ship['hits'] == ship['size']:
                        ship['sunk'] = True
                        return 'sunk'
                    break
            
            return 'hit'
        else:  # Miss
            return 'miss'
    
    def is_game_over(self) -> bool:
        """Check if all ships have been sunk"""
        return self.hit_count == self.total_ship_cells
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all coordinates that haven't been guessed yet"""
        valid_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if not self.guesses[row][col]:
                    valid_moves.append((row, col))
        return valid_moves
    
    def get_sunk_ship_coords(self, ship_id: int) -> List[Tuple[int, int]]:
        """Get coordinates of a sunk ship"""
        for ship in self.ships:
            if ship['id'] == ship_id and ship.get('sunk', False):
                return ship['coords']
        return []
    
    def get_board_state(self) -> dict:
        """Get current state of the board for display purposes"""
        return {
            'grid': self.grid,
            'guesses': self.guesses,
            'hits': self.hits,
            'ships': self.ships,
            'hit_count': self.hit_count,
            'total_moves': sum(sum(row) for row in self.guesses)
        }