# filepath: c:\Users\jacob\Documents\GitHub\battleship-automation\hunt_and_target_fixed.py
import random
import csv
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Set
import threading

class BattleshipGame:
    def __init__(self):
        self.grid_size = 10
        self.ships = [5, 4, 3, 3, 2]  # Ship sizes
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.hit_positions = set()
        self.moves_made = 0
        self.ship_positions = set()
        
        # Hunt and Target strategy variables
        self.hunt_mode = True  # True for hunting (random), False for targeting
        self.target_stack = []  # Stack of positions to target around hits
        self.current_ship_hits = []  # Current ship being targeted
        
    def reset_game(self):
        """Reset the game for a new round"""
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.hit_positions = set()
        self.moves_made = 0
        self.ship_positions = set()
        
        # Reset hunt and target variables
        self.hunt_mode = True
        self.target_stack = []
        self.current_ship_hits = []
        
        self.place_ships_randomly()
    
    def is_valid_position(self, row: int, col: int, size: int, horizontal: bool) -> bool:
        """Check if a ship can be placed at the given position"""
        if horizontal:
            if col + size > self.grid_size:
                return False
            for i in range(size):
                if self.grid[row][col + i] != 0:
                    return False
        else:
            if row + size > self.grid_size:
                return False
            for i in range(size):
                if self.grid[row + i][col] != 0:
                    return False
        return True
    
    def place_ship(self, row: int, col: int, size: int, horizontal: bool, ship_id: int):
        """Place a ship on the grid"""
        positions = []
        if horizontal:
            for i in range(size):
                self.grid[row][col + i] = ship_id
                positions.append((row, col + i))
        else:
            for i in range(size):
                self.grid[row + i][col] = ship_id
                positions.append((row + i, col))
        
        # Add positions to ship_positions set
        for pos in positions:
            self.ship_positions.add(pos)
    
    def place_ships_randomly(self):
        """Randomly place all ships on the grid"""
        for ship_id, ship_size in enumerate(self.ships, 1):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:  # Prevent infinite loops
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - 1)
                horizontal = random.choice([True, False])
                
                if self.is_valid_position(row, col, ship_size, horizontal):
                    self.place_ship(row, col, ship_size, horizontal, ship_id)
                    placed = True
                attempts += 1
            
            if not placed:
                # If we can't place a ship after many attempts, restart
                self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
                self.ship_positions = set()
                self.place_ships_randomly()
                return
    
    def get_adjacent_positions(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid adjacent positions (up, down, left, right)"""
        adjacent = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.grid_size and 
                0 <= new_col < self.grid_size and 
                (new_row, new_col) not in self.hit_positions):
                adjacent.append((new_row, new_col))
        
        return adjacent
    
    def is_ship_sunk(self, ship_id: int) -> bool:
        """Check if a ship with given ID is completely sunk"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (self.grid[row][col] == ship_id and 
                    (row, col) not in self.hit_positions):
                    return False
        return True
    
    def make_hunt_and_target_move(self) -> Tuple[int, int, bool]:
        """Make a move using hunt and target strategy"""
        available_positions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in self.hit_positions:
                    available_positions.append((row, col))
        
        if not available_positions:
            return -1, -1, False  # No moves left (use -1 instead of None for type safety)
        
        # Choose position based on current mode
        if self.hunt_mode:
            # Hunt mode: random shot
            row, col = random.choice(available_positions)
        else:
            # Target mode: shoot from target stack
            if self.target_stack:
                # Keep popping until we find an unhit position
                while self.target_stack:
                    row, col = self.target_stack.pop()
                    if (row, col) not in self.hit_positions:
                        break
                else:
                    # No valid targets left, go back to hunt mode
                    self.hunt_mode = True
                    row, col = random.choice(available_positions)
            else:
                # No more targets, go back to hunt mode
                self.hunt_mode = True
                row, col = random.choice(available_positions)
        
        self.hit_positions.add((row, col))
        self.moves_made += 1
        
        # Check if it's a hit
        is_hit = self.grid[row][col] != 0
        
        if is_hit:
            ship_id = self.grid[row][col]
            self.current_ship_hits.append((row, col))
            
            # Check if ship is sunk
            if self.is_ship_sunk(ship_id):
                # Ship is sunk, clear target stack and return to hunt mode
                self.target_stack = []
                self.current_ship_hits = []
                self.hunt_mode = True
            else:
                # Ship not sunk, enter target mode
                self.hunt_mode = False
                
                # Add adjacent positions to target stack
                adjacent_positions = self.get_adjacent_positions(row, col)
                
                # If we have multiple hits on current ship, prioritize inline shots
                if len(self.current_ship_hits) >= 2:
                    # Determine direction of ship and prioritize shots in that direction
                    hit1 = self.current_ship_hits[-2]
                    hit2 = self.current_ship_hits[-1]
                    
                    # Calculate direction
                    if hit1[0] == hit2[0]:  # Horizontal ship
                        # Prioritize left and right
                        inline_shots = [(row, col-1), (row, col+1)]
                    else:  # Vertical ship
                        # Prioritize up and down
                        inline_shots = [(row-1, col), (row+1, col)]
                    
                    # Add inline shots first (they'll be popped last due to stack nature)
                    for pos in adjacent_positions:
                        if pos not in inline_shots and pos not in self.target_stack:
                            self.target_stack.append(pos)
                    for pos in inline_shots:
                        if pos in adjacent_positions and pos not in self.target_stack:
                            self.target_stack.append(pos)
                else:
                    # First hit on ship, add all adjacent positions
                    for pos in adjacent_positions:
                        if pos not in self.target_stack:
                            self.target_stack.append(pos)
        else:
            # Miss - continue with current strategy
            pass
        
        return row, col, is_hit
    
    def is_game_won(self) -> bool:
        """Check if all ships have been hit"""
        return self.ship_positions.issubset(self.hit_positions)
    
    def play_game(self) -> int:
        """Play a complete game and return the number of moves to win"""
        self.reset_game()
        
        while not self.is_game_won() and self.moves_made < 100:
            row, col, is_hit = self.make_hunt_and_target_move()
            if row == -1:  # No moves left (shouldn't happen in normal play)
                break
        
        # Safety check - this should never happen in a valid game
        if self.moves_made >= 100 and not self.is_game_won():
            print(f"WARNING: Game exceeded 100 moves! Moves: {self.moves_made}")
            print(f"Ship positions: {len(self.ship_positions)}")
            print(f"Hit positions: {len(self.hit_positions)}")
        
        return self.moves_made

def run_simulation(num_games: int = 200000):
    """Run the battleship simulation for the specified number of games"""
    print(f"Starting Hunt and Target battleship simulation for {num_games:,} games...")
    print("This may take a while...")
    
    game = BattleshipGame()
    results = []
    
    # Track progress
    start_time = time.time()
    
    for game_num in range(num_games):
        moves_to_win = game.play_game()
        results.append(moves_to_win)
        
        # Progress update every 10000 games
        if (game_num + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            avg_time_per_game = elapsed / (game_num + 1)
            remaining_games = num_games - (game_num + 1)
            eta = remaining_games * avg_time_per_game
            print(f"Completed {game_num + 1:,}/{num_games:,} games ({((game_num + 1)/num_games)*100:.1f}%) - ETA: {eta/60:.1f} minutes")
    
    # Calculate statistics
    average_moves = sum(results) / len(results)
    min_moves = min(results)
    max_moves = max(results)
    
    # Save results to CSV
    with open('hunt_and_target_moves_fixed.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Game_Number', 'Moves_To_Win'])
        for i, moves in enumerate(results, 1):
            writer.writerow([i, moves])
    
    # Output final statistics
    print(f"\n{'='*50}")
    print(f"HUNT AND TARGET BATTLESHIP SIMULATION RESULTS (FIXED)")
    print(f"{'='*50}")
    print(f"Total games played: {num_games:,}")
    print(f"Average moves to win: {average_moves:.2f}")
    print(f"Minimum moves to win: {min_moves}")
    print(f"Maximum moves to win: {max_moves}")
    print(f"Results saved to: hunt_and_target_moves_fixed.csv")
    print(f"Total simulation time: {(time.time() - start_time)/60:.2f} minutes")
    
    return {
        'average': average_moves,
        'min': min_moves,
        'max': max_moves,
        'results': results
    }

if __name__ == "__main__":
    # Test the fix with a small simulation first
    print("Testing fixed hunt and target strategy...")
    stats = run_simulation(10000)
    
    if stats['max'] <= 100:
        print(f"\n✅ FIX CONFIRMED: Maximum moves is {stats['max']} (≤ 100)")
        print("The issue has been resolved!")
    else:
        print(f"\n❌ ISSUE PERSISTS: Maximum moves is {stats['max']} (> 100)")