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
        
    def reset_game(self):
        """Reset the game for a new round"""
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.hit_positions = set()
        self.moves_made = 0
        self.ship_positions = set()
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
    
    def make_random_move(self) -> Tuple[int, int, bool]:
        """Make a random move on positions not yet hit"""
        available_positions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in self.hit_positions:
                    available_positions.append((row, col))
        
        if not available_positions:
            return None, None, False  # No moves left
        
        row, col = random.choice(available_positions)
        self.hit_positions.add((row, col))
        self.moves_made += 1
        
        # Check if it's a hit
        is_hit = self.grid[row][col] != 0
        return row, col, is_hit
    
    def is_game_won(self) -> bool:
        """Check if all ships have been hit"""
        return self.ship_positions.issubset(self.hit_positions)
    
    def play_game(self) -> int:
        """Play a complete game and return the number of moves to win"""
        self.reset_game()
        
        while not self.is_game_won():
            row, col, is_hit = self.make_random_move()
            if row is None:  # No moves left (shouldn't happen in normal play)
                break
        
        return self.moves_made

class BattleshipGUI:
    def __init__(self, game: BattleshipGame):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Battleship Game Simulation")
        self.root.geometry("800x650")
        self.root.resizable(False, False)
        
        # Colors for different cell states
        self.colors = {
            'empty': '#87CEEB',      # Sky blue for water
            'ship': '#4169E1',       # Royal blue for ships
            'hit': '#FF4500',        # Orange red for hits
            'miss': '#FFFFFF',       # White for misses
            'sunk': '#8B0000'        # Dark red for sunk ships
        }
        
        self.cell_size = 40
        self.buttons = []
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Battleship Game Simulation", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Info frame
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=5)
        
        self.moves_label = tk.Label(self.info_frame, text="Moves: 0", 
                                   font=("Arial", 12))
        self.moves_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(self.info_frame, text="Status: Ready", 
                                    font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Grid frame
        self.grid_frame = tk.Frame(self.root, bg='black')
        self.grid_frame.pack(pady=10)
        
        # Create grid buttons
        for row in range(self.game.grid_size):
            button_row = []
            for col in range(self.game.grid_size):
                btn = tk.Button(self.grid_frame, 
                               width=3, height=1,
                               bg=self.colors['empty'],
                               font=("Arial", 8),
                               relief='raised',
                               bd=1)
                btn.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(btn)
            self.buttons.append(button_row)
        
        # Control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(self.control_frame, text="Single Game", 
                                  command=self.start_single_game,
                                  font=("Arial", 12))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.sim_btn = tk.Button(self.control_frame, text="Run 10 Games", 
                                command=lambda: self.start_simulation(10),
                                font=("Arial", 12))
        self.sim_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(self.control_frame, text="Reset", 
                                  command=self.reset_display,
                                  font=("Arial", 12))
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Legend
        self.legend_frame = tk.Frame(self.root)
        self.legend_frame.pack(pady=10)
        
        legend_items = [
            ("Water", self.colors['empty']),
            ("Ship", self.colors['ship']),
            ("Hit", self.colors['hit']),
            ("Miss", self.colors['miss'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            legend_btn = tk.Button(self.legend_frame, text=text, bg=color,
                                  width=8, height=1, font=("Arial", 10))
            legend_btn.pack(side=tk.LEFT, padx=5)
    
    def update_display(self):
        """Update the visual display of the grid"""
        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                btn = self.buttons[row][col]
                
                if (row, col) in self.game.hit_positions:
                    if self.game.grid[row][col] != 0:  # Hit a ship
                        btn.config(bg=self.colors['hit'], text='X')
                    else:  # Missed
                        btn.config(bg=self.colors['miss'], text='•')
                elif self.game.grid[row][col] != 0:  # Ship (only show during game)
                    btn.config(bg=self.colors['ship'], text='S')
                else:  # Empty water
                    btn.config(bg=self.colors['empty'], text='')
        
        # Update info labels
        self.moves_label.config(text=f"Moves: {self.game.moves_made}")
        if self.game.is_game_won():
            self.status_label.config(text="Status: Game Won!")
        else:
            self.status_label.config(text="Status: Playing...")
    
    def reset_display(self):
        """Reset the visual display"""
        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                btn = self.buttons[row][col]
                btn.config(bg=self.colors['empty'], text='')
        
        self.moves_label.config(text="Moves: 0")
        self.status_label.config(text="Status: Ready")
        self.start_btn.config(state='normal')
        self.sim_btn.config(state='normal')
    
    def start_single_game(self):
        """Start a single game with visual updates"""
        self.start_btn.config(state='disabled')
        self.sim_btn.config(state='disabled')
        self.game.reset_game()
        self.update_display()
        self.root.after(500, self.play_step)
    
    def play_step(self):
        """Play one step of the game"""
        if not self.game.is_game_won():
            row, col, is_hit = self.game.make_random_move()
            if row is not None:
                self.update_display()
                self.root.after(100, self.play_step)  # Continue after 100ms
        else:
            self.start_btn.config(state='normal')
            self.sim_btn.config(state='normal')
    
    def start_simulation(self, num_games: int = 10):
        """Start a simulation of multiple games with visual updates"""
        self.start_btn.config(state='disabled')
        self.sim_btn.config(state='disabled')
        self.current_game = 0
        self.total_games = num_games
        self.total_moves = 0
        self.results = []
        
        self.status_label.config(text=f"Simulation: Game 1/{num_games}")
        self.run_next_game()
    
    def run_next_game(self):
        """Run the next game in the simulation"""
        if self.current_game < self.total_games:
            self.current_game += 1
            self.game.reset_game()
            self.update_display()
            self.status_label.config(text=f"Simulation: Game {self.current_game}/{self.total_games}")
            self.root.after(1000, self.play_simulation_step)
        else:
            # Simulation complete
            avg_moves = sum(self.results) / len(self.results) if self.results else 0
            self.status_label.config(text=f"Simulation Complete! Avg: {avg_moves:.1f} moves")
            self.start_btn.config(state='normal')
            self.sim_btn.config(state='normal')
    
    def play_simulation_step(self):
        """Play one step of the current simulation game"""
        if not self.game.is_game_won():
            row, col, is_hit = self.game.make_random_move()
            if row is not None:
                self.update_display()
                self.root.after(50, self.play_simulation_step)  # Faster for simulation
        else:
            # Game complete, record result and move to next
            self.results.append(self.game.moves_made)
            self.root.after(500, self.run_next_game)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def run_simulation(num_games: int = 200000):
    """Run the battleship simulation for the specified number of games"""
    print(f"Starting battleship simulation for {num_games:,} games...")
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
    with open('random_moves.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Game_Number', 'Moves_To_Win'])
        for i, moves in enumerate(results, 1):
            writer.writerow([i, moves])
    
    # Output final statistics
    print(f"\n{'='*50}")
    print(f"BATTLESHIP SIMULATION RESULTS")
    print(f"{'='*50}")
    print(f"Total games played: {num_games:,}")
    print(f"Average moves to win: {average_moves:.2f}")
    print(f"Minimum moves to win: {min_moves}")
    print(f"Maximum moves to win: {max_moves}")
    print(f"Results saved to: random_moves.csv")
    print(f"Total simulation time: {(time.time() - start_time)/60:.2f} minutes")
    
    return {
        'average': average_moves,
        'min': min_moves,
        'max': max_moves,
        'results': results
    }


# Ask user for mode selection
print("Battleship Game Options:")
print("1. Visual Game Mode (GUI)")
print("2. Simulation Mode (200,000 games)")

choice = input("Select mode (1 or 2): ").strip()

if choice == "1":
    # Run visual game mode
    print("Starting visual game mode...")
    game = BattleshipGame()
    gui = BattleshipGUI(game)
    gui.run()
elif choice == "2":
    # Run the simulation
    stats = run_simulation(200000)
else:
    print("Invalid choice. Running visual mode by default...")
    game = BattleshipGame()
    gui = BattleshipGUI(game)
    gui.run()
