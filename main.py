import tkinter as tk
from tkinter import messagebox
from game import Game
from analysis import analyze_and_save_results
import time
import statistics
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import os
import glob
import importlib.util
import inspect
from datetime import datetime
from tqdm import tqdm

class BattleshipGUI:
    def __init__(self, num_games):
        self.num_games = num_games
        self.current_game = 0
        self.games_results = []
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Battleship Automation - GUI Mode")
        self.root.geometry("800x700")
        self.root.configure(bg="#2c3e50")
        
        # Create info frame
        self.info_frame = tk.Frame(self.root, bg="#2c3e50")
        self.info_frame.pack(pady=10)
        
        self.game_label = tk.Label(self.info_frame, text=f"Game 1 of {num_games}", 
                                  font=("Arial", 14, "bold"), fg="white", bg="#2c3e50")
        self.game_label.pack()
        
        self.move_label = tk.Label(self.info_frame, text="Move: 0", 
                                  font=("Arial", 12), fg="white", bg="#2c3e50")
        self.move_label.pack()
        
        # Create grid frame
        self.grid_frame = tk.Frame(self.root, bg="#34495e", relief="solid", bd=2)
        self.grid_frame.pack(expand=True, padx=20, pady=20)
        
        # Create button grid
        self.buttons = []
        for row in range(10):
            button_row = []
            for col in range(10):
                button = tk.Button(
                    self.grid_frame,
                    width=4,
                    height=2,
                    text="",
                    bg="#3498db",
                    fg="white",
                    relief="raised",
                    borderwidth=2,
                    font=("Arial", 8, "bold"),
                    state="disabled"
                )
                button.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(button)
            self.buttons.append(button_row)
        
        # Control buttons
        self.control_frame = tk.Frame(self.root, bg="#2c3e50")
        self.control_frame.pack(pady=10)
        
        self.start_button = tk.Button(self.control_frame, text="Start Games", 
                                     command=self.start_games, font=("Arial", 12, "bold"),
                                     bg="#27ae60", fg="white", padx=20)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.pause_button = tk.Button(self.control_frame, text="Pause Games", 
                                     command=self.pause_games, font=("Arial", 12, "bold"),
                                     bg="#e67e22", fg="white", padx=20, state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=10)
        
        self.next_button = tk.Button(self.control_frame, text="Next Move", 
                                    command=self.next_move, font=("Arial", 12),
                                    bg="#f39c12", fg="white", padx=20, state="disabled")
        self.next_button.pack(side=tk.LEFT, padx=10)
          # Game state
        self.game = None
        self.player = None  # Will be set externally
        self.auto_playing = False
        self.games_paused = False
        self.next_move_pos = None  # Track next move position for highlighting
        
    def start_games(self):
        """Start the first game and begin auto-play"""
        self.current_game = 0
        self.games_results = []
        self.games_paused = False
        self.auto_playing = True
        self.start_new_game()
        
    def pause_games(self):
        """Pause or resume the automatic game progression"""
        self.games_paused = not self.games_paused
        self.auto_playing = not self.games_paused
        
        if self.games_paused:
            self.pause_button.config(text="Resume Games", bg="#e74c3c")
            self.next_button.config(state="normal")
        else:
            self.pause_button.config(text="Pause Games", bg="#e67e22")
            self.next_button.config(state="disabled")
            # Resume auto-play if we're in the middle of a game or need to start next game
            if self.game and not self.game.board.is_game_over():
                self.auto_play_step()
            elif self.current_game < self.num_games:
                # Need to continue to next game
                self.root.after(1, self.start_new_game)
        
    def start_new_game(self):
        """Start a new game"""
        self.current_game += 1
        self.game = Game(self.player)
        self.game.setup_game()
        
        # Update labels
        self.game_label.config(text=f"Game {self.current_game} of {self.num_games}")
        self.move_label.config(text="Move: 0")
        
        # Get the first move and highlight it
        self.next_move_pos = self.player.make_move(self.game.board)
        
        # Reset grid and show ships
        self.update_grid()
        
        # Enable controls
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        if self.games_paused:
            self.next_button.config(state="normal")
        else:
            self.next_button.config(state="disabled")
            # Start auto-play immediately if not paused
            self.root.after(1, self.auto_play_step)
        
    def update_grid(self):
        """Update the visual grid"""
        board_state = self.game.board.get_board_state()
        
        for row in range(10):
            for col in range(10):
                button = self.buttons[row][col]
                
                # Check if this position has been guessed
                if board_state['guesses'][row][col]:
                    if board_state['hits'][row][col]:
                        # Hit
                        button.config(bg="#e74c3c", text="HIT", fg="white")
                    else:
                        # Miss
                        button.config(bg="#95a5a6", text="MISS", fg="black")
                elif self.next_move_pos and (row, col) == self.next_move_pos:
                    # Highlight next move in pink
                    if board_state['grid'][row][col] != 0:
                        # Ship present - pink with ship indicator
                        button.config(bg="#ff69b4", text="NEXT", fg="white")
                    else:
                        # Water - pink highlight
                        button.config(bg="#ff69b4", text="NEXT", fg="white")
                else:
                    # Show ships (for demonstration)
                    if board_state['grid'][row][col] != 0:
                        # Ship present
                        button.config(bg="#2ecc71", text="SHIP", fg="white")
                    else:
                        # Water
                        button.config(bg="#3498db", text="", fg="white")
    
    def next_move(self):
        """Execute the next move"""
        if self.game.board.is_game_over():
            self.finish_current_game()
            return
            
        # Execute the already planned move
        if self.next_move_pos:
            row, col = self.next_move_pos
            result = self.game.board.make_guess(row, col)
            self.game.move_count += 1
            
            # Plan the next move (if game continues)
            if not self.game.board.is_game_over():
                self.next_move_pos = self.player.make_move(self.game.board)
            else:
                self.next_move_pos = None
            
            # Update display
            self.move_label.config(text=f"Move: {self.game.move_count} - Last: ({row},{col}) = {result.upper()}")
            self.update_grid()
            
            # Check if game is over
            if self.game.board.is_game_over():
                self.root.after(1, self.finish_current_game)  # Wait 1 second before finishing
    
    def finish_current_game(self):
        """Finish the current game and start next or show results"""
        # Record game result
        result = {
            'moves': self.game.move_count,
            'game_number': self.current_game
        }
        self.games_results.append(result)
        
        if self.current_game < self.num_games and not self.games_paused:
            # Automatically start next game if not paused
            self.root.after(1, self.start_new_game)
        elif self.current_game < self.num_games and self.games_paused:
            # Game finished but paused - wait for user to resume
            self.move_label.config(text=f"Game {self.current_game} completed in {self.game.move_count} moves. Paused.")
        else:
            # All games finished, show results
            self.show_final_results()
    
    def auto_play_step(self):
        """Execute one step of auto-play"""
        if self.auto_playing and not self.games_paused and self.game and not self.game.board.is_game_over():
            self.next_move()
            self.root.after(1, self.auto_play_step)  # Continue instantly
    
    def show_final_results(self):
        """Show final results of all games"""
        if not self.games_results:
            return
            
        move_counts = [result['moves'] for result in self.games_results]
        best_game = min(self.games_results, key=lambda x: x['moves'])
        worst_game = max(self.games_results, key=lambda x: x['moves'])
        average_moves = sum(move_counts) / len(move_counts)
        
        result_text = f"""BATTLESHIP AUTOMATION RESULTS
{'='*40}
Total games played: {self.num_games}

MOVE STATISTICS:
Best game: {best_game['moves']} moves (Game #{best_game['game_number']})
Worst game: {worst_game['moves']} moves (Game #{worst_game['game_number']})
Average moves: {average_moves:.2f}
{'='*40}"""
        
        messagebox.showinfo("Game Results", result_text)
        print(result_text)  # Also print to console
          # Reset for new games
        self.start_button.config(state="normal")
        self.pause_button.config(state="disabled", text="Pause Games", bg="#e67e22")
        self.next_button.config(state="disabled")
        self.auto_playing = False
        self.games_paused = False
        self.next_move_pos = None
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def play_single_game(game_num, module_name, class_name, player_name):
    """Worker function to play a single game - designed for multiprocessing"""
    # Import the module and get the class (fixes pickling issues)
    spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    player_class = getattr(module, class_name)
    
    # Create a new player instance for this process
    player = player_class(player_name)
    game = Game(player)
    result = game.play_game(show_moves=False)
    return result

def run_simulation_games_batch(num_games: int, player, player_module_name: str, player_class_name: str, 
                               batch_size: int = 10000, show_gui: bool = False):
    """Run multiple games in batches to avoid memory issues"""
    print(f"\nRunning {num_games} games with {player.name}...")
    
    # Determine batch size based on total games
    if num_games <= batch_size:
        # If total games is small, just run normally
        return run_simulation_games(num_games, player, player_module_name, player_class_name, show_gui)
    
    # Calculate number of batches
    num_batches = (num_games + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of up to {batch_size} games each...")
      # Track aggregated statistics across all batches
    all_move_counts = []
    total_time = 0
    best_overall = float('inf')
    worst_overall = 0
    
    overall_start_time = time.time()
    
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, num_games)
        batch_games = batch_end - batch_start
        
        print(f"\nBatch {batch_num + 1}/{num_batches}: Games {batch_start + 1}-{batch_end}")
        
        # Run this batch
        batch_results = run_simulation_games(batch_games, player, player_module_name, 
                                            player_class_name, show_gui=False)
        
        # Extract only the statistics we need (don't keep full results)
        batch_move_counts = batch_results['stats']['move_counts']
        all_move_counts.extend(batch_move_counts)
        
        total_time += batch_results['stats']['total_time']
        best_overall = min(best_overall, batch_results['stats']['best_moves'])
        worst_overall = max(worst_overall, batch_results['stats']['worst_moves'])
        
        # Save checkpoint after each batch
        save_checkpoint(player.name, batch_num + 1, num_batches, all_move_counts, 
                       best_overall, worst_overall, total_time)
        
        # Clear batch results to free memory
        del batch_results
        import gc
        gc.collect()
        
        # Show batch summary
        print(f"Batch {batch_num + 1} complete | Total: {len(all_move_counts)}/{num_games} games | "
              f"Avg: {sum(all_move_counts) / len(all_move_counts):.2f} moves | "
              f"Best: {best_overall} | Worst: {worst_overall}")
    
    overall_end_time = time.time()
    
    # Calculate final statistics
    average_moves = sum(all_move_counts) / len(all_move_counts)
    median_moves = statistics.median(all_move_counts)
    std_dev = statistics.stdev(all_move_counts) if len(all_move_counts) > 1 else 0
    
    # Display final results
    print(f"\n{'='*60}")
    print(f"FINAL BATTLESHIP SIMULATION RESULTS - {player.name}")
    print(f"{'='*60}")
    print(f"Total games played: {num_games}")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Average time per game: {(overall_end_time - overall_start_time) / num_games:.3f} seconds")
    print(f"\nMOVE STATISTICS:")
    print(f"Best game (least moves): {best_overall} moves")
    print(f"Worst game (most moves): {worst_overall} moves")
    print(f"Average moves: {average_moves:.2f}")
    print(f"Median moves: {median_moves:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"{'='*60}")
    
    return {
        'results': [],  # Don't store individual results for memory efficiency
        'stats': {
            'player_name': player.name,
            'num_games': num_games,
            'total_time': overall_end_time - overall_start_time,
            'avg_time_per_game': (overall_end_time - overall_start_time) / num_games,
            'best_moves': best_overall,
            'worst_moves': worst_overall,
            'average_moves': average_moves,
            'median_moves': median_moves,
            'std_dev': std_dev,
            'move_counts': all_move_counts
        }
    }


def run_simulation_games(num_games: int, player, player_module_name: str, player_class_name: str, show_gui: bool = False):
    """Run multiple games and return statistics - optimized for memory efficiency"""
    print(f"\nRunning {num_games} games with {player.name}...")
    
    # Determine number of processes to use
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_processes} parallel processes for faster execution...")
    
    # Only store move counts, not full results, to save memory
    move_counts = []
    best_moves = float('inf')
    worst_moves = 0
    
    start_time = time.time()
    
    # Get player info for multiprocessing
    player_name = player.name
    
    # Create a partial function with player info
    worker_func = partial(play_single_game, module_name=player_module_name, 
                         class_name=player_class_name, player_name=player_name)
    
    # Use multiprocessing pool with progress bar
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better progress tracking
        with tqdm(total=num_games, desc=f"{player.name}", unit="games", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for games_completed, result in enumerate(pool.imap_unordered(worker_func, range(num_games)), 1):
                # Only store the move count, not the entire result object
                moves = result['moves']
                move_counts.append(moves)
                best_moves = min(best_moves, moves)
                worst_moves = max(worst_moves, moves)
                
                # Update progress bar
                pbar.update(1)
    
    end_time = time.time()
    
    # Calculate statistics
    average_moves = sum(move_counts) / len(move_counts)
    median_moves = statistics.median(move_counts)
    std_dev = statistics.stdev(move_counts) if len(move_counts) > 1 else 0
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BATTLESHIP SIMULATION RESULTS - {player.name}")
    print(f"{'='*60}")
    print(f"Total games played: {num_games}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Average time per game: {(end_time - start_time) / num_games:.3f} seconds")
    print(f"\nMOVE STATISTICS:")
    print(f"Best game (least moves): {best_moves} moves")
    print(f"Worst game (most moves): {worst_moves} moves")
    print(f"Average moves: {average_moves:.2f}")
    print(f"Median moves: {median_moves:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"{'='*60}")
    
    return {
        'results': [],  # Don't store individual results for memory efficiency
        'stats': {
            'player_name': player.name,
            'num_games': num_games,
            'total_time': end_time - start_time,
            'avg_time_per_game': (end_time - start_time) / num_games,
            'best_moves': best_moves,
            'worst_moves': worst_moves,
            'average_moves': average_moves,
            'median_moves': median_moves,
            'std_dev': std_dev,
            'move_counts': move_counts
        }
    }


def save_checkpoint(player_name: str, batch_num: int, total_batches: int, 
                    move_counts: list, best_moves: float, worst_moves: float, total_time: float):
    """Save checkpoint data to file for recovery"""
    checkpoint_dir = "Data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_file = f"{checkpoint_dir}/{player_name.replace(' ', '_')}_batch_{batch_num}_of_{total_batches}_{timestamp}.txt"
    
    with open(checkpoint_file, 'w') as f:
        f.write(f"Player: {player_name}\n")
        f.write(f"Batch: {batch_num}/{total_batches}\n")
        f.write(f"Games completed: {len(move_counts)}\n")
        f.write(f"Best moves: {best_moves}\n")
        f.write(f"Worst moves: {worst_moves}\n")
        f.write(f"Average moves: {sum(move_counts) / len(move_counts):.2f}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Timestamp: {timestamp}\n")





def discover_player_classes():
    """Dynamically discover all player classes in *player.py files"""
    player_classes = []
    player_files = glob.glob("*player.py")
    
    for file_path in sorted(player_files):
        try:
            # Get module name from file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {file_path}")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all classes in the module that look like player classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if class has required methods (make_move and reset)
                if (hasattr(obj, 'make_move') and 
                    hasattr(obj, 'reset') and 
                    obj.__module__ == module_name):
                    player_classes.append({
                        'name': name,
                        'class': obj,
                        'module': module_name,
                        'file': file_path
                    })
                    break  # Only take first valid class from each file
        except Exception as e:
            print(f"Warning: Could not load player from {file_path}: {e}")
    
    return player_classes

def get_bot_choice(player_classes):
    """Get user choice for bot type"""
    print("\nAVAILABLE BOT TYPES:")
    
    # Display individual player options
    for i, player_info in enumerate(player_classes, 1):
        # Create a temp instance to get the default name
        try:
            temp_instance = player_info['class']()
            display_name = temp_instance.name
        except:
            display_name = player_info['name']
        print(f"{i}. {display_name} (from {player_info['file']})")
    
    # Add "compare all" option
    compare_all_option = len(player_classes) + 1
    print(f"{compare_all_option}. Compare All Players - Run all bots and compare performance")
    
    while True:
        try:
            choice = int(input(f"\nSelect bot type (1-{compare_all_option}): "))
            if 1 <= choice <= compare_all_option:
                return choice
            else:
                print(f"Please enter a number between 1 and {compare_all_option}")
        except ValueError:
            print("Please enter a valid number")

def get_user_choice(player_classes):
    """Get user preferences for running the simulation"""
    print("BATTLESHIP AUTOMATION")
    print("=" * 30)
    
    # Get bot choice first
    bot_choice = get_bot_choice(player_classes)
    
    # Ask about number of games
    while True:
        try:
            num_games = int(input("How many games do you want to run? "))
            if num_games > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Ask about GUI (only for single bot, not comparison)
    show_gui = False
    compare_all_option = len(player_classes) + 1
    if bot_choice != compare_all_option:  # Single bot mode
        while True:
            gui_choice = input("Do you want to run with GUI? (y/n): ").lower().strip()
            if gui_choice in ['y', 'yes']:
                show_gui = True
                break
            elif gui_choice in ['n', 'no']:
                show_gui = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    return bot_choice, show_gui, num_games

def main():
    # Discover all player classes
    player_classes = discover_player_classes()
    
    if not player_classes:
        print("Error: No player classes found! Make sure you have *player.py files in the directory.")
        return
    
    print(f"Found {len(player_classes)} player classes:")
    for player_info in player_classes:
        print(f"  - {player_info['name']} (from {player_info['file']})")
    
    bot_choice, show_gui, num_games = get_user_choice(player_classes)
    
    # Determine batch size based on number of games
    # Use smaller batches for very large runs to manage memory
    if num_games >= 100000:
        batch_size = 100000
    else:
        batch_size = num_games  # No batching for small runs
    
    compare_all_option = len(player_classes) + 1
    # Single player mode
    if bot_choice < compare_all_option:
        player_info = player_classes[bot_choice - 1]
        player_instance = player_info['class']()
        
        if show_gui:
            print(f"\nStarting GUI mode for {num_games} games with {player_instance.name}...")
            gui = BattleshipGUI(num_games)
            gui.player = player_instance
            gui.run()
        else:
            print(f"\nStarting simulation mode with {player_instance.name}...")
            results = run_simulation_games_batch(num_games, player_instance, 
                                          player_info['module'], player_info['name'], 
                                          batch_size=batch_size, show_gui=False)
            analyze_and_save_results([results], "single-bot")
    # Compare all players
    else:
        print(f"\nStarting comparison mode - running {num_games} games with each bot...")
        print("This will run background simulations only (no GUI for comparison mode)")
        
        all_results = []
        for i, player_info in enumerate(player_classes, 1):
            player_instance = player_info['class']()
            
            print(f"\n{'='*60}")
            print(f"PHASE {i}: Testing {player_instance.name}")
            print(f"{'='*60}")
            
            results = run_simulation_games_batch(num_games, player_instance, 
                                          player_info['module'], player_info['name'],
                                          batch_size=batch_size, show_gui=False)
            all_results.append(results)
        
        # Analyze and save results
        comparison_name = f"{len(player_classes)}-bot-comparison"
        analyze_and_save_results(all_results, comparison_name)


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    multiprocessing.freeze_support()
    main()