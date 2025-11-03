from Source.game import Game
from Source.analysis import analyze_and_save_results
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
import threading
import webbrowser
from Source.web_server import start_server, update_game_state, set_paused, set_running

class BattleshipGUI:
    def __init__(self, num_games):
        self.num_games = num_games
        self.current_game = 0
        self.games_results = []
        
        # Game state
        self.game = None
        self.player = None  # Will be set externally
        self.auto_playing = False
        self.games_paused = False
        self.next_move_pos = None  # Track next move position for highlighting
        
        # Start the web server in a background thread
        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()
        
        # Give the server a moment to start
        time.sleep(1)
        
        # Open the browser
        webbrowser.open('http://127.0.0.1:5000')
        
        print("\n" + "="*60)
        print("Web GUI started at http://127.0.0.1:5000")
        print("The browser should open automatically.")
        print("Starting games automatically...")
        print("="*60 + "\n")
        
    def start_games(self):
        """Start the first game and begin auto-play"""
        self.current_game = 0
        self.games_results = []
        self.games_paused = False
        self.auto_playing = True
        set_running(True)
        self.start_new_game()
        
    def start_new_game(self):
        """Start a new game"""
        self.current_game += 1
        self.game = Game(self.player)
        self.game.setup_game()
        
        # Get the first move
        self.next_move_pos = self.player.make_move(self.game.board)
        
        # Update web display
        self.update_web_display()
        
    def update_web_display(self, last_move=None):
        """Update the web display with current game state"""
        board_state = self.game.board.get_board_state()
        
        # Get probability map from player if available
        probability_map = None
        if hasattr(self.player, 'probability_map') and self.player.probability_map is not None:
            probability_map = self.player.probability_map
        
        update_game_state(
            board_state,
            self.current_game,
            self.num_games,
            self.game.move_count,
            last_move,
            probability_map
        )
    
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
            
            # Prepare last move info for display
            last_move_info = {
                'position': [row, col],
                'result': result
            }
            
            # Plan the next move (if game continues)
            if not self.game.board.is_game_over():
                self.next_move_pos = self.player.make_move(self.game.board)
            else:
                self.next_move_pos = None
            
            # Update web display
            self.update_web_display(last_move_info)
            
            # Check if game is over
            if self.game.board.is_game_over():
                time.sleep(0.1)  # Brief pause before finishing
                self.finish_current_game()
    
    def finish_current_game(self):
        """Finish the current game and start next or show results"""
        # Record game result
        result = {
            'moves': self.game.move_count,
            'game_number': self.current_game
        }
        self.games_results.append(result)
        
        print(f"Game {self.current_game} completed in {self.game.move_count} moves.")
        
        if self.current_game >= self.num_games:
            # All games finished, show results
            self.auto_playing = False
            self.show_final_results()
        else:
            # Brief pause between games
            time.sleep(0.5)
    
    def play_all_games(self):
        """Play all games in a loop"""
        while self.current_game < self.num_games and self.auto_playing and not self.games_paused:
            # Start a new game
            self.start_new_game()
            
            # Play through the entire game
            while not self.game.board.is_game_over() and self.auto_playing and not self.games_paused:
                self.next_move()
                time.sleep(0.1)  # Small delay for smooth animation
            
            # Finish the current game
            if self.game.board.is_game_over():
                self.finish_current_game()
    
    def show_final_results(self):
        """Show final results of all games"""
        if not self.games_results:
            return
            
        move_counts = [result['moves'] for result in self.games_results]
        best_game = min(self.games_results, key=lambda x: x['moves'])
        worst_game = max(self.games_results, key=lambda x: x['moves'])
        average_moves = sum(move_counts) / len(move_counts)
        
        result_text = f"""
{'='*60}
BATTLESHIP AUTOMATION RESULTS
{'='*60}
Total games played: {self.num_games}

MOVE STATISTICS:
Best game: {best_game['moves']} moves (Game #{best_game['game_number']})
Worst game: {worst_game['moves']} moves (Game #{worst_game['game_number']})
Average moves: {average_moves:.2f}
{'='*60}
"""
        
        print(result_text)
        
        # Reset for new games
        self.auto_playing = False
        self.games_paused = False
        self.next_move_pos = None
        set_running(False)
    
    def run(self):
        """Start the GUI - automatically begins playing games"""
        # Start games automatically
        self.start_games()
        
        # Play all games
        try:
            self.play_all_games()
        except KeyboardInterrupt:
            print("\nGames interrupted by user.")
            self.auto_playing = False
            set_running(False)
        
        # Keep server running briefly to view final state
        print("\nAll games complete. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            set_running(False)

def play_single_game(game_num, module_name, class_name, player_name):
    """Worker function to play a single game - designed for multiprocessing"""
    # Import the module and get the class (fixes pickling issues)
    spec = importlib.util.spec_from_file_location(module_name, f"Players/{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    player_class = getattr(module, class_name)
    
    # Create a new player instance for this process
    player = player_class(player_name)
    game = Game(player)
    result = game.play_game(show_moves=False)
    
    # Explicitly reset player state to free memory
    if hasattr(player, 'reset'):
        player.reset()
    
    # Clean up references
    del player
    del game
    
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
    print(f"Processing in {num_batches} batches (for memory management)...")
    
    # Determine number of processes to use - USE ALL CPUs for maximum performance
    num_processes = cpu_count()  # Use ALL CPU cores
    print(f"Using {num_processes} parallel processes (ALL CPU cores) for maximum performance...")
    
    # Track aggregated statistics across all batches
    all_move_counts = []
    best_overall = float('inf')
    worst_overall = 0
    
    overall_start_time = time.time()
    
    # Create a partial function with player info
    worker_func = partial(play_single_game, module_name=player_module_name, 
                         class_name=player_class_name, player_name=player.name)    # Single unified progress bar for all batches
    with tqdm(total=num_games, desc=f"{player.name}", unit="games", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Process each batch with a fresh Pool to prevent memory leaks
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, num_games)
            batch_games = batch_end - batch_start
            
            # Create a new Pool for each batch - this ensures complete cleanup
            # Use a low maxtasksperchild to recycle workers frequently
            with Pool(processes=num_processes, maxtasksperchild=100) as pool:
                # Run this batch with progress updates - use chunksize for better throughput
                chunksize = max(1, batch_games // (num_processes * 4))
                for result in pool.imap_unordered(worker_func, range(batch_start, batch_end), chunksize=chunksize):
                    # Store move count
                    moves = result['moves']
                    all_move_counts.append(moves)
                    best_overall = min(best_overall, moves)
                    worst_overall = max(worst_overall, moves)
                    
                    # Update the unified progress bar
                    pbar.update(1)
                
                # Explicitly close and join the pool to ensure cleanup
                pool.close()
                pool.join()
            
            # Save checkpoint after each batch
            save_checkpoint(player.name, batch_num + 1, num_batches, all_move_counts, 
                           best_overall, worst_overall, time.time() - overall_start_time)
            
            # Aggressive garbage collection after each batch
            import gc
            gc.collect()
    
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
    
    # Determine number of processes to use - USE ALL CPUs for maximum performance
    num_processes = cpu_count()  # Use ALL CPU cores
    print(f"Using {num_processes} parallel processes (ALL CPU cores) for maximum performance...")
    
    # Only store move counts, not full results, to save memory
    move_counts = []
    best_moves = float('inf')
    worst_moves = 0
    
    start_time = time.time()
    
    # Get player info for multiprocessing
    player_name = player.name
    
    # Create a partial function with player info
    worker_func = partial(play_single_game, module_name=player_module_name, 
                         class_name=player_class_name, player_name=player_name)    # Use multiprocessing pool with progress bar
    # Use low maxtasksperchild to prevent memory leaks - recycle workers frequently
    with Pool(processes=num_processes, maxtasksperchild=100) as pool:
        # Use imap_unordered for better progress tracking with chunksize for better throughput
        chunksize = max(1, num_games // (num_processes * 4))
        with tqdm(total=num_games, desc=f"{player.name}", unit="games", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for games_completed, result in enumerate(pool.imap_unordered(worker_func, range(num_games), chunksize=chunksize), 1):
                # Only store the move count, not the entire result object
                moves = result['moves']
                move_counts.append(moves)
                best_moves = min(best_moves, moves)
                worst_moves = max(worst_moves, moves)
                
                # Update progress bar
                pbar.update(1)
                
                # Periodic garbage collection to prevent memory buildup
                if games_completed % 500 == 0:
                    import gc
                    gc.collect()
        
        # Explicit cleanup
        pool.close()
        pool.join()
    
    # Final garbage collection
    import gc
    gc.collect()
    
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
    """Dynamically discover all player classes in Players/*player.py files"""
    player_classes = []
    player_files = glob.glob("Players/*player.py")
    
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
      # Helper function to determine batch size for a specific player
    def get_batch_size_for_player(player_name: str, num_games: int) -> int:
        """Determine optimal batch size based on player type and number of games"""
        # Monte Carlo players need smaller batches due to memory-intensive simulations
        # Each batch creates a fresh pool, so smaller is better for memory
        if 'monte' in player_name.lower() and 'carlo' in player_name.lower():
            print(f"  → Using small batch size (50) for Monte Carlo player to manage memory usage")
            return 50
        
        # Other players can use slightly larger batches but still conservative
        if num_games >= 500000:
            print(f"  → Using batch size (200) for large simulation")
            return 200
        elif num_games >= 100000:
            print(f"  → Using batch size (100) for medium simulation")
            return 100
        else:
            # For smaller runs, use even smaller batches to minimize overhead
            print(f"  → Using batch size (50) for small simulation")
            return 50
    
    compare_all_option = len(player_classes) + 1
    # Single player mode
    if bot_choice < compare_all_option:
        player_info = player_classes[bot_choice - 1]
        player_instance = player_info['class']()
        
        # Determine batch size for this specific player
        batch_size = get_batch_size_for_player(player_instance.name, num_games)
        
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
            analyze_and_save_results([results], "single-bot")    # Compare all players
    else:
        print(f"\nStarting comparison mode - running {num_games} games with each bot...")
        print("This will run background simulations only (no GUI for comparison mode)")
        
        all_results = []
        for i, player_info in enumerate(player_classes, 1):
            player_instance = player_info['class']()
            
            # Determine batch size for this specific player
            batch_size = get_batch_size_for_player(player_instance.name, num_games)
            
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