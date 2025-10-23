import tkinter as tk
from tkinter import messagebox
from game import Game
from random_player import RandomPlayer
from random_target_player import RandomTargetPlayer
from spaced_random_player import SpacedRandomPlayer
from analysis import analyze_and_save_results
import time
import statistics

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
        self.player = RandomPlayer("Random AI")
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

def run_simulation_games(num_games: int, player, show_gui: bool = False):
    """Run multiple games and return statistics"""
    print(f"\nRunning {num_games} games with {player.name}...")
    
    results = []
    
    start_time = time.time()
    
    for game_num in range(num_games):
        game = Game(player)
        result = game.play_game(show_moves=False)
        results.append(result)
        
        # Show progress for large simulations
        if num_games > 10 and (game_num + 1) % max(1, num_games // 10) == 0:
            progress = ((game_num + 1) / num_games) * 100
            print(f"Progress: {progress:.1f}% ({game_num + 1}/{num_games} games)")
    
    end_time = time.time()
    
    # Calculate statistics
    move_counts = [result['moves'] for result in results]
    best_game = min(results, key=lambda x: x['moves'])
    worst_game = max(results, key=lambda x: x['moves'])
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
    print(f"Best game (least moves): {best_game['moves']} moves")
    print(f"Worst game (most moves): {worst_game['moves']} moves")
    print(f"Average moves: {average_moves:.2f}")
    print(f"Median moves: {median_moves:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"{'='*60}")
    
    return {
        'results': results,
        'stats': {
            'player_name': player.name,
            'num_games': num_games,
            'total_time': end_time - start_time,
            'avg_time_per_game': (end_time - start_time) / num_games,
            'best_moves': best_game['moves'],
            'worst_moves': worst_game['moves'],
            'average_moves': average_moves,
            'median_moves': median_moves,
            'std_dev': std_dev,
            'move_counts': move_counts
        }
    }





def get_bot_choice():
    """Get user choice for bot type"""
    print("\nAVAILABLE BOT TYPES:")
    print("1. Random Player - Makes completely random moves")
    print("2. Random Target Player - Random hunting, but targets ships when found")
    print("3. Spaced Random Player - Uses spacing strategy to reduce guesses")
    print("4. Compare Random vs Random Target - Run first two bots and compare")
    print("5. Compare All Three Bots - Run all bots and compare performance")
    
    while True:
        try:
            choice = int(input("\nSelect bot type (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                return choice
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        except ValueError:
            print("Please enter a valid number")

def get_user_choice():
    """Get user preferences for running the simulation"""
    print("BATTLESHIP AUTOMATION")
    print("=" * 30)
    
    # Get bot choice first
    bot_choice = get_bot_choice()
    
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
    if bot_choice in [1, 2, 3]:  # Single bot modes only
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
    bot_choice, show_gui, num_games = get_user_choice()
    
    # Create bot instances
    random_player = RandomPlayer("Random Player")
    target_player = RandomTargetPlayer("Random Target Player")
    spaced_player = SpacedRandomPlayer("Spaced Random Player")
    
    if bot_choice == 1:  # Random Player only
        if show_gui:
            print(f"\nStarting GUI mode for {num_games} games with Random Player...")
            gui = BattleshipGUI(num_games)
            gui.player = random_player
            gui.run()
        else:
            print(f"\nStarting simulation mode with Random Player...")
            results = run_simulation_games(num_games, random_player, show_gui=False)
            
    elif bot_choice == 2:  # Random Target Player only
        if show_gui:
            print(f"\nStarting GUI mode for {num_games} games with Random Target Player...")
            gui = BattleshipGUI(num_games)
            gui.player = target_player
            gui.run()
        else:
            print(f"\nStarting simulation mode with Random Target Player...")
            results = run_simulation_games(num_games, target_player, show_gui=False)
    
    elif bot_choice == 3:  # Spaced Random Player only
        if show_gui:
            print(f"\nStarting GUI mode for {num_games} games with Spaced Random Player...")
            gui = BattleshipGUI(num_games)
            gui.player = spaced_player
            gui.run()
        else:
            print(f"\nStarting simulation mode with Spaced Random Player...")
            results = run_simulation_games(num_games, spaced_player, show_gui=False)
            
    elif bot_choice == 4:  # Compare Random vs Random Target
        print(f"\nStarting comparison mode - running {num_games} games with each bot...")
        print("This will run background simulations only (no GUI for comparison mode)")
        
        # Run Random Player
        print(f"\n{'='*60}")
        print("PHASE 1: Testing Random Player")
        print(f"{'='*60}")
        results1 = run_simulation_games(num_games, random_player, show_gui=False)
        
        # Run Random Target Player
        print(f"\n{'='*60}")
        print("PHASE 2: Testing Random Target Player")
        print(f"{'='*60}")
        results2 = run_simulation_games(num_games, target_player, show_gui=False)
        
        # Compare results
        compare_bot_performance(results1, results2)
        
        # Additional detailed comparison
        print(f"\n{'='*60}")
        print("DETAILED COMPARISON")  
        print(f"{'='*60}")
        
        # Move distribution comparison
        moves1 = results1['stats']['move_counts']
        moves2 = results2['stats']['move_counts']
        
        print(f"\nMove Distribution Analysis:")
        print(f"Random Player:")
        print(f"  Min: {min(moves1)}, Max: {max(moves1)}, Range: {max(moves1) - min(moves1)}")
        print(f"Random Target Player:")
        print(f"  Min: {min(moves2)}, Max: {max(moves2)}, Range: {max(moves2) - min(moves2)}")
        
        # Games where each bot performed better
        better_games = 0
        for i in range(len(moves1)):
            if moves2[i] < moves1[i]:
                better_games += 1
        
        print(f"\nGame-by-game comparison:")
        print(f"Random Target Player won {better_games}/{num_games} games ({(better_games/num_games)*100:.1f}%)")
        print(f"Random Player won {num_games - better_games}/{num_games} games ({((num_games - better_games)/num_games)*100:.1f}%)")
        
        # Efficiency ranges
        print(f"\nEfficiency Analysis:")
        efficient_games1 = len([m for m in moves1 if m <= 50])  # Games finished in 50 or fewer moves
        efficient_games2 = len([m for m in moves2 if m <= 50])
        
        print(f"Games finished in ≤50 moves:")
        print(f"  Random Player: {efficient_games1}/{num_games} ({(efficient_games1/num_games)*100:.1f}%)")
        print(f"  Random Target Player: {efficient_games2}/{num_games} ({(efficient_games2/num_games)*100:.1f}%)")
        
        poor_games1 = len([m for m in moves1 if m >= 80])  # Games that took 80+ moves
        poor_games2 = len([m for m in moves2 if m >= 80])
        
        print(f"Games requiring ≥80 moves:")
        print(f"  Random Player: {poor_games1}/{num_games} ({(poor_games1/num_games)*100:.1f}%)")
        print(f"  Random Target Player: {poor_games2}/{num_games} ({(poor_games2/num_games)*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print("CONCLUSION")
        print(f"{'='*60}")
        
        avg_improvement = ((results1['stats']['average_moves'] - results2['stats']['average_moves']) / results1['stats']['average_moves']) * 100
        if avg_improvement > 5:
            print(f"Random Target Player shows significant improvement ({avg_improvement:.1f}% better)")
            print("The targeting strategy effectively reduces average game length.")
        elif avg_improvement > 0:
            print(f"Random Target Player shows modest improvement ({avg_improvement:.1f}% better)")
            print("The targeting strategy provides some benefit over pure random moves.")
        elif avg_improvement < -5:
            print(f"Random Player surprisingly performs better ({abs(avg_improvement):.1f}% better)")
            print("The random strategy may be avoiding some inefficiencies in the targeting approach.")
        else:
            print("Both strategies perform similarly on average.")
            print("The sample size may need to be increased to see significant differences.")
    
    elif bot_choice == 5:  # Compare all three
        print(f"\nStarting three-way comparison - running {num_games} games with each bot...")
        print("This will run background simulations only (no GUI for comparison mode)")
        
        # Run Random Player
        print(f"\n{'='*60}")
        print("PHASE 1: Testing Random Player")
        print(f"{'='*60}")
        results1 = run_simulation_games(num_games, random_player, show_gui=False)
        
        # Run Random Target Player
        print(f"\n{'='*60}")
        print("PHASE 2: Testing Random Target Player")
        print(f"{'='*60}")
        results2 = run_simulation_games(num_games, target_player, show_gui=False)
        
        # Run Spaced Random Player
        print(f"\n{'='*60}")
        print("PHASE 3: Testing Spaced Random Player")
        print(f"{'='*60}")
        results3 = run_simulation_games(num_games, spaced_player, show_gui=False)
        
        # Compare all three
        wins = compare_three_bots_performance(results1, results2, results3)
        
        # Three-bot specific analysis
        moves1 = results1['stats']['move_counts']
        moves2 = results2['stats']['move_counts']
        moves3 = results3['stats']['move_counts']
        
        print(f"\n{'='*80}")
        print("THREE-WAY DETAILED COMPARISON")
        print(f"{'='*80}")
        
        print(f"\nMove Distribution Analysis:")
        print(f"Random Player:")
        print(f"  Min: {min(moves1)}, Max: {max(moves1)}, Range: {max(moves1) - min(moves1)}")
        print(f"Random Target Player:")
        print(f"  Min: {min(moves2)}, Max: {max(moves2)}, Range: {max(moves2) - min(moves2)}")
        print(f"Spaced Random Player:")
        print(f"  Min: {min(moves3)}, Max: {max(moves3)}, Range: {max(moves3) - min(moves3)}")
        
        # Head-to-head win rates
        wins_rt_vs_r = sum(1 for i in range(len(moves1)) if moves2[i] < moves1[i])
        wins_sr_vs_r = sum(1 for i in range(len(moves1)) if moves3[i] < moves1[i])
        wins_sr_vs_rt = sum(1 for i in range(len(moves2)) if moves3[i] < moves2[i])
        
        print(f"\nHead-to-Head Game Wins:")
        print(f"Random Target vs Random: {wins_rt_vs_r}/{num_games} ({(wins_rt_vs_r/num_games)*100:.1f}%)")
        print(f"Spaced Random vs Random: {wins_sr_vs_r}/{num_games} ({(wins_sr_vs_r/num_games)*100:.1f}%)")
        print(f"Spaced Random vs Random Target: {wins_sr_vs_rt}/{num_games} ({(wins_sr_vs_rt/num_games)*100:.1f}%)")
        
        # Efficiency analysis
        print(f"\nEfficiency Analysis (Games ≤50 moves):")
        efficient_games1 = len([m for m in moves1 if m <= 50])
        efficient_games2 = len([m for m in moves2 if m <= 50])
        efficient_games3 = len([m for m in moves3 if m <= 50])
        
        print(f"  Random Player: {efficient_games1}/{num_games} ({(efficient_games1/num_games)*100:.1f}%)")
        print(f"  Random Target Player: {efficient_games2}/{num_games} ({(efficient_games2/num_games)*100:.1f}%)")
        print(f"  Spaced Random Player: {efficient_games3}/{num_games} ({(efficient_games3/num_games)*100:.1f}%)")
        
        print(f"\nPoor Performance Analysis (Games ≥80 moves):")
        poor_games1 = len([m for m in moves1 if m >= 80])
        poor_games2 = len([m for m in moves2 if m >= 80])
        poor_games3 = len([m for m in moves3 if m >= 80])
        
        print(f"  Random Player: {poor_games1}/{num_games} ({(poor_games1/num_games)*100:.1f}%)")
        print(f"  Random Target Player: {poor_games2}/{num_games} ({(poor_games2/num_games)*100:.1f}%)")
        print(f"  Spaced Random Player: {poor_games3}/{num_games} ({(poor_games3/num_games)*100:.1f}%)")
        
        print(f"\n{'='*80}")
        print("STRATEGY INSIGHTS")
        print(f"{'='*80}")
        
        print(f"\nSPACED RANDOM STRATEGY ANALYSIS:")
        print(f"The Spaced Random Player uses a grid-spacing strategy where:")
        print(f"- For remaining ships of size N, it checks every Nth square")
        print(f"- This guarantees finding any ship while reducing search space")
        print(f"- Size 5 ship: 80% search space reduction")
        print(f"- Size 4 ship: 75% search space reduction") 
        print(f"- Size 3 ship: 67% search space reduction")
        print(f"- Size 2 ship: 50% search space reduction")
        print(f"\nTheoretical maximum guesses needed (spaced strategy):")
        print(f"- Size 5: 20 guesses (every 5th square)")
        print(f"- Size 4: 25 guesses (every 4th square)")  
        print(f"- Size 3: 34 guesses (every 3rd square)")
        print(f"- Size 2: 50 guesses (every 2nd square)")
        print(f"- Total theoretical minimum: ~35-40 moves with perfect spacing")
        
        print(f"\n{'='*80}")
        print("FINAL CONCLUSIONS")
        print(f"{'='*80}")
        
        # Compare improvements
        rt_improvement = ((results1['stats']['average_moves'] - results2['stats']['average_moves']) / results1['stats']['average_moves']) * 100
        sr_improvement = ((results1['stats']['average_moves'] - results3['stats']['average_moves']) / results1['stats']['average_moves']) * 100
        sr_vs_rt_improvement = ((results2['stats']['average_moves'] - results3['stats']['average_moves']) / results2['stats']['average_moves']) * 100
        
        print(f"\nPerformance Improvements vs Random Player:")
        print(f"  Random Target Player: {rt_improvement:.1f}% improvement")
        print(f"  Spaced Random Player: {sr_improvement:.1f}% improvement")
        print(f"\nSpaced Random vs Random Target:")
        print(f"  Additional improvement: {sr_vs_rt_improvement:.1f}%")
        
        if sr_improvement > rt_improvement:
            print(f"\n✓ Spaced Random strategy shows superior performance")
            print(f"  The spacing approach provides {abs(sr_improvement - rt_improvement):.1f}% better results than targeting alone")
        elif rt_improvement > sr_improvement:
            print(f"\n✓ Random Target strategy shows superior performance")
            print(f"  The targeting approach provides {abs(rt_improvement - sr_improvement):.1f}% better results than spacing alone")
        else:
            print(f"\n≈ Both advanced strategies perform similarly")
            print(f"  Sample size may need to be increased to distinguish performance")
        


if __name__ == "__main__":
    main()