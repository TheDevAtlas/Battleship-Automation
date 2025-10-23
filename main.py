import tkinter as tk
from tkinter import messagebox
from game import Game
from random_player import RandomPlayer
import time

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

def run_simulation_games(num_games: int, show_gui: bool = False):
    """Run multiple games and return statistics"""
    print(f"\nRunning {num_games} games...")
    
    results = []
    player = RandomPlayer("Random AI")
    
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
    
    # Display results
    print(f"\n{'='*50}")
    print(f"BATTLESHIP SIMULATION RESULTS")
    print(f"{'='*50}")
    print(f"Total games played: {num_games}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Average time per game: {(end_time - start_time) / num_games:.3f} seconds")
    print(f"\nMOVE STATISTICS:")
    print(f"Best game (least moves): {best_game['moves']} moves")
    print(f"Worst game (most moves): {worst_game['moves']} moves")
    print(f"Average moves: {average_moves:.2f}")
    print(f"{'='*50}")
    
    return results

def get_user_choice():
    """Get user preferences for running the simulation"""
    print("BATTLESHIP AUTOMATION")
    print("=" * 30)
    
    # Ask about number of games first
    while True:
        try:
            num_games = int(input("How many games do you want to run? "))
            if num_games > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Ask about GUI
    while True:
        gui_choice = input("Do you want to run with GUI? (y/n): ").lower().strip()
        if gui_choice in ['y', 'yes']:
            return True, num_games
        elif gui_choice in ['n', 'no']:
            return False, num_games
        else:
            print("Please enter 'y' for yes or 'n' for no")

def main():
    show_gui, num_games = get_user_choice()
    
    if show_gui:
        print(f"\nStarting GUI mode for {num_games} games...")
        gui = BattleshipGUI(num_games)
        gui.run()
    else:
        print(f"\nStarting background simulation mode...")
        run_simulation_games(num_games, show_gui=False)

if __name__ == "__main__":
    main()