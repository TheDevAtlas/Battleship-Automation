"""
Quick test script to compare original vs optimized Monte Carlo player
"""
import time
import importlib.util
from game import Game

# Load both player modules
print("Loading player modules...")
spec5 = importlib.util.spec_from_file_location("mc5", "5_monte_carlo_player.py")
mod5 = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(mod5)

spec6 = importlib.util.spec_from_file_location("mc6", "6_monte_carlo_player.py")
mod6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(mod6)

# Test parameters
NUM_GAMES = 5
NUM_SIMS = 300

print(f"\n{'='*60}")
print(f"MONTE CARLO PERFORMANCE COMPARISON")
print(f"{'='*60}")
print(f"Games per test: {NUM_GAMES}")
print(f"Simulations per move: {NUM_SIMS}")
print(f"{'='*60}\n")

# Test original player
print("Testing ORIGINAL Monte Carlo Player...")
print("-" * 60)
player5 = mod5.MonteCarloPlayer(num_simulations=NUM_SIMS)
start_time = time.time()

moves5 = []
for i in range(NUM_GAMES):
    game = Game(player5)
    result = game.play_game(show_moves=False)
    moves5.append(result['moves'])
    print(f"Game {i+1}/{NUM_GAMES}: {result['moves']} moves")

end_time = time.time()
time5 = end_time - start_time
avg_moves5 = sum(moves5) / len(moves5)

print(f"\nTotal time: {time5:.2f}s")
print(f"Time per game: {time5/NUM_GAMES:.2f}s")
print(f"Average moves: {avg_moves5:.1f}")

# Test optimized player
print(f"\n{'='*60}")
print("Testing OPTIMIZED Monte Carlo Player...")
print("-" * 60)
player6 = mod6.OptimizedMonteCarloPlayer(num_simulations=NUM_SIMS)

# Warm up JIT compiler with one game first
print("Warming up JIT compiler (first game)...")
warmup_game = Game(player6)
warmup_result = warmup_game.play_game(show_moves=False)
print(f"Warmup game: {warmup_result['moves']} moves (compilation included)\n")

# Now run the real test
print("Running timed test (JIT already compiled)...")
start_time = time.time()

moves6 = []
for i in range(NUM_GAMES):
    game = Game(player6)
    result = game.play_game(show_moves=False)
    moves6.append(result['moves'])
    print(f"Game {i+1}/{NUM_GAMES}: {result['moves']} moves")

end_time = time.time()
time6 = end_time - start_time
avg_moves6 = sum(moves6) / len(moves6)

print(f"\nTotal time: {time6:.2f}s")
print(f"Time per game: {time6/NUM_GAMES:.2f}s")
print(f"Average moves: {avg_moves6:.1f}")

# Summary
print(f"\n{'='*60}")
print("PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"Original Player:  {time5:.2f}s total, {time5/NUM_GAMES:.2f}s per game")
print(f"Optimized Player: {time6:.2f}s total, {time6/NUM_GAMES:.2f}s per game")
print(f"\nSpeedup: {time5/time6:.1f}x FASTER! 🚀")
print(f"\nMove quality comparison:")
print(f"Original avg:  {avg_moves5:.1f} moves")
print(f"Optimized avg: {avg_moves6:.1f} moves")
print(f"Difference: {abs(avg_moves5 - avg_moves6):.1f} moves (should be similar)")
print(f"{'='*60}")
