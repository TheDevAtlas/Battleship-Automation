"""
Quick speed test to demonstrate parallel execution performance.
This script runs a small batch of games and times the execution.
"""
import asyncio
import time
from battleship import BattleshipGame, ProbabilityAgent

async def run_single_game():
    """Run a single game."""
    game = BattleshipGame()
    game.setup_board()
    agent = ProbabilityAgent()
    
    while not game.is_game_over():
        row, col = agent.get_move()
        result, sunk_ship = game.make_guess(row, col)
        agent.update(row, col, result, sunk_ship)
    
    return game.moves

async def test_sequential(num_games=10):
    """Test sequential execution."""
    print(f"\n🐌 Sequential Execution - {num_games} games")
    start = time.time()
    
    results = []
    for i in range(num_games):
        moves = await run_single_game()
        results.append(moves)
        print(f"  Game {i+1}/{num_games} completed: {moves} moves")
    
    elapsed = time.time() - start
    print(f"  ✓ Total time: {elapsed:.2f} seconds")
    print(f"  ✓ Average: {sum(results)/len(results):.1f} moves")
    return elapsed

async def test_parallel(num_games=10):
    """Test parallel execution."""
    print(f"\n⚡ Parallel Execution - {num_games} games")
    start = time.time()
    
    # Run all games in parallel
    tasks = [run_single_game() for _ in range(num_games)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    for i, moves in enumerate(results, 1):
        print(f"  Game {i}/{num_games} completed: {moves} moves")
    
    print(f"  ✓ Total time: {elapsed:.2f} seconds")
    print(f"  ✓ Average: {sum(results)/len(results):.1f} moves")
    return elapsed

async def main():
    print("=" * 60)
    print("PARALLEL EXECUTION SPEED TEST")
    print("=" * 60)
    print("\nTesting with Probability Agent (computationally expensive)")
    print("Running 10 games each way...\n")
    
    # Test sequential
    seq_time = await test_sequential(10)
    
    # Test parallel
    par_time = await test_parallel(10)
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequential time: {seq_time:.2f} seconds")
    print(f"Parallel time:   {par_time:.2f} seconds")
    speedup = seq_time / par_time
    print(f"\n🚀 Speed improvement: {speedup:.1f}x faster!")
    print(f"⏱️  Time saved: {seq_time - par_time:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
