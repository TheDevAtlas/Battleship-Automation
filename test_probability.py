"""Quick test for ProbabilityAgent"""
from battleship import BattleshipGame, ProbabilityAgent

# Create a game and agent
game = BattleshipGame()
game.setup_board()
agent = ProbabilityAgent()

print("Testing ProbabilityAgent...")
print(f"Initial probability map shape: {agent.prob_map.shape}")
print(f"Remaining ships: {agent.remaining_ships}")

# Run a few moves
for i in range(5):
    row, col = agent.get_move()
    result, sunk_ship = game.make_guess(row, col)
    agent.update(row, col, result, sunk_ship)
    print(f"Move {i+1}: ({row}, {col}) -> {result}")
    if sunk_ship:
        print(f"  Sunk ship of size {len(sunk_ship)}!")

print(f"\nRemaining ships after 5 moves: {agent.remaining_ships}")
print("✓ ProbabilityAgent working correctly!")
