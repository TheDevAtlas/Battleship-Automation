"""Quick test script to compare Random vs Hunt and Target agents."""
import asyncio
from battleship import BattleshipGame, RandomAgent, HuntAndTargetAgent

async def test_agent(agent_class, agent_name, num_games=100):
    """Test an agent and return statistics."""
    moves_list = []
    
    for i in range(num_games):
        game = BattleshipGame()
        game.setup_board()
        agent = agent_class()
        
        while not game.is_game_over():
            row, col = agent.get_move()
            result, sunk_ship = game.make_guess(row, col)
            agent.update(row, col, result, sunk_ship)
        
        moves_list.append(game.moves)
    
    avg_moves = sum(moves_list) / len(moves_list)
    min_moves = min(moves_list)
    max_moves = max(moves_list)
    
    print(f"\n{agent_name} Agent Results ({num_games} games):")
    print(f"  Average moves: {avg_moves:.2f}")
    print(f"  Best game: {min_moves} moves")
    print(f"  Worst game: {max_moves} moves")
    
    return avg_moves

async def main():
    print("=" * 60)
    print("BATTLESHIP AGENT COMPARISON")
    print("=" * 60)
    
    # Test Random Agent
    random_avg = await test_agent(RandomAgent, "Random", 100)
    
    # Test Hunt and Target Agent
    hunt_avg = await test_agent(HuntAndTargetAgent, "Hunt and Target", 100)
    
    # Compare
    improvement = ((random_avg - hunt_avg) / random_avg) * 100
    print("\n" + "=" * 60)
    print(f"Hunt and Target is {improvement:.1f}% more efficient than Random!")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(main())
