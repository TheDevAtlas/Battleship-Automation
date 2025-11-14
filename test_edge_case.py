import asyncio
import webbrowser
from pathlib import Path
from aiohttp import web
import aiohttp

from main import BattleshipServer
from battleship import BattleshipGame, HuntAndTargetAgent


class EdgeCaseTestGame(BattleshipGame):
    """Custom game with predefined ship positions for edge case testing."""
    
    def setup_board(self):
        """Place ships in specific positions for the edge case test."""
        # Clear the board
        self.board = [[0 for _ in range(10)] for _ in range(10)]
        self.ship_positions.clear()
        self.ship_tiles = []
        
        # Place a 4-tile horizontal ship in the middle (row 4, columns 3-6)
        ship1_positions = set()
        for col in range(3, 7):  # 4 tiles: columns 3, 4, 5, 6
            self.ship_positions.add((4, col))
            ship1_positions.add((4, col))
        self.ship_tiles.append(ship1_positions)
        
        # Place a 3-tile vertical ship perpendicular UNDER the first hit
        # First hit will be at (4, 3), so place vertical ship at column 3, rows 5-7
        ship2_positions = set()
        for row in range(5, 8):  # 3 tiles: rows 5, 6, 7
            self.ship_positions.add((row, 3))
            ship2_positions.add((row, 3))
        self.ship_tiles.append(ship2_positions)
        
        # Update ships list to match our setup
        self.ships = [4, 3]


async def run_edge_case_test():
    """Run the edge case test with visual GUI."""
    # Create server
    server = BattleshipServer(use_visual=True)
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print(f"Server started at http://localhost:8080")
    
    # Open browser
    webbrowser.open('http://localhost:8080')
    print("Opening browser for edge case test...")
    await asyncio.sleep(2)  # Wait for browser to load
    
    # Create game with custom setup
    game = EdgeCaseTestGame()
    game.setup_board()
    
    # Create Hunt and Target agent
    agent = HuntAndTargetAgent()
    
    # Force the first move to be the top-left of the horizontal ship (4, 3)
    # This is where the two ships intersect
    agent.available_positions = [(4, 3)] + [(r, c) for r in range(10) for c in range(10) if (r, c) != (4, 3)]
    agent.position_index = 0
    
    # Send initial empty board
    await server.send_board_update(game.get_board_state())
    await asyncio.sleep(2)
    
    print("\n" + "=" * 50)
    print("EDGE CASE TEST: Two Perpendicular Ships")
    print("=" * 50)
    print("Ship 1: 4-tile horizontal at row 4, columns 3-6")
    print("Ship 2: 3-tile vertical at column 3, rows 5-7")
    print("First hit will be at intersection: (4, 3)")
    print("=" * 50 + "\n")
    
    move_count = 0
    
    while not game.is_game_over():
        move_count += 1
        
        # Show targeting highlights if agent has targets
        if agent.targets:
            await server.send_targeting_highlights(list(agent.targets))
            await asyncio.sleep(0.5)
        
        # Get move and make guess
        row, col = agent.get_move()
        result, sunk_ship = game.make_guess(row, col)
        
        print(f"Move {move_count}: ({row}, {col}) -> {result.upper()}" + 
              (f" - SHIP SUNK!" if sunk_ship else ""))
        
        # Update agent
        agent.update(row, col, result, sunk_ship)
        
        # Send board update
        board_state = game.get_board_state_with_sunk()
        await server.send_board_update(board_state)
        
        # If ship sunk, clear highlights
        if sunk_ship:
            await server.send_clear_highlights()
            await asyncio.sleep(0.5)
        
        await asyncio.sleep(1.0)  # 1 second delay for video mode
    
    # Game over
    await server.send_game_over(game.moves)
    
    print("\n" + "=" * 50)
    print(f"TEST COMPLETE - Total moves: {game.moves}")
    print("=" * 50 + "\n")
    
    # Keep server running for a bit to see final state
    await asyncio.sleep(5)
    
    await runner.cleanup()


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("EDGE CASE TEST - Hunt and Target")
    print("Testing perpendicular ships edge case")
    print("=" * 50 + "\n")
    
    try:
        asyncio.run(run_edge_case_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
