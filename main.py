import asyncio
import json
import os
import webbrowser
from pathlib import Path
from aiohttp import web
import aiohttp
from tqdm import tqdm

from battleship import BattleshipGame, RandomAgent, HuntAndTargetAgent, GameStatistics


class BattleshipServer:
    """Web server to host the HTML and handle WebSocket communication."""
    
    def __init__(self, use_visual: bool = True):
        self.app = web.Application()
        self.websocket = None
        self.use_visual = use_visual
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP and WebSocket routes."""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
    
    async def handle_index(self, request):
        """Serve the index.html file."""
        html_path = Path(__file__).parent / 'index.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    
    async def handle_websocket(self, request):
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket = ws
        print("WebSocket connected")
        
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                # Handle any messages from the client if needed
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
        
        print("WebSocket disconnected")
        self.websocket = None
        return ws
    
    async def send_board_update(self, board: list):
        """Send board update to the connected WebSocket client."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send_json({
                'type': 'board_update',
                'board': board
            })
    
    async def send_game_over(self, moves: int):
        """Send game over message to client."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send_json({
                'type': 'game_over',
                'moves': moves
            })


class BattleshipRunner:
    """Main runner for the battleship automation."""
    def __init__(self):
        self.stats = GameStatistics()
        self.server = None
        self.use_visual = False
        self.delay_between_moves = 0.025  # No delay - super speed!
        self.delay_between_games = 0.025  # No delay - super speed!
        self.brain_choice = "1"  # Track which agent to use
    
    def get_user_input(self) -> tuple:
        """Get configuration from user."""
        print("\n" + "=" * 50)
        print("BATTLESHIP AUTOMATION")
        print("=" * 50)
        
        # Question 1: Brain type
        print("\n1) What brain are we using?")
        print("   [1] Random")
        print("   [2] Hunt and Target")
        brain_choice = input("   Enter choice (1): ").strip() or "1"
        
        # Question 2: Number of games
        print("\n2) How many games to run?")
        num_games = input("   Enter number (default 10): ").strip() or "10"
        try:            num_games = int(num_games)
        except ValueError:
            print("   Invalid input, using default of 10")
            num_games = 10
        
        # Question 3: Visual GUI
        print("\n3) Should we use the visual GUI?")
        print("   [y] Yes - Open browser and show games")
        print("   [n] No - Run in background")
        visual_choice = input("   Enter choice (y/n): ").strip().lower()
        use_visual = visual_choice in ['y', 'yes', '']
        
        print("\n" + "=" * 50)
        print(f"Starting {num_games} game(s) with {'visual' if use_visual else 'background'} mode...")
        print("=" * 50 + "\n")
        
        return brain_choice, num_games, use_visual
    
    async def run_single_game(self, game_number: int, total_games: int) -> int:
        """Run a single game and return the number of moves."""
        game = BattleshipGame()
        game.setup_board()
        
        # Select agent based on brain choice
        if self.brain_choice == "2":
            agent = HuntAndTargetAgent()
        else:
            agent = RandomAgent()
        
        # Send initial empty board if visual
        if self.use_visual and self.server:
            await self.server.send_board_update(game.get_board_state())
        
        while not game.is_game_over():
            row, col = agent.get_move()
            result, sunk_ship = game.make_guess(row, col)
            agent.update(row, col, result, sunk_ship)
            
            # Send update to visual if enabled
            if self.use_visual and self.server:
                await self.server.send_board_update(game.get_board_state())
                await asyncio.sleep(self.delay_between_moves)
        
        # Send game over message
        if self.use_visual and self.server:
            await self.server.send_game_over(game.moves)
            await asyncio.sleep(self.delay_between_games)
        
        return game.moves
    
    async def run_games(self, num_games: int):
        """Run multiple games with a progress bar."""
        with tqdm(total=num_games, desc="Running games", unit="game", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for i in range(1, num_games + 1):
                moves = await self.run_single_game(i, num_games)
                self.stats.add_game(moves)
                pbar.update(1)
    
    async def start_server_and_run(self, num_games: int):
        """Start the web server and run games."""
        self.server = BattleshipServer(self.use_visual)
        runner = web.AppRunner(self.server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        print(f"Server started at http://localhost:8080")
        if self.use_visual:            # Open browser
            webbrowser.open('http://localhost:8080')
            print("Opening browser...")
            # Removed artificial delay - super speed mode!
        
        # Run the games
        await self.run_games(num_games)
        
        # Print statistics immediately after games complete
        print("\n")  # Add spacing after progress bar
        self.stats.print_summary()
        
        # Removed artificial delay - super speed mode!
        
        await runner.cleanup()
    
    async def run_without_server(self, num_games: int):
        """Run games without the visual server."""
        await self.run_games(num_games)
        # Print statistics immediately after games complete        print("\n")  # Add spacing after progress bar
        self.stats.print_summary()
    
    def run(self):
        """Main entry point."""
        self.brain_choice, num_games, self.use_visual = self.get_user_input()
        
        try:
            if self.use_visual:
                asyncio.run(self.start_server_and_run(num_games))
            else:
                asyncio.run(self.run_without_server(num_games))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            # Print statistics even when interrupted
            self.stats.print_summary()


if __name__ == '__main__':
    runner = BattleshipRunner()
    runner.run()
