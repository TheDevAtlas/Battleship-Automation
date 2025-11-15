import asyncio
import json
import os
import webbrowser
from pathlib import Path
from aiohttp import web
import aiohttp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from battleship import BattleshipGame, RandomAgent, HuntAndTargetAgent, ParityHuntAgent, ProbabilityAgent, GameStatistics


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
    
    async def send_targeting_highlights(self, target_squares: list):
        """Send targeting highlights to show suspected squares with yellow gradient."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send_json({
                'type': 'targeting_highlights',
                'squares': target_squares
            })
    
    async def send_clear_highlights(self):
        """Send message to clear targeting highlights."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send_json({
                'type': 'clear_highlights'
            })
    
    async def send_eliminated_squares(self, eliminated_squares: list):
        """Send eliminated squares to show in very light blue."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send_json({
                'type': 'eliminated_squares',
                'squares': eliminated_squares
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
        self.use_video_mode = False  # New: For video enhancement features
        self.show_targeting_highlights = False  # New: Show yellow gradient on suspected squares        self.multi_agent_mode = False  # Track if running multiple agents
        self.agent_stats = {}  # Store stats per agent
        self.selected_agents = [1, 2, 3, 4]  # Which agents to run in multi-agent mode
    
    def get_user_input(self) -> tuple:
        """Get configuration from user."""
        print("\n" + "=" * 50)
        print("BATTLESHIP AUTOMATION")
        print("=" * 50)
        
        # Question 0: Single or multi-agent mode
        print("\n0) Run mode:")
        print("   [1] Single Agent - Run one agent type")
        print("   [2] Multi-Agent Comparison - Run all agents and compare")
        run_mode = input("   Enter choice (1): ").strip() or "1"
        if run_mode == "2":
            self.multi_agent_mode = True            # Multi-agent mode
            print("\n1) Which agents to run?")
            print("   [1] Random")
            print("   [2] Hunt and Target")
            print("   [3] Parity Hunt")
            print("   [4] Probability (Heat Map)")
            print("   Enter agent numbers separated by commas (e.g., 1,2 or 1,2,3,4)")
            agent_choice = input("   Enter choices (default all: 1,2,3,4): ").strip() or "1,2,3,4"
            
            # Parse and validate agent choices
            try:
                selected_agents = [int(x.strip()) for x in agent_choice.split(',')]
                selected_agents = [x for x in selected_agents if x in [1, 2, 3, 4]]
                if not selected_agents:
                    print("   Invalid input, using all agents")
                    selected_agents = [1, 2, 3, 4]
            except ValueError:
                print("   Invalid input, using all agents")
                selected_agents = [1, 2, 3, 4]
            
            self.selected_agents = selected_agents
            
            print("\n2) How many games to run per agent?")
            num_games = input("   Enter number (default 100): ").strip() or "100"
            try:
                num_games = int(num_games)
            except ValueError:
                print("   Invalid input, using default of 100")
                num_games = 100
            
            print("\n3) Should we use the visual GUI?")
            print("   [y] Yes - Open browser and show games")
            print("   [n] No - Run in background (recommended for multi-agent)")
            visual_choice = input("   Enter choice (y/n): ").strip().lower()
            use_visual = visual_choice in ['y', 'yes']
              # Multi-agent always runs in background for speed
            if use_visual:
                print("   NOTE: Visual mode will slow down multi-agent runs significantly")
            
            return "multi", num_games, use_visual, False, False
        
        # Single agent mode (original behavior)
        self.multi_agent_mode = False
        # Question 1: Brain type
        print("\n1) What brain are we using?")
        print("   [1] Random")
        print("   [2] Hunt and Target")
        print("   [3] Parity Hunt (eliminates unnecessary squares)")
        print("   [4] Probability (Heat Map)")
        brain_choice = input("   Enter choice (1): ").strip() or "1"
          # Question 2: Number of games
        print("\n2) How many games to run?")
        num_games = input("   Enter number (default 10): ").strip() or "10"
        try:
            num_games = int(num_games)
        except ValueError:
            print("   Invalid input, using default of 10")
            num_games = 10
        
        # Question 3: Visual GUI
        print("\n3) Should we use the visual GUI?")
        print("   [y] Yes - Open browser and show games")
        print("   [n] No - Run in background")
        visual_choice = input("   Enter choice (y/n): ").strip().lower()
        use_visual = visual_choice in ['y', 'yes', '']          # NEW: Question 4: Video mode enhancements (only if GUI + Hunt & Target)
        use_video_mode = False
        show_targeting_highlights = False
        if use_visual and brain_choice in ["2", "3", "4"]:
            print("\n4) Video mode enhancements for Hunt & Target / Parity Hunt / Probability?")
            print("   [y] Yes - 1 second delays + targeting highlights")
            print("   [n] No - Super speed mode")
            video_choice = input("   Enter choice (y/n): ").strip().lower()
            use_video_mode = video_choice in ['y', 'yes']
            
            if use_video_mode:
                self.delay_between_moves = 1.0  # 1 second increments
                show_targeting_highlights = True
                print("   ✓ Enabled: 1 second delays")
                print("   ✓ Enabled: Yellow gradient highlights on suspected squares")
                if brain_choice == "3":
                    print("   ✓ Enabled: Light blue eliminated squares (parity pattern)")
        print("\n" + "=" * 50)
        print(f"Starting {num_games} game(s) with {'visual' if use_visual else 'background'} mode...")
        if use_visual and brain_choice in ["2", "3", "4"] and use_video_mode:
            print("Video enhancement mode: ON")
        print("=" * 50 + "\n")
        
        return brain_choice, num_games, use_visual, use_video_mode, show_targeting_highlights
    async def run_single_game(self, game_number: int, total_games: int) -> int:
        """Run a single game and return the number of moves."""
        game = BattleshipGame()
        game.setup_board()        # Select agent based on brain choice
        if self.brain_choice == "2":
            agent = HuntAndTargetAgent()
        elif self.brain_choice == "3":
            agent = ParityHuntAgent()
        elif self.brain_choice == "4":
            agent = ProbabilityAgent()
        else:
            agent = RandomAgent()
        
        # Send initial empty board if visual
        if self.use_visual and self.server:
            await self.server.send_board_update(game.get_board_state())
            
            # For Parity Hunt, send initial eliminated squares
            if self.brain_choice == "3" and isinstance(agent, ParityHuntAgent):
                eliminated = agent.get_eliminated_squares()
                await self.server.send_eliminated_squares(eliminated)
        
        while not game.is_game_over():
            # NEW: Send targeting highlights if in video mode with hunt & target
            if (self.use_visual and self.server and self.show_targeting_highlights and 
                self.brain_choice == "2" and isinstance(agent, HuntAndTargetAgent) and agent.targets):
                # Show suspected squares with yellow gradient
                await self.server.send_targeting_highlights(list(agent.targets))
                await asyncio.sleep(0.5)  # Brief pause to show highlights
            
            # NEW: Send targeting highlights AND eliminated squares for Parity Hunt
            if (self.use_visual and self.server and self.show_targeting_highlights and 
                self.brain_choice == "3" and isinstance(agent, ParityHuntAgent)):
                # Show suspected squares with yellow gradient
                if agent.targets:
                    await self.server.send_targeting_highlights(list(agent.targets))
                # Show eliminated squares with light blue
                eliminated = agent.get_eliminated_squares()
                await self.server.send_eliminated_squares(eliminated)
                await asyncio.sleep(0.5)  # Brief pause to show highlights
            row, col = agent.get_move()
            result, sunk_ship = game.make_guess(row, col)
            agent.update(row, col, result, sunk_ship)
            
            # Send update to visual if enabled
            if self.use_visual and self.server:
                # Get board state with sunk ships marked as state 3
                board_state = game.get_board_state_with_sunk()
                await self.server.send_board_update(board_state)
                
                # If a ship was sunk, clear targeting highlights
                if sunk_ship and self.show_targeting_highlights:
                    await self.server.send_clear_highlights()
                    await asyncio.sleep(0.2)  # Brief pause after clearing highlights
                
                # Update eliminated squares for Parity Hunt (in case parity changed)
                if self.brain_choice == "3" and isinstance(agent, ParityHuntAgent):
                    eliminated = agent.get_eliminated_squares()
                    await self.server.send_eliminated_squares(eliminated)
                
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
        # Print statistics immediately after games complete
        print("\n")  # Add spacing after progress bar
        self.stats.print_summary()
    
    async def run_multi_agent_comparison(self, num_games: int, use_visual: bool):
        """Run all agents back-to-back and compare their statistics."""
        all_agents_config = [
            ("1", "Random", RandomAgent),
            ("2", "Hunt and Target", HuntAndTargetAgent),
            ("3", "Parity Hunt", ParityHuntAgent),
            ("4", "Probability", ProbabilityAgent)
        ]
        
        # Filter agents based on user selection
        agents_config = [config for config in all_agents_config if int(config[0]) in self.selected_agents]
        
        # Store results for each agent
        for choice, name, agent_class in agents_config:
            print("\n" + "=" * 50)
            print(f"Running {name} Agent...")
            print("=" * 50)
            
            # Reset stats for this agent
            self.brain_choice = choice
            self.stats = GameStatistics()
            
            # Run games for this agent
            if use_visual:
                await self.start_server_and_run(num_games)
            else:
                await self.run_games(num_games)
                print("\n")
                self.stats.print_summary()
            
            # Store the stats
            self.agent_stats[name] = {
                'stats': self.stats,
                'moves': self.stats.game_moves.copy()
            }
        
        # Print comparison summary
        self.print_multi_agent_summary()
        
        # Generate visualization
        self.visualize_multi_agent_results()
    
    def print_multi_agent_summary(self):
        """Print comparison summary for all agents."""
        print("\n" + "=" * 70)
        print("MULTI-AGENT COMPARISON SUMMARY")
        print("=" * 70)
        
        for agent_name, data in self.agent_stats.items():
            stats = data['stats']
            summary = stats.get_summary()
            print(f"\n{agent_name}:")
            print(f"  Average Moves: {summary['average_moves']:.2f}")
            print(f"  Best Game: {summary['best_game']}")
            print(f"  Worst Game: {summary['worst_game']}")
            print(f"  Median Moves: {summary['median_moves']:.1f}")
        
        print("\n" + "=" * 70)
    
    def visualize_multi_agent_results(self):
        """Create a Seaborn line graph visualization of agent performance."""
        # Set dark mode style
        plt.style.use('dark_background')
        sns.set_palette("husl")
          # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define aesthetic colors for dark mode
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD700']  # Added golden/yellow for Probability
        
        # Process data for each agent
        for idx, (agent_name, data) in enumerate(self.agent_stats.items()):
            moves = data['moves']
            
            # Count frequency of each move count
            move_counter = Counter(moves)
            
            # Sort by move count (x-axis)
            x_values = sorted(move_counter.keys())
            y_values = [move_counter[x] for x in x_values]
            
            # Plot line
            ax.plot(x_values, y_values, marker='o', linewidth=2.5, 
                   markersize=6, label=agent_name, color=colors[idx],
                   alpha=0.8)
        
        # Styling
        ax.set_xlabel('Number of Moves to Finish Game', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Games Finished', fontsize=14, fontweight='bold')
        ax.set_title('Battleship Agent Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Legend
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure
        output_path = Path(__file__).parent / 'agent_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"\n📊 Visualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
        print("✓ Visualization complete!")
    
    def run(self):
        """Main entry point."""
        self.brain_choice, num_games, self.use_visual, self.use_video_mode, self.show_targeting_highlights = self.get_user_input()
        
        try:
            if self.multi_agent_mode:
                # Multi-agent comparison mode
                asyncio.run(self.run_multi_agent_comparison(num_games, self.use_visual))
            elif self.use_visual:
                asyncio.run(self.start_server_and_run(num_games))
            else:
                asyncio.run(self.run_without_server(num_games))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            # Print statistics even when interrupted
            if self.multi_agent_mode and self.agent_stats:
                self.print_multi_agent_summary()
            else:
                self.stats.print_summary()


if __name__ == '__main__':
    runner = BattleshipRunner()
    runner.run()
