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
import csv

from battleship import BattleshipGame, RandomAgent, HuntAndTargetAgent, ParityHuntAgent, ProbabilityAgent, GameStatistics


class BattleshipServer:
    """Web server to host the HTML and handle SSE communication."""
    
    def __init__(self, use_visual: bool = True):
        self.app = web.Application()
        self.sse_queue = asyncio.Queue()  # Queue for SSE messages
        self.use_visual = use_visual
        self.client_connected = False
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP and SSE routes."""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/events', self.handle_sse)
    
    async def handle_index(self, request):
        """Serve the index.html file."""
        html_path = Path(__file__).parent / 'index.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    
    async def handle_sse(self, request):
        """Handle Server-Sent Events connection."""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        await response.prepare(request)
        
        self.client_connected = True
        print("SSE client connected")
        
        try:
            while True:
                # Get message from queue
                message = await self.sse_queue.get()
                
                # Send as SSE format
                data = f"data: {json.dumps(message)}\n\n"
                await response.write(data.encode('utf-8'))
                await response.drain()
                
        except Exception as e:
            print(f"SSE connection closed: {e}")
        finally:
            self.client_connected = False
            print("SSE client disconnected")
        
        return response
    
    async def send_board_update(self, board: list):
        """Send board update to the connected SSE client."""
        await self.sse_queue.put({
            'type': 'board_update',
            'board': board
        })
    
    async def send_game_over(self, moves: int):
        """Send game over message to client."""
        await self.sse_queue.put({
            'type': 'game_over',
            'moves': moves
        })
    
    async def send_targeting_highlights(self, target_squares: list):
        """Send targeting highlights to show suspected squares with yellow gradient."""
        await self.sse_queue.put({
            'type': 'targeting_highlights',
            'squares': target_squares
        })
    
    async def send_clear_highlights(self):
        """Send message to clear targeting highlights."""
        await self.sse_queue.put({
            'type': 'clear_highlights'
        })
    
    async def send_eliminated_squares(self, eliminated_squares: list):
        """Send eliminated squares to show in very light blue."""
        await self.sse_queue.put({
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
        self.show_targeting_highlights = False  # New: Show yellow gradient on suspected squares        
        self.multi_agent_mode = False  # Track if running multiple agents
        self.agent_stats = {}  # Store stats per agent
        self.selected_agents = [1, 2, 3, 4]  # Which agents to run in multi-agent mode
        self.parallel_batch_size = 50  # How many games to run in parallel at once
    
    def get_agent_name(self, choice: str) -> str:
        """Get agent name from choice number."""
        agent_names = {
            "1": "Random",
            "2": "Hunt and Target",
            "3": "Parity Hunt",
            "4": "Probability"
        }
        return agent_names.get(choice, "Random")
    
    def get_csv_path(self, agent_name: str) -> Path:
        """Get the CSV file path for an agent."""
        # Clean filename: remove spaces and special characters
        clean_name = agent_name.replace(" ", "_").replace("(", "").replace(")", "")
        return Path(__file__).parent / f"{clean_name}.csv"
    
    def load_csv_data(self, agent_name: str) -> list:
        """Load existing game results from CSV file."""
        csv_path = self.get_csv_path(agent_name)
        moves = []
        
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header if exists
                    for row in reader:
                        if row and row[0].isdigit():
                            moves.append(int(row[0]))
                print(f"   Loaded {len(moves)} existing games from {csv_path.name}")
            except Exception as e:
                print(f"   Warning: Could not load CSV data: {e}")
        
        return moves
    
    def save_game_to_csv(self, agent_name: str, moves: int):
        """Save a single game result to CSV file."""
        csv_path = self.get_csv_path(agent_name)
        
        # Check if file exists to determine if we need a header
        file_exists = csv_path.exists()
        
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['moves'])  # Header
                writer.writerow([moves])
        except Exception as e:
            print(f"   Warning: Could not save to CSV: {e}")
    
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
            
            # Configure parallel batch size for non-visual mode
            if not use_visual:
                print("\n4) Parallel batch size (how many games to run simultaneously)?")
                print("   Higher = faster but more CPU/memory usage")
                print("   Recommended: 50-100 for most systems, 20-30 for Probability agent")
                batch_input = input("   Enter batch size (default 50): ").strip() or "50"
                try:
                    self.parallel_batch_size = int(batch_input)
                    if self.parallel_batch_size < 1:
                        self.parallel_batch_size = 50
                except ValueError:
                    self.parallel_batch_size = 50
                print(f"   ✓ Using batch size: {self.parallel_batch_size}")
            
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
        
        # Save game result to CSV
        agent_name = self.get_agent_name(self.brain_choice)
        self.save_game_to_csv(agent_name, game.moves)
        
        return game.moves
    
    async def run_single_game_for_agent(self, agent_class, name: str) -> int:
        """Run a single game for a specific agent class (used for parallel execution)."""
        # Create game and agent
        game = BattleshipGame()
        game.setup_board()
        agent = agent_class()
        
        # Play the game
        while not game.is_game_over():
            row, col = agent.get_move()
            result, sunk_ship = game.make_guess(row, col)
            agent.update(row, col, result, sunk_ship)
        
        # Save to CSV (thread-safe for async)
        self.save_game_to_csv(name, game.moves)
        return game.moves
    
    async def run_games(self, num_games: int):
        """Run multiple games with a progress bar, using parallel execution for non-visual mode."""
        if self.use_visual:
            # Visual mode must run sequentially
            with tqdm(total=num_games, desc="Running games", unit="game", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for i in range(1, num_games + 1):
                    moves = await self.run_single_game(i, num_games)
                    self.stats.add_game(moves)
                    pbar.update(1)
        else:
            # Background mode can run in parallel batches
            agent_name = self.get_agent_name(self.brain_choice)
            
            # Determine agent class
            if self.brain_choice == "2":
                agent_class = HuntAndTargetAgent
            elif self.brain_choice == "3":
                agent_class = ParityHuntAgent
            elif self.brain_choice == "4":
                agent_class = ProbabilityAgent
            else:
                agent_class = RandomAgent
            
            with tqdm(total=num_games, desc="Running games", unit="game", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                # Run games in batches
                for batch_start in range(0, num_games, self.parallel_batch_size):
                    batch_end = min(batch_start + self.parallel_batch_size, num_games)
                    batch_count = batch_end - batch_start
                    
                    # Create tasks for this batch
                    tasks = [
                        self.run_single_game_for_agent(agent_class, agent_name)
                        for _ in range(batch_count)
                    ]
                      # Run batch in parallel
                    results = await asyncio.gather(*tasks)
                    
                    # Record results
                    for move_count in results:
                        self.stats.add_game(move_count)
                    
                    # Update progress bar
                    pbar.update(batch_count)
    
    async def start_server_and_run(self, num_games: int):
        """Start the web server and run games."""
        self.server = BattleshipServer(self.use_visual)
        runner = web.AppRunner(self.server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        print(f"Server started at http://localhost:8080")
        
        if self.use_visual:
            # Open browser
            webbrowser.open('http://localhost:8080')
            print("Opening browser...")
            print("Waiting for SSE client connection...")
            
            # Wait for SSE client to connect (max 10 seconds)
            wait_time = 0
            while not self.server.client_connected and wait_time < 10:
                await asyncio.sleep(0.5)
                wait_time += 0.5
            
            if self.server.client_connected:
                print("✓ SSE client connected! Starting games...")
                await asyncio.sleep(0.5)  # Brief additional delay for stability
            else:
                print("⚠ Warning: SSE client not connected after 10 seconds, but continuing anyway...")
        
        # Run the games
        await self.run_games(num_games)
        
        # Print statistics immediately after games complete
        print("\n")  # Add spacing after progress bar
        self.stats.print_summary()
        
        await runner.cleanup()
    
    async def run_without_server(self, num_games: int):
        """Run games without the visual server."""
        await self.run_games(num_games)
        # Print statistics immediately after games complete
        print("\n")  # Add spacing after progress bar
        self.stats.print_summary()
    
    async def run_multi_agent_comparison(self, num_games: int, use_visual: bool):
        """Run all agents with parallel game execution and compare their statistics."""
        all_agents_config = [
            ("1", "Random", RandomAgent),
            ("2", "Hunt and Target", HuntAndTargetAgent),
            ("3", "Parity Hunt", ParityHuntAgent),
            ("4", "Probability", ProbabilityAgent)
        ]
        
        # Filter agents based on user selection
        agents_config = [config for config in all_agents_config if int(config[0]) in self.selected_agents]
        
        print("\n" + "=" * 70)
        print("🚀 PARALLEL GAME EXECUTION")
        print("=" * 70)
        print(f"Running {len(agents_config)} agent(s) with {num_games} games each")
        if not use_visual:
            print(f"Batch size: {self.parallel_batch_size} games at a time")
        print("=" * 70)
        print()
        
        # Store results for each agent
        for choice, name, agent_class in agents_config:
            print("\n" + "=" * 50)
            print(f"Running {name} Agent...")
            print("=" * 50)
            
            # Reset stats for this agent
            self.brain_choice = choice
            self.stats = GameStatistics()
            
            # Load existing CSV data
            existing_moves = self.load_csv_data(name)
            
            # If we have existing data and not using visual mode, skip running
            if existing_moves and not use_visual:
                print(f"   Using existing {len(existing_moves)} games from CSV")
                # Populate stats with existing data
                for move_count in existing_moves:
                    self.stats.add_game(move_count)
                
                # Only run NEW games if needed
                games_to_run = max(0, num_games - len(existing_moves))
                if games_to_run > 0:
                    print(f"   Running {games_to_run} additional games...")
                    await self.run_games(games_to_run)
                    
                print("\n")
                self.stats.print_summary()
            else:
                # Visual mode or no existing data - run all games
                if use_visual:
                    await self.start_server_and_run(num_games)
                else:
                    # Load existing data first if any
                    if existing_moves:
                        for move_count in existing_moves:
                            self.stats.add_game(move_count)
                        games_to_run = max(0, num_games - len(existing_moves))
                        if games_to_run > 0:
                            print(f"   Running {games_to_run} additional games...")
                            await self.run_games(games_to_run)
                    else:
                        await self.run_games(num_games)
                    
                    print("\n")
                    self.stats.print_summary()
            
            # Store the stats (use CSV data for visualization)
            csv_moves = self.load_csv_data(name)
            self.agent_stats[name] = {
                'stats': self.stats,
                'moves': csv_moves if csv_moves else self.stats.game_moves.copy()
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
        """Create a line graph visualization of agent performance, including Monte Carlo."""
        # Set dark mode style
        plt.style.use('dark_background')
        sns.set_palette("husl")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Define aesthetic colors for the 4 built-in agents
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD700']  # Random, H&T, Parity, Probability

        # Plot all agents we have in self.agent_stats
        for idx, (agent_name, data) in enumerate(self.agent_stats.items()):
            moves = data['moves']

            move_counter = Counter(moves)
            x_values = sorted(move_counter.keys())
            y_values = [move_counter[x] for x in x_values]

            ax.plot(
                x_values,
                y_values,
                marker='o',
                linewidth=2.5,
                markersize=6,
                label=agent_name,
                color=colors[idx % len(colors)],
                alpha=0.8,
            )

        # --- NEW: Add Monte Carlo line from existing CSV data ---
        monte_carlo_path = Path(__file__).parent / 'Monte_Carlo.csv'
        if monte_carlo_path.exists():
            monte_moves = []
            with open(monte_carlo_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header if present
                for row in reader:
                    if not row:
                        continue
                    try:
                        monte_moves.append(int(row[0]))
                    except ValueError:
                        continue

            if monte_moves:
                mc_counter = Counter(monte_moves)
                mc_x = sorted(mc_counter.keys())
                mc_y = [mc_counter[x] for x in mc_x]

                # Solid RED line for Monte Carlo
                ax.plot(
                    mc_x,
                    mc_y,
                    marker='o',
                    linewidth=2.5,
                    markersize=6,
                    label='Monte Carlo',
                    color='#FF0000',
                    alpha=0.9,
                    zorder=10,
                )

        # Styling
        ax.set_xlabel('Number of Moves to Finish Game', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Games Finished', fontsize=14, fontweight='bold')
        ax.set_title('Battleship Agent Performance Comparison', fontsize=16,
                     fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)

        plt.tight_layout()

        # Save + show
        output_path = Path(__file__).parent / 'agent_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"\n📊 Visualization saved to: {output_path}")
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
                # Visual mode always runs games
                asyncio.run(self.start_server_and_run(num_games))
            else:
                # Background mode - load existing CSV data if available
                agent_name = self.get_agent_name(self.brain_choice)
                existing_moves = self.load_csv_data(agent_name)
                
                if existing_moves:
                    print(f"   Using existing {len(existing_moves)} games from CSV")
                    # Populate stats with existing data
                    for move_count in existing_moves:
                        self.stats.add_game(move_count)
                    
                    # Only run NEW games if needed
                    games_to_run = max(0, num_games - len(existing_moves))
                    if games_to_run > 0:
                        print(f"   Running {games_to_run} additional games...")
                        asyncio.run(self.run_without_server(games_to_run))
                    else:
                        print("\n")
                        self.stats.print_summary()
                else:
                    # No existing data - run all games
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
