# Battleship Automation

A Python-driven battleship game with optional visual feedback using a web interface.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running

Run the main script:
```bash
python main.py
```

You'll be asked three questions:

1. **What brain are we using?**
   - `1` - Random (makes random guesses without duplicates)
   - `2` - Smart (coming soon)

2. **How many games to run?**
   - Enter any number (default: 10)

3. **Should we use the visual GUI?**
   - `y` - Opens browser to visualize each game
   - `n` - Runs games in the background (faster)

## How It Works

- **Python Backend**: Controls the game logic, ship placement, and agent decision-making
- **Web Interface**: Displays the game board with animations (optional)
- **WebSocket**: Connects Python to the browser for real-time updates

## Statistics Tracked

After all games complete, you'll see:
- Total games played
- Average moves per game
- Best game (fewest moves)
- Worst game (most moves)
- Median moves
- Distribution of moves across ranges

## Controls (when using keyboard mode in HTML)

The HTML still supports manual controls if you open it directly:
- **Space**: Start/stop random board generation
- **R**: Reset board
- **T**: Test random updates
- **G**: Generate single random grid

## Project Structure

- `main.py` - Main entry point and game runner
- `battleship.py` - Game logic, agent, and statistics
- `index.html` - Visual interface
- `requirements.txt` - Python dependencies
