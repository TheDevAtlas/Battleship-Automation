from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import threading
import time
from queue import Queue

app = Flask(__name__)
CORS(app)

# Global state for game data
game_state = {
    'board': [[{'state': 'unknown', 'ship': False} for _ in range(10)] for _ in range(10)],
    'game_number': 0,
    'total_games': 0,
    'move_count': 0,
    'last_move': None,
    'is_running': False,
    'is_paused': False
}

state_lock = threading.Lock()

@app.route('/')
def index():
    """Serve the main game page"""
    return render_template('game.html')

@app.route('/api/state')
def get_state():
    """API endpoint to get current game state"""
    with state_lock:
        return jsonify(game_state)

def update_game_state(board_data, game_num, total_games, move_count, last_move=None):
    """Update the global game state from the game logic"""
    with state_lock:
        game_state['game_number'] = game_num
        game_state['total_games'] = total_games
        game_state['move_count'] = move_count
        game_state['last_move'] = last_move
        game_state['is_running'] = True
        
        # Update board state
        for row in range(10):
            for col in range(10):
                cell = game_state['board'][row][col]
                
                # Check if guessed
                if board_data['guesses'][row][col]:
                    if board_data['hits'][row][col]:
                        # Check if this ship is sunk
                        ship_id = board_data['grid'][row][col]
                        is_sunk = False
                        for ship in board_data['ships']:
                            if ship['id'] == ship_id and ship.get('sunk', False):
                                is_sunk = True
                                break
                        
                        cell['state'] = 'sunk' if is_sunk else 'hit'
                    else:
                        cell['state'] = 'miss'
                else:
                    cell['state'] = 'unknown'
                
                # Track if there's a ship here (for debugging/analysis)
                cell['ship'] = board_data['grid'][row][col] != 0

def set_paused(paused):
    """Set the paused state"""
    with state_lock:
        game_state['is_paused'] = paused

def set_running(running):
    """Set the running state"""
    with state_lock:
        game_state['is_running'] = running

def start_server():
    """Start the Flask server in a separate thread"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    # For development/testing only
    app.run(host='127.0.0.1', port=5000, debug=True)
