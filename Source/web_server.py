from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import threading
import time
from queue import Queue
import os
import logging

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Global state for game data
game_state = {
    'board': [[{'state': 'unknown', 'ship': False} for _ in range(10)] for _ in range(10)],
    'game_number': 0,
    'total_games': 0,
    'move_count': 0,
    'last_move': None,
    'is_running': False,
    'is_paused': False,
    'ships_status': []  # [{id, size, name, sunk}]
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

def update_game_state(board_data, game_num, total_games, move_count, last_move=None, probability_map=None):
    """Update the global game state from the game logic"""
    with state_lock:
        game_state['game_number'] = game_num
        game_state['total_games'] = total_games
        game_state['move_count'] = move_count
        game_state['last_move'] = last_move
        game_state['is_running'] = True
        
        # Normalize probability map if provided
        max_prob = 1.0
        if probability_map is not None:
            max_prob = float(probability_map.max()) if hasattr(probability_map, 'max') else max(max(row) for row in probability_map)
            if max_prob == 0:
                max_prob = 1.0  # Avoid division by zero
        
        # Prepare ship status (names inferred by size and order)
        # Standard naming for 10x10: 5: Carrier, 4: Battleship, 3: Cruiser/Submarine, 2: Destroyer
        size_name_pool = {5: ['Carrier'], 4: ['Battleship'], 3: ['Cruiser', 'Submarine'], 2: ['Destroyer']}
        size_name_used = {5:0,4:0,3:0,2:0}
        ships_status = []
        for ship in board_data['ships']:
            size = ship.get('size', 0)
            sunk = ship.get('sunk', False)
            names = size_name_pool.get(size, [f"Size {size}"])
            idx = min(size_name_used.get(size, 0), len(names)-1)
            name = names[idx]
            size_name_used[size] = size_name_used.get(size, 0) + 1
            ships_status.append({
                'id': ship.get('id'),
                'size': size,
                'name': name,
                'sunk': bool(sunk)
            })
        game_state['ships_status'] = ships_status

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
                    # Clear probability for guessed cells
                    cell['probability'] = 0
                    cell['rawProbability'] = 0
                else:
                    cell['state'] = 'unknown'
                    # Add normalized probability data for unknown cells
                    if probability_map is not None:
                        raw_prob = float(probability_map[row][col])
                        normalized_prob = raw_prob / max_prob
                        cell['probability'] = normalized_prob
                        cell['rawProbability'] = raw_prob
                    else:
                        cell['probability'] = 0
                        cell['rawProbability'] = 0
                
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
    # Suppress Flask's request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    # For development/testing only
    app.run(host='127.0.0.1', port=5000, debug=True)
