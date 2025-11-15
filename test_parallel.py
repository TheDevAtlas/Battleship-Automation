"""
Quick test script to demonstrate parallel agent execution
This simulates the user input to test multi-agent mode
"""
import sys
from io import StringIO
from unittest.mock import patch
import asyncio

# Simulate user input: Multi-agent mode, all agents, 10 games, no visual
mock_input = StringIO("2\n1,2,3,4\n10\nn\n")

# Redirect stdin to our mock input
with patch('builtins.input', side_effect=["2", "1,2,3,4", "10", "n"]):
    from main import BattleshipRunner
    runner = BattleshipRunner()
    runner.run()
