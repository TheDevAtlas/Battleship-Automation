import csv
import os
from datetime import datetime
import statistics

class GameAnalyzer:
    """Handles game analysis, comparison, and data logging"""
    
    def __init__(self):
        self.data_dir = "Data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create Data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def generate_filename(self, comparison_type="single", run_number=1):
        """Generate filename with date and run number"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{self.data_dir}/{date_str}-{comparison_type}-Run-{run_number}.csv"
    
    def save_detailed_results(self, results_list, comparison_type="single", run_number=1):
        """Save detailed results to CSV file"""
        filename = self.generate_filename(comparison_type, run_number)
        
        # Check if file exists and increment run number if needed
        counter = run_number
        while os.path.exists(filename):
            counter += 1
            filename = self.generate_filename(comparison_type, counter)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with metadata
            writer.writerow(['# Battleship Game Analysis Results'])
            writer.writerow([f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
            writer.writerow([f'# Comparison Type: {comparison_type}'])
            writer.writerow([])
            
            if len(results_list) == 1:
                # Single bot analysis
                self._write_single_bot_data(writer, results_list[0])
            elif len(results_list) == 2:
                # Two-bot comparison
                self._write_two_bot_comparison(writer, results_list[0], results_list[1])
            elif len(results_list) == 3:
                # Three-bot comparison
                self._write_three_bot_comparison(writer, results_list[0], results_list[1], results_list[2])
        
        print(f"\nDetailed analysis saved to: {filename}")
        return filename
    
    def _write_single_bot_data(self, writer, results):
        """Write single bot data to CSV"""
        stats = results['stats']
        
        writer.writerow(['# SUMMARY STATISTICS'])
        writer.writerow(['Player Name', stats['player_name']])
        writer.writerow(['Total Games', stats['num_games']])
        writer.writerow(['Average Moves', f"{stats['average_moves']:.2f}"])
        writer.writerow(['Median Moves', f"{stats['median_moves']:.2f}"])
        writer.writerow(['Best Game', stats['best_moves']])
        writer.writerow(['Worst Game', stats['worst_moves']])
        writer.writerow(['Standard Deviation', f"{stats['std_dev']:.2f}"])
        writer.writerow(['Total Time (s)', f"{stats['total_time']:.2f}"])
        writer.writerow(['Avg Time per Game (s)', f"{stats['avg_time_per_game']:.3f}"])
        writer.writerow([])
        
        writer.writerow(['# GAME-BY-GAME RESULTS'])
        writer.writerow(['Game Number', 'Moves Required'])
        for i, result in enumerate(results['results'], 1):
            writer.writerow([i, result['moves']])
        
        writer.writerow([])
        writer.writerow(['# MOVE COUNT DISTRIBUTION'])
        move_counts = stats['move_counts']
        
        # Create distribution buckets
        buckets = {
            '≤30 moves': len([m for m in move_counts if m <= 30]),
            '31-40 moves': len([m for m in move_counts if 31 <= m <= 40]),
            '41-50 moves': len([m for m in move_counts if 41 <= m <= 50]),
            '51-60 moves': len([m for m in move_counts if 51 <= m <= 60]),
            '61-70 moves': len([m for m in move_counts if 61 <= m <= 70]),
            '71-80 moves': len([m for m in move_counts if 71 <= m <= 80]),
            '81-90 moves': len([m for m in move_counts if 81 <= m <= 90]),
            '91+ moves': len([m for m in move_counts if m >= 91])
        }
        
        writer.writerow(['Move Range', 'Game Count', 'Percentage'])
        for range_name, count in buckets.items():
            percentage = (count / len(move_counts)) * 100
            writer.writerow([range_name, count, f"{percentage:.1f}%"])
    
    def _write_two_bot_comparison(self, writer, results1, results2):
        """Write two-bot comparison data to CSV"""
        stats1, stats2 = results1['stats'], results2['stats']
        
        writer.writerow(['# COMPARISON SUMMARY'])
        writer.writerow(['Metric', stats1['player_name'], stats2['player_name'], 'Winner'])
        
        metrics = [
            ('Average Moves', 'average_moves', 'lower'),
            ('Median Moves', 'median_moves', 'lower'),
            ('Best Game', 'best_moves', 'lower'),
            ('Worst Game', 'worst_moves', 'lower'),
            ('Consistency (Std Dev)', 'std_dev', 'lower'),
            ('Avg Time per Game', 'avg_time_per_game', 'lower')
        ]
        
        for metric_name, metric_key, better in metrics:
            val1, val2 = stats1[metric_key], stats2[metric_key]
            winner = stats1['player_name'] if val1 < val2 else stats2['player_name']
            
            if isinstance(val1, float):
                writer.writerow([metric_name, f"{val1:.2f}", f"{val2:.2f}", winner])
            else:
                writer.writerow([metric_name, val1, val2, winner])
        
        writer.writerow([])
        writer.writerow(['# HEAD-TO-HEAD GAME RESULTS'])
        writer.writerow(['Game Number', f'{stats1["player_name"]} Moves', f'{stats2["player_name"]} Moves', 'Winner'])
        
        moves1 = stats1['move_counts']
        moves2 = stats2['move_counts']
        
        for i in range(len(moves1)):
            winner = stats1['player_name'] if moves1[i] < moves2[i] else stats2['player_name']
            if moves1[i] == moves2[i]:
                winner = 'Tie'
            writer.writerow([i+1, moves1[i], moves2[i], winner])
        
        # Win statistics
        wins1 = sum(1 for i in range(len(moves1)) if moves1[i] < moves2[i])
        wins2 = sum(1 for i in range(len(moves1)) if moves2[i] < moves1[i])
        ties = sum(1 for i in range(len(moves1)) if moves1[i] == moves2[i])
        
        writer.writerow([])
        writer.writerow(['# WIN STATISTICS'])
        writer.writerow([f'{stats1["player_name"]} Wins', wins1])
        writer.writerow([f'{stats2["player_name"]} Wins', wins2])
        writer.writerow(['Ties', ties])
        writer.writerow([f'{stats1["player_name"]} Win Rate', f"{(wins1/len(moves1))*100:.1f}%"])
        writer.writerow([f'{stats2["player_name"]} Win Rate', f"{(wins2/len(moves1))*100:.1f}%"])
        
        # Performance improvement
        improvement = ((stats1['average_moves'] - stats2['average_moves']) / stats1['average_moves']) * 100
        writer.writerow([])
        writer.writerow(['# PERFORMANCE ANALYSIS'])
        writer.writerow(['Performance Improvement', f"{improvement:.1f}%"])
        writer.writerow(['Better Performer', stats2['player_name'] if improvement > 0 else stats1['player_name']])
        
        # Efficiency analysis
        writer.writerow([])
        writer.writerow(['# EFFICIENCY ANALYSIS'])
        efficient1 = len([m for m in moves1 if m <= 50])
        efficient2 = len([m for m in moves2 if m <= 50])
        poor1 = len([m for m in moves1 if m >= 80])
        poor2 = len([m for m in moves2 if m >= 80])
        
        writer.writerow(['Efficient Games (≤50 moves)', f'{stats1["player_name"]}', f'{stats2["player_name"]}'])
        writer.writerow(['Count', efficient1, efficient2])
        writer.writerow(['Percentage', f"{(efficient1/len(moves1))*100:.1f}%", f"{(efficient2/len(moves2))*100:.1f}%"])
        writer.writerow([])
        writer.writerow(['Poor Performance (≥80 moves)', f'{stats1["player_name"]}', f'{stats2["player_name"]}'])
        writer.writerow(['Count', poor1, poor2])
        writer.writerow(['Percentage', f"{(poor1/len(moves1))*100:.1f}%", f"{(poor2/len(moves2))*100:.1f}%"])
    
    def _write_three_bot_comparison(self, writer, results1, results2, results3):
        """Write three-bot comparison data to CSV"""
        stats = [results1['stats'], results2['stats'], results3['stats']]
        
        writer.writerow(['# THREE-WAY COMPARISON SUMMARY'])
        writer.writerow(['Metric', stats[0]['player_name'], stats[1]['player_name'], stats[2]['player_name'], 'Winner'])
        
        metrics = [
            ('Average Moves', 'average_moves', 'lower'),
            ('Median Moves', 'median_moves', 'lower'),
            ('Best Game', 'best_moves', 'lower'),
            ('Worst Game', 'worst_moves', 'lower'),
            ('Consistency (Std Dev)', 'std_dev', 'lower'),
            ('Avg Time per Game', 'avg_time_per_game', 'lower')
        ]
        
        wins = [0, 0, 0]
        
        for metric_name, metric_key, better in metrics:
            values = [stat[metric_key] for stat in stats]
            
            if better == 'lower':
                best_idx = values.index(min(values))
            else:
                best_idx = values.index(max(values))
            
            wins[best_idx] += 1
            winner = stats[best_idx]['player_name']
            
            formatted_vals = [f"{val:.2f}" if isinstance(val, float) else str(val) for val in values]
            writer.writerow([metric_name] + formatted_vals + [winner])
        
        writer.writerow(['Overall Metric Wins'] + [str(w) for w in wins] + [''])
        
        writer.writerow([])
        writer.writerow(['# HEAD-TO-HEAD GAME RESULTS'])
        writer.writerow(['Game Number', stats[0]['player_name'], stats[1]['player_name'], stats[2]['player_name'], 'Winner'])
        
        moves = [stat['move_counts'] for stat in stats]
        
        for i in range(len(moves[0])):
            game_moves = [moves[j][i] for j in range(3)]
            min_moves = min(game_moves)
            winners = [j for j, m in enumerate(game_moves) if m == min_moves]
            
            if len(winners) == 1:
                winner = stats[winners[0]]['player_name']
            else:
                winner = 'Tie'
            
            writer.writerow([i+1] + game_moves + [winner])
        
        # Win statistics
        writer.writerow([])
        writer.writerow(['# WIN STATISTICS'])
        
        game_wins = [0, 0, 0]
        for i in range(len(moves[0])):
            game_moves = [moves[j][i] for j in range(3)]
            min_moves = min(game_moves)
            winners = [j for j, m in enumerate(game_moves) if m == min_moves]
            if len(winners) == 1:
                game_wins[winners[0]] += 1
        
        for i, stat in enumerate(stats):
            writer.writerow([f'{stat["player_name"]} Game Wins', game_wins[i]])
            writer.writerow([f'{stat["player_name"]} Win Rate', f"{(game_wins[i]/len(moves[0]))*100:.1f}%"])
        
        # Head-to-head comparisons
        writer.writerow([])
        writer.writerow(['# HEAD-TO-HEAD WIN RATES'])
        
        for i in range(3):
            for j in range(i+1, 3):
                wins_i = sum(1 for k in range(len(moves[0])) if moves[i][k] < moves[j][k])
                wins_j = sum(1 for k in range(len(moves[0])) if moves[j][k] < moves[i][k])
                writer.writerow([f'{stats[i]["player_name"]} vs {stats[j]["player_name"]}', 
                               f'{wins_i}/{len(moves[0])} ({(wins_i/len(moves[0]))*100:.1f}%)'])
        
        # Performance improvements
        writer.writerow([])
        writer.writerow(['# PERFORMANCE IMPROVEMENTS vs Random Player'])
        baseline = stats[0]['average_moves']  # Assuming first is Random Player
        for i in range(1, 3):
            improvement = ((baseline - stats[i]['average_moves']) / baseline) * 100
            writer.writerow([f'{stats[i]["player_name"]} Improvement', f"{improvement:.1f}%"])
    
    def print_concise_summary(self, results_list, comparison_type="single"):
        """Print concise summary to terminal"""
        if len(results_list) == 1:
            self._print_single_summary(results_list[0])
        elif len(results_list) == 2:
            self._print_two_bot_summary(results_list[0], results_list[1])
        elif len(results_list) == 3:
            self._print_three_bot_summary(results_list[0], results_list[1], results_list[2])
    
    def _print_single_summary(self, results):
        """Print concise single bot summary"""
        stats = results['stats']
        
        print(f"\n{'='*60}")
        print(f"BATTLESHIP RESULTS - {stats['player_name']}")
        print(f"{'='*60}")
        print(f"Games played: {stats['num_games']}")
        print(f"Average moves: {stats['average_moves']:.1f}")
        print(f"Best game: {stats['best_moves']} moves")
        print(f"Worst game: {stats['worst_moves']} moves")
        print(f"Consistency: {stats['std_dev']:.1f} std dev")
        
        # Efficiency stats
        moves = stats['move_counts']
        efficient = len([m for m in moves if m <= 50])
        poor = len([m for m in moves if m >= 80])
        
        print(f"\nEfficiency:")
        print(f"  ≤50 moves: {efficient}/{len(moves)} ({(efficient/len(moves))*100:.1f}%)")
        print(f"  ≥80 moves: {poor}/{len(moves)} ({(poor/len(moves))*100:.1f}%)")
        print(f"{'='*60}")
    
    def _print_two_bot_summary(self, results1, results2):
        """Print concise two-bot comparison summary"""
        stats1, stats2 = results1['stats'], results2['stats']
        
        print(f"\n{'='*80}")
        print(f"BATTLESHIP COMPARISON - {stats1['player_name']} vs {stats2['player_name']}")
        print(f"{'='*80}")
        
        print(f"{'Metric':<20} | {stats1['player_name']:<20} | {stats2['player_name']:<20} | Winner")
        print(f"{'-'*80}")
        
        metrics = [
            ('Average Moves', 'average_moves'),
            ('Best Game', 'best_moves'),
            ('Worst Game', 'worst_moves'),
            ('Consistency', 'std_dev')
        ]
        
        wins1 = wins2 = 0
        
        for metric_name, metric_key in metrics:
            val1, val2 = stats1[metric_key], stats2[metric_key]
            winner = stats1['player_name'] if val1 < val2 else stats2['player_name']
            
            if val1 < val2:
                wins1 += 1
            elif val2 < val1:
                wins2 += 1
            
            print(f"{metric_name:<20} | {val1:<20.1f} | {val2:<20.1f} | {winner}")
        
        print(f"{'-'*80}")
        
        # Game win rate
        moves1, moves2 = stats1['move_counts'], stats2['move_counts']
        game_wins2 = sum(1 for i in range(len(moves1)) if moves2[i] < moves1[i])
        win_rate = (game_wins2 / len(moves1)) * 100
        
        print(f"Game Win Rate: {stats2['player_name']} won {game_wins2}/{len(moves1)} games ({win_rate:.1f}%)")
        
        # Performance improvement
        improvement = ((stats1['average_moves'] - stats2['average_moves']) / stats1['average_moves']) * 100
        print(f"Performance Improvement: {improvement:.1f}% better" if improvement > 0 else f"Performance: {abs(improvement):.1f}% worse")
        
        # Efficiency comparison
        efficient1 = len([m for m in moves1 if m <= 50])
        efficient2 = len([m for m in moves2 if m <= 50])
        
        print(f"\nEfficiency (≤50 moves):")
        print(f"  {stats1['player_name']}: {efficient1}/{len(moves1)} ({(efficient1/len(moves1))*100:.1f}%)")
        print(f"  {stats2['player_name']}: {efficient2}/{len(moves2)} ({(efficient2/len(moves2))*100:.1f}%)")
        
        print(f"\n🏆 WINNER: {stats2['player_name'] if improvement > 0 else stats1['player_name']}")
        print(f"{'='*80}")
    
    def _print_three_bot_summary(self, results1, results2, results3):
        """Print concise three-bot comparison summary"""
        stats = [results1['stats'], results2['stats'], results3['stats']]
        
        print(f"\n{'='*100}")
        print(f"THREE-WAY BATTLESHIP COMPARISON")
        print(f"{'='*100}")
        
        print(f"{'Metric':<18} | {stats[0]['player_name']:<18} | {stats[1]['player_name']:<18} | {stats[2]['player_name']:<18} | Winner")
        print(f"{'-'*100}")
        
        metrics = [
            ('Average Moves', 'average_moves'),
            ('Best Game', 'best_moves'),
            ('Worst Game', 'worst_moves'),
            ('Consistency', 'std_dev')
        ]
        
        wins = [0, 0, 0]
        
        for metric_name, metric_key in metrics:
            values = [stat[metric_key] for stat in stats]
            best_idx = values.index(min(values))
            wins[best_idx] += 1
            winner = stats[best_idx]['player_name']
            
            formatted_vals = [f"{val:.1f}" for val in values]
            print(f"{metric_name:<18} | {formatted_vals[0]:<18} | {formatted_vals[1]:<18} | {formatted_vals[2]:<18} | {winner}")
        
        print(f"{'-'*100}")
        print(f"{'Metric Wins':<18} | {wins[0]:<18} | {wins[1]:<18} | {wins[2]:<18} |")
        
        # Overall winner
        max_wins = max(wins)
        if wins.count(max_wins) == 1:
            winner_idx = wins.index(max_wins)
            print(f"\n🏆 OVERALL WINNER: {stats[winner_idx]['player_name']} ({max_wins}/4 metrics)")
        else:
            tied_bots = [stats[i]['player_name'] for i, w in enumerate(wins) if w == max_wins]
            print(f"\n🤝 TIE between {' and '.join(tied_bots)}")
        
        # Performance improvements vs baseline
        baseline = stats[0]['average_moves']  # Random Player
        print(f"\nPerformance vs Random Player:")
        for i in range(1, 3):
            improvement = ((baseline - stats[i]['average_moves']) / baseline) * 100
            print(f"  {stats[i]['player_name']}: {improvement:.1f}% improvement")
        
        print(f"{'='*100}")

# Global analyzer instance
analyzer = GameAnalyzer()

def analyze_and_save_results(results_list, comparison_type="single", run_number=1):
    """Main function to analyze results and save data"""
    # Save detailed results to CSV
    filename = analyzer.save_detailed_results(results_list, comparison_type, run_number)
    
    # Print concise summary to terminal
    analyzer.print_concise_summary(results_list, comparison_type)
    
    return filename