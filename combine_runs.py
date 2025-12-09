#!/usr/bin/env python3
"""
Combine multiple ultimatum game runs into a single CSV file.

This script takes multiple CSV files from separate runs (e.g., 5 runs of 1 game each)
and combines them in the same order as if they were run together in a single run.

Usage:
    # Combine specific CSV files
    python combine_runs.py file1.csv file2.csv file3.csv -o combined.csv
    
    # Combine all CSV files matching a pattern
    python combine_runs.py test_results/ultimatum_p0-9_g1_r1.5*.csv -o combined.csv
    
    # Combine all CSV files in a directory
    python combine_runs.py test_results/*.csv -o combined.csv
"""

import argparse
import csv
import glob
import os
from typing import List, Dict, Any


def load_csv_results(filename: str) -> List[Dict[str, Any]]:
    """Load game results from a CSV file."""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['game_id'] = int(row['game_id'])
            row['proposer_idx'] = int(row['proposer_idx'])
            row['responder_idx'] = int(row['responder_idx'])
            row['offer'] = float(row['offer']) if row['offer'] else 0.0
            row['proposer_payoff'] = float(row['proposer_payoff']) if row['proposer_payoff'] else 0.0
            row['responder_payoff'] = float(row['responder_payoff']) if row['responder_payoff'] else 0.0
            row['proposer_age'] = int(row['proposer_age']) if row['proposer_age'] else 0
            row['responder_age'] = int(row['responder_age']) if row['responder_age'] else 0
            row['euclidean_distance'] = float(row['euclidean_distance']) if row['euclidean_distance'] else 0.0
            row['manhattan_distance'] = float(row['manhattan_distance']) if row['manhattan_distance'] else 0.0
            row['proposer_tokens'] = int(row['proposer_tokens']) if row['proposer_tokens'] else 0
            row['responder_tokens'] = int(row['responder_tokens']) if row['responder_tokens'] else 0
            row['total_tokens'] = int(row['total_tokens']) if row['total_tokens'] else 0
            row['proposer_failed'] = row['proposer_failed'].lower() == 'true' if row['proposer_failed'] else False
            row['responder_failed'] = row['responder_failed'].lower() == 'true' if row['responder_failed'] else False
            # Store source file for tracking
            row['_source_file'] = filename
            results.append(row)
    return results


def combine_runs(csv_files: List[str], output_file: str, verbose: bool = True):
    """
    Combine multiple CSV files into one, maintaining the correct order.
    
    The order is:
    1. Sort by proposer_idx (ascending)
    2. Then by responder_idx (ascending, skipping self)
    3. Then by original game_id (to preserve order within each pair from each run)
    4. Then by source file order (to preserve order across runs)
    
    Args:
        csv_files: List of CSV file paths to combine
        output_file: Output CSV file path
        verbose: Print progress information
    """
    if verbose:
        print("=" * 70)
        print("COMBINING ULTIMATUM GAME RUNS")
        print("=" * 70)
        print(f"Input files: {len(csv_files)}")
        for i, f in enumerate(csv_files, 1):
            print(f"  {i}. {f}")
        print()
    
    # Load all results from all files
    all_results = []
    file_order = {}  # Track order of files for consistent sorting
    
    for idx, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            print(f"⚠️  Warning: File not found: {csv_file}")
            continue
        
        file_order[csv_file] = idx
        results = load_csv_results(csv_file)
        
        if verbose:
            print(f"Loaded {len(results)} games from {os.path.basename(csv_file)}")
        
        all_results.extend(results)
    
    if not all_results:
        print("❌ No results to combine!")
        return
    
    if verbose:
        print(f"\nTotal games loaded: {len(all_results)}")
        print("\nSorting games...")
    
    # Sort results to match the order from run_simulation:
    # 1. proposer_idx (ascending)
    # 2. responder_idx (ascending, but skip self)
    # 3. Within each (proposer_idx, responder_idx) pair, maintain order from original runs
    #    We'll use source file order and original game_id to preserve the sequence
    
    def sort_key(result: Dict[str, Any]) -> tuple:
        """Sort key to match the order from run_simulation."""
        proposer_idx = result['proposer_idx']
        responder_idx = result['responder_idx']
        source_file = result['_source_file']
        original_game_id = result['game_id']
        file_idx = file_order.get(source_file, 999)
        
        # Primary sort: proposer_idx, responder_idx
        # Secondary sort: file order (to preserve order across runs), then original game_id
        return (proposer_idx, responder_idx, file_idx, original_game_id)
    
    all_results.sort(key=sort_key)
    
    # Renumber game_id sequentially
    for idx, result in enumerate(all_results, start=1):
        result['game_id'] = idx
    
    # Remove the temporary _source_file field
    for result in all_results:
        result.pop('_source_file', None)
    
    # Get fieldnames from first result (should be consistent across all files)
    if all_results:
        fieldnames = list(all_results[0].keys())
        
        # Ensure standard order (matching save_results_to_csv)
        standard_order = [
            "game_id",
            "proposer_name",
            "responder_name",
            "proposer_idx",
            "responder_idx",
            "offer",
            "decision",
            "proposer_payoff",
            "responder_payoff",
            "proposer_age",
            "proposer_gender",
            "proposer_job",
            "responder_age",
            "responder_gender",
            "responder_job",
            "euclidean_distance",
            "manhattan_distance",
            "proposer_tokens",
            "responder_tokens",
            "total_tokens",
            "proposer_failed",
            "responder_failed",
            "proposer_reasoning",
            "responder_reasoning",
        ]
        
        # Use standard order, but include any extra fields
        fieldnames = [f for f in standard_order if f in fieldnames]
        extra_fields = [f for f in all_results[0].keys() if f not in standard_order]
        fieldnames.extend(extra_fields)
    
    # Write combined results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            # Convert boolean fields back to strings
            row = result.copy()
            row['proposer_failed'] = str(row['proposer_failed'])
            row['responder_failed'] = str(row['responder_failed'])
            writer.writerow(row)
    
    if verbose:
        print(f"\n✅ Combined {len(all_results)} games into {output_file}")
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Count games per proposer-responder pair
        pair_counts = {}
        for result in all_results:
            pair = (result['proposer_idx'], result['responder_idx'])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        print(f"Total games: {len(all_results)}")
        print(f"Unique proposer-responder pairs: {len(pair_counts)}")
        
        # Show games per pair
        if len(pair_counts) <= 20:  # Only show if not too many
            print("\nGames per pair:")
            for (p_idx, r_idx), count in sorted(pair_counts.items()):
                proposer_name = all_results[0]['proposer_name'] if all_results else "?"
                responder_name = all_results[0]['responder_name'] if all_results else "?"
                # Find names for this pair
                for r in all_results:
                    if r['proposer_idx'] == p_idx and r['responder_idx'] == r_idx:
                        proposer_name = r['proposer_name']
                        responder_name = r['responder_name']
                        break
                print(f"  Player {p_idx} → Player {r_idx}: {count} games")
        
        print("=" * 70)


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns to actual file paths."""
    files = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            files.extend(expanded)
        elif os.path.exists(pattern):
            files.append(pattern)
        else:
            print(f"⚠️  Warning: Pattern/file not found: {pattern}")
    return sorted(files)  # Sort for consistent ordering


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine multiple ultimatum game CSV runs into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine specific CSV files
  python combine_runs.py file1.csv file2.csv file3.csv -o combined.csv
  
  # Combine all CSV files matching a pattern
  python combine_runs.py test_results/ultimatum_p0-9_g1_r1.5*.csv -o combined.csv
  
  # Combine all CSV files in a directory (sorted alphabetically)
  python combine_runs.py test_results/*.csv -o combined.csv
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='CSV files to combine (supports glob patterns)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='combined_results.csv',
        help='Output CSV file path (default: combined_results.csv)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Expand file patterns
    input_files = expand_file_patterns(args.input_files)
    
    if not input_files:
        print("❌ No input files found!")
        return
    
    verbose = args.verbose and not args.quiet
    
    # Combine runs
    combine_runs(input_files, args.output, verbose=verbose)


if __name__ == "__main__":
    main()

