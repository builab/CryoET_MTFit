#!/usr/bin/env python3
"""
Analyze distances between consecutive particles along helical tubes.

This script reads a STAR file, groups particles by tomogram and tube ID,
sorts them along Y-coordinate, calculates consecutive distances, and 
plots a histogram of distances within a specified range.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starfile


def calculate_consecutive_distances(df: pd.DataFrame, angpix: float) -> np.ndarray:
    """
    Calculate distances between consecutive particles along Y-axis.
    
    Args:
        df: DataFrame with particles from a single tube, assumed sorted by Y.
        angpix: Pixel size in Angstroms.
    
    Returns:
        Array of distances in Angstroms between consecutive particles.
    """
    if len(df) < 2:
        return np.array([])
    
    # Get coordinates in pixels
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    
    # Calculate differences between consecutive points
    diffs = np.diff(coords, axis=0)
    
    # Calculate Euclidean distances and convert to Angstroms
    distances = np.sqrt(np.sum(diffs**2, axis=1)) * angpix
    
    return distances


def main():
    parser = argparse.ArgumentParser(
        description='Analyze distances between consecutive particles along tubes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--i', '--input',
        dest='input',
        required=True,
        help='Input STAR file'
    )
    parser.add_argument(
        '--o', '--output',
        dest='output',
        required=True,
        help='Output plot filename (e.g., plot.png)'
    )
    parser.add_argument(
        '--min_distance',
        type=float,
        required=True,
        help='Minimum distance in Angstroms to include in histogram'
    )
    parser.add_argument(
        '--max_distance',
        type=float,
        required=True,
        help='Maximum distance in Angstroms to include in histogram'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of bins for histogram'
    )
    
    args = parser.parse_args()
    
    # Validate distance range
    if args.min_distance >= args.max_distance:
        print(f"Error: min_distance ({args.min_distance}) must be less than "
              f"max_distance ({args.max_distance})")
        sys.exit(1)
    
    # Read STAR file
    print(f"Reading STAR file: {args.input}")
    try:
        data = starfile.read(args.input)
    except Exception as e:
        print(f"Error reading STAR file: {e}")
        sys.exit(1)
    
    # Handle different STAR file formats
    optics_df = None
    if isinstance(data, dict):
        # Multi-table STAR file (e.g., RELION 3.1+)
        if 'optics' in data:
            optics_df = data['optics']
        
        if 'particles' in data:
            df = data['particles']
        elif len(data) > 1:
            # Get the data table (not optics)
            df = [v for k, v in data.items() if k != 'optics'][0]
        else:
            df = list(data.values())[0]
    else:
        df = data
    
    print(f"Total particles: {len(df)}")
    
    # Check required columns in particles table
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnTomoName']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in particles: {missing_cols}")
        sys.exit(1)
    
    # Get pixel size from optics table or particles table
    if optics_df is not None and 'rlnImagePixelSize' in optics_df.columns:
        # Use pixel size from optics table (assume single optics group or take first)
        angpix = optics_df['rlnImagePixelSize'].iloc[0]
        print(f"Pixel size from optics table: {angpix:.3f} Å/px")
    elif 'rlnImagePixelSize' in df.columns:
        # Use pixel size from particles table
        angpix = df['rlnImagePixelSize'].iloc[0]
        print(f"Pixel size from particles table: {angpix:.3f} Å/px")
    else:
        print("Error: rlnImagePixelSize not found in optics or particles table")
        sys.exit(1)
    
    # Check for tube ID column
    has_tube_id = 'rlnHelicalTubeID' in df.columns
    
    # Determine grouping columns
    if has_tube_id:
        group_cols = ['rlnTomoName', 'rlnHelicalTubeID']
        print("Grouping by: rlnTomoName and rlnHelicalTubeID")
    else:
        group_cols = ['rlnTomoName']
        print("⚠ No rlnHelicalTubeID column found - grouping by rlnTomoName only")
    
    # Group particles and calculate distances
    print(f"\nProcessing groups...")
    all_distances = []
    
    grouped = df.groupby(group_cols)
    n_groups = len(grouped)
    print(f"Number of groups: {n_groups}")
    
    for group_name, group_df in grouped:
        # Sort by Y coordinate (ascending)
        group_sorted = group_df.sort_values('rlnCoordinateY', ascending=True)
        
        # Calculate distances
        distances = calculate_consecutive_distances(group_sorted, angpix)
        all_distances.extend(distances)
    
    all_distances = np.array(all_distances)
    
    if len(all_distances) == 0:
        print("Error: No distances calculated (all groups have < 2 particles)")
        sys.exit(1)
    
    print(f"\nTotal consecutive distances calculated: {len(all_distances)}")
    print(f"Distance range (all): {all_distances.min():.2f} - {all_distances.max():.2f} Å")
    print(f"Mean distance (all): {all_distances.mean():.2f} Å")
    print(f"Median distance (all): {np.median(all_distances):.2f} Å")
    
    # Filter distances by specified range
    mask = (all_distances >= args.min_distance) & (all_distances <= args.max_distance)
    filtered_distances = all_distances[mask]
    
    n_filtered = len(filtered_distances)
    n_total = len(all_distances)
    print(f"\nFiltered to range [{args.min_distance}, {args.max_distance}] Å:")
    print(f"  Kept: {n_filtered}/{n_total} distances ({100*n_filtered/n_total:.1f}%)")
    
    if n_filtered == 0:
        print(f"Warning: No distances in range [{args.min_distance}, {args.max_distance}] Å")
        print("Plotting empty histogram...")
    else:
        print(f"  Mean: {filtered_distances.mean():.2f} Å")
        print(f"  Median: {np.median(filtered_distances):.2f} Å")
        print(f"  Std dev: {filtered_distances.std():.2f} Å")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if n_filtered > 0:
        ax.hist(filtered_distances, bins=args.bins, color='steelblue', 
                edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Consecutive Particle Distances Along Tubes\n'
                 f'Range: [{args.min_distance:.1f}, {args.max_distance:.1f}] Å '
                 f'(n={n_filtered})',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    if n_filtered > 0:
        stats_text = (f'Mean: {filtered_distances.mean():.2f} Å\n'
                     f'Median: {np.median(filtered_distances):.2f} Å\n'
                     f'Std: {filtered_distances.std():.2f} Å')
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {args.output}")
    
    plt.close()


if __name__ == '__main__':
    main()