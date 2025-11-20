#!/usr/bin/env python3
"""
Plot histogram of pairwise distances between scatter points in a STAR file.
Groups points by rlnTomoName and calculates distances in Angstroms.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys
import starfile
import pandas as pd
from scipy.stats import gaussian_kde

def calculate_pairwise_distances(df: pd.DataFrame, angpix: float, dist_threshold: float):
    """
    Calculate pairwise distances for particles with the same rlnTomoName.
    
    Args:
        df: pandas DataFrame with particle data
        angpix: Pixel size in Angstroms
        dist_threshold: Maximum distance threshold in Angstroms
        
    Returns:
        list: All pairwise distances below threshold
    """
    # Check required columns
    required_cols = ['rlnTomoName', 'rlnCoordinateX', 'rlnCoordinateY', 
                     'rlnCoordinateZ']
    
    # Handle column naming (with or without _rln prefix) for coordinates/tomoName
    for col in required_cols:
        if col not in df.columns and f'_{col}' in df.columns:
            df = df.rename(columns={f'_{col}': col})
    
    # Check for origin columns (optional)
    origin_cols = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
    has_origin = True
    for col in origin_cols:
        if col not in df.columns and f'_{col}' in df.columns:
            df = df.rename(columns={f'_{col}': col})
        if col not in df.columns:
            has_origin = False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)
    
    if not has_origin:
        print("Warning: Origin columns (rlnOriginXAngst/Y/Z) not found. Using coordinates without origin adjustment.", file=sys.stderr)
    
    # Group particles by rlnTomoName
    all_distances = []
    
    for tomo_name, group in df.groupby('rlnTomoName'):
        if len(group) < 2:
            continue
        
        # Convert coordinates to Angstroms using the single angpix value
        coords_x = group['rlnCoordinateX'].values * angpix
        coords_y = group['rlnCoordinateY'].values * angpix
        coords_z = group['rlnCoordinateZ'].values * angpix

        if has_origin:
            # Adjusted coordinates: xyz = coordinate * angpix - originAngst
            coords_angstrom = np.column_stack([
                coords_x - group['rlnOriginXAngst'].values,
                coords_y - group['rlnOriginYAngst'].values,
                coords_z - group['rlnOriginZAngst'].values
            ])
        else:
            # Without origin adjustment
            coords_angstrom = np.column_stack([coords_x, coords_y, coords_z])
        
        # Calculate all pairwise distances
        for i, j in combinations(range(len(coords_angstrom)), 2):
            distance = np.linalg.norm(coords_angstrom[i] - coords_angstrom[j])
            if distance <= dist_threshold:
                all_distances.append(distance)
    
    return all_distances


def plot_histogram(distances: list, min_dist: float, max_dist: float, output_path: str):
    """
    Plot histogram of distances with overlaid KDE and save to file.
    
    Args:
        distances: List of distances in Angstroms
        min_dist: Minimum distance for histogram range
        max_dist: Maximum distance for histogram range
        output_path: Path to save the output image
    """
    # Filter distances to plotting range
    filtered_distances = [d for d in distances if min_dist <= d <= max_dist]
    
    if not filtered_distances:
        print(f"Warning: No distances found in range [{min_dist}, {max_dist}] Angstroms", file=sys.stderr)
        return
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to numpy array
    distances_array = np.array(filtered_distances)
    
    # Plot histogram with density normalization
    counts, bins, patches = ax.hist(distances_array, bins=50, density=True, 
                                     alpha=0.6, edgecolor='black', label='Histogram (density)')
    
    # Create KDE using scipy
    kde = gaussian_kde(distances_array)
    x_range = np.linspace(min_dist, max_dist, 200)
    kde_values = kde(x_range)
    
    # Plot KDE on the same axis
    ax.plot(x_range, kde_values, color='red', linewidth=2, label='KDE')

    # Labels and title
    ax.set_xlabel('Distance (Angstroms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Pairwise Distance Distribution with KDE ({min_dist}-{max_dist} Å)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    mean_dist = np.mean(filtered_distances)
    median_dist = np.median(filtered_distances)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_dist, color='darkblue', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_dist:.2f} Å')
    ax.axvline(median_dist, color='darkgreen', linestyle='--', linewidth=2, 
               label=f'Median: {median_dist:.2f} Å')
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram and KDE saved to: {output_path}")
    print(f"Total distances in range: {len(filtered_distances)}")
    print(f"Mean distance: {mean_dist:.2f} Å")
    print(f"Median distance: {median_dist:.2f} Å")


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram of pairwise distances from STAR file'
    )
    parser.add_argument('--i', required=True, help='Input STAR file')
    parser.add_argument('--o', required=True, help='Output PNG or EPS file')
    parser.add_argument('--dist_threshold', type=float, required=True,
                        help='Distance threshold in Angstroms for calculating pairwise distances')
    parser.add_argument('--min_distance', type=float, required=True,
                        help='Minimum distance for histogram range (Angstroms)')
    parser.add_argument('--max_distance', type=float, required=True,
                        help='Maximum distance for histogram range (Angstroms)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_distance >= args.max_distance:
        print("Error: min_distance must be less than max_distance", file=sys.stderr)
        sys.exit(1)
    
    if args.dist_threshold < args.max_distance:
        print("Warning: dist_threshold is less than max_distance. "
              "Some distances in plotting range may be excluded.", file=sys.stderr)
    
    # Process STAR file
    print(f"Reading STAR file: {args.i}")
    data = starfile.read(args.i)
    
    df = None
    optics_df = None

    if isinstance(data, dict):
        # Multi-block STAR file
        optics_df = data.get('optics')
        df = data.get('particles')
        
        if df is not None:
            print(f"Using particle data block (found {len(df)} particles)")
        if optics_df is not None:
            print("Found optics data block")
    else:
        # Single dataframe - assume it's the particles block
        df = data
    
    if df is None:
        print("Error: No particle data block found in STAR file", file=sys.stderr)
        sys.exit(1)
    
    # Determine the Pixel Size (angpix)
    angpix = None
    pixel_size_cols = ['rlnImagePixelSize']

    # 1. Check Optics Table (prioritized)
    if optics_df is not None:
        for col in pixel_size_cols:
            if col in optics_df.columns:
                angpix = optics_df[col].iloc[0]
                print(f"Pixel size from optics table: {angpix:.3f} Å/px")
                break

    # 2. Check Particles Table if not found in Optics
    if angpix is None:
        for col in pixel_size_cols:
            if col in df.columns:
                angpix = df[col].iloc[0]
                print(f"Pixel size from particles table: {angpix:.3f} Å/px")
                break

    if angpix is None:
        print("Error: rlnImagePixelSize not found in optics or particles table", file=sys.stderr)
        sys.exit(1)

    # Calculate distances
    print(f"Calculating pairwise distances (threshold: {args.dist_threshold} Å)...")
    distances = calculate_pairwise_distances(df, angpix, args.dist_threshold)
    print(f"Found {len(distances)} pairwise distances below threshold")
    
    # Plot histogram
    plot_histogram(distances, args.min_distance, args.max_distance, args.o)


if __name__ == '__main__':
    main()