#!/usr/bin/env python
# coding: utf-8

"""
A quick star file viewer
If there is rlnHelicalTubeID, then it will display as Tube.
"""

import argparse
import sys
import os

# --- Module Import Setup ---
# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
 
    
# Import the updated visualization function
from utils.view import visualize_star_df
# Import the file reading utility from the I/O module
from utils.io import read_star 
    
# --------------------------

def main():
    """
    Main function for the command-line interface.
    Parses the STAR file path argument, reads the file, and initiates visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize 3D coordinates from a STAR file.",
        epilog="Usage example: python script/view_star.py particles.star"
    )
    parser.add_argument(
        'star_file',
        type=str,
        help="Path to the input STAR file (e.g., coordinates.star)."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help="Optional path to save the plot as an interactive HTML file (e.g., plot.html). If not provided, the plot attempts to open in a browser."
    )

    args = parser.parse_args()
    filepath = args.star_file
    output_path = args.output

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)

    print(f"--- Loading STAR file: {os.path.basename(filepath)} ---")
    
    try:
        # Read the file using the custom I/O function
        df = read_star(filepath)
        
        if df is None or df.empty:
            print("Error: STAR file could not be read or was empty.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during file reading: {e}")
        sys.exit(1)
        
    # NOTE: The 'visualize_star_df' function must be updated to accept and handle 
    # the 'output_path' argument for this to fully work.
    visualize_star_df(df, os.path.basename(filepath), output_path=output_path)

if __name__ == '__main__':
    main()