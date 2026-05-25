#!/usr/bin/env python
# coding: utf-8

"""
A quick star file viewer
If there is rlnHelicalTubeID, then it will display as Tube.
Supports overlaying multiple STAR files with --overlay option.
"""

import argparse
import sys
import os

# --- Module Import Setup ---
# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
 
    
# Import the updated visualization function
from utils.view import visualize_star_df, visualize_overlay_star_dfs
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
        epilog="Usage examples:\n"
               "  python script/view_star.py particles.star\n"
               "  python script/view_star.py --overlay file2.star file1.star\n"
               "  python script/view_star.py --overlay file2.star file3.star file1.star -o output.html",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'star_file',
        type=str,
        help="Path to the input STAR file (e.g., coordinates.star)."
    )
    parser.add_argument(
        '--overlay',
        type=str,
        default=None,
        help="An additional STAR file to overlay on top of the main file (shown with transparency). "
             "Example: --overlay file2.star"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help="Optional path to save the plot as an interactive HTML file (e.g., plot.html). "
             "If not provided, the plot attempts to open in a browser."
    )

    args = parser.parse_args()
    filepath = args.star_file
    overlay_file = args.overlay
    output_path = args.output

    # Check if main file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)

    # If overlay mode, process multiple files
    if overlay_file:
        print(f"--- Overlay Mode: Loading 2 STAR files ---")
        
        # Main file
        print(f"Loading main file: {os.path.basename(filepath)}")
        try:
            main_df = read_star(filepath)
            if main_df is None or main_df.empty:
                print(f"Error: Main STAR file could not be read or was empty.")
                sys.exit(1)
        except Exception as e:
            print(f"Error reading main file: {e}")
            sys.exit(1)
        
        # Overlay file
        if not os.path.exists(overlay_file):
            print(f"Error: Overlay file not found at '{overlay_file}'")
            sys.exit(1)
            
        print(f"Loading overlay file: {os.path.basename(overlay_file)}")
        try:
            overlay_df = read_star(overlay_file)
            if overlay_df is None or overlay_df.empty:
                print(f"Error: Overlay STAR file could not be read or was empty.")
                sys.exit(1)
        except Exception as e:
            print(f"Error reading overlay file: {e}")
            sys.exit(1)
        
        print(f"--- Both files loaded successfully ---")
        visualize_overlay_star_dfs(
            main_df, 
            overlay_df, 
            os.path.basename(filepath),
            os.path.basename(overlay_file),
            output_path=output_path
        )
        
    else:
        # Single file mode (original behavior)
        print(f"--- Loading STAR file: {os.path.basename(filepath)} ---")
        
        try:
            df = read_star(filepath)
            
            if df is None or df.empty:
                print("Error: STAR file could not be read or was empty.")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error during file reading: {e}")
            sys.exit(1)
            
        visualize_star_df(df, os.path.basename(filepath), output_path=output_path)

if __name__ == '__main__':
    main()