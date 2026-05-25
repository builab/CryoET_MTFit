#!/usr/bin/env python
# coding: utf-8

"""
Combine multiple STAR files matching a pattern and convert to relionwarp format.
@Builab 2025
"""

import argparse
import glob
import sys, os
from pathlib import Path

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.io import combine_star_files

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description='Combine multiple STAR files and convert to relionwarp format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a pattern:
  %(prog)s Pattern*.star --output relionwarp.star
  %(prog)s "TS_*.star" --output combined_output.star
  
  # Using explicit file list:
  %(prog)s --output combine.star file1.star file2.star file3.star
  
  # Mix of patterns and files:
  %(prog)s Pattern*.star file1.star --output relionwarp.star
        """
    )
    
    parser.add_argument(
        'inputs',
        type=str,
        nargs='*',
        help='Input STAR files or glob patterns (e.g., Pattern*.star, file1.star, file2.star)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output STAR file path'
    )
    
    args = parser.parse_args()
    
    # Check if inputs were provided
    if not args.inputs:
        parser.error("At least one input file or pattern is required")
    
    # Run the combination
    combine_star_files(args.inputs, args.output)


if __name__ == '__main__':
    main()