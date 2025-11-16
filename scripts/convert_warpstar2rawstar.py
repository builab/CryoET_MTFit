#!/usr/bin/env python3
"""
Convert mtstar to rawstar (no optics) ready for line fitting.
Quick & dirty. Will change in the future with rlnHelicalTubeID intact.

Usage:
    convert_warpstar2rawstar.py --i input.star --o output.star --rlnTomoName TS_001.tomostar
    
Builab@McGill 2025
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import starfile
from rich.console import Console

console = Console()


def extract_tomo_df(df_in: dict, tomo_name: str) -> pd.DataFrame:
    """
    Extract particles for a specific tomogram from a RELION STAR file.
    
    Args:
        df_in: Dictionary containing 'particles' and 'optics' DataFrames
        tomo_name: Name of the tomogram to extract (e.g., 'TS_001')
        
    Returns:
        DataFrame containing particles for the specified tomogram with:
        - Coordinates corrected for origin shifts
        - Only essential columns retained
        - rlnTomoName normalized to base name without .tomostar extension
        
    Raises:
        ValueError: If required blocks are missing from input
        KeyError: If required columns are missing
    """
    # Validate input structure
    if not all(key in df_in for key in ('particles', 'optics')):
        raise ValueError(
            "Expected RELION 3.1+ style STAR file containing 'particles' and 'optics' blocks"
        )
    
    # Check for required columns
    required_particle_cols = ['rlnOpticsGroup', 'rlnTomoName', 
                              'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    for col in required_particle_cols:
        if col not in df_in['particles'].columns:
            raise KeyError(f"'{col}' column not found in particles block")
    
    if 'rlnOpticsGroup' not in df_in['optics'].columns:
        raise KeyError("'rlnOpticsGroup' column not found in optics block")
    
    if 'rlnImagePixelSize' not in df_in['optics'].columns:
        raise KeyError("'rlnImagePixelSize' column not found in optics block")
    
    # Merge particles with optics information
    df_combined = df_in['particles'].merge(
        df_in['optics'], 
        on='rlnOpticsGroup',
        how='left'
    )
    
    # Normalize tomo_name (remove .tomostar extension if present)
    tomo_base_name = tomo_name.replace('.tomostar', '')
    
    # Filter for specific tomogram - check both with and without .tomostar extension
    mask = (df_combined['rlnTomoName'] == tomo_base_name) | \
           (df_combined['rlnTomoName'] == f"{tomo_base_name}.tomostar")
    df_out = df_combined[mask].copy()
    
    if df_out.empty:
        console.log(
            f"[yellow]Warning: No particles found for tomogram '{tomo_name}' "
            f"(searched for '{tomo_base_name}' and '{tomo_base_name}.tomostar')[/yellow]"
        )
        return df_out
    
    # Correct coordinates for origin shifts
    origin_cols = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
    coord_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    
    for origin_col, coord_col in zip(origin_cols, coord_cols):
        if origin_col in df_out.columns:
            df_out[coord_col] = df_out[coord_col] - (df_out[origin_col] / df_out['rlnImagePixelSize'])
        else:
            console.log(f"[yellow]Warning: '{origin_col}' not found, skipping origin correction[/yellow]")
    
    # Normalize rlnTomoName to base name (without .tomostar)
    df_out['rlnTomoName'] = tomo_base_name
    
    # Add rlnLCCmax column with value of 1
    df_out['rlnLCCmax'] = 1.0
    
    # Select only required output columns
    output_columns = [
        'rlnTomoName',
        'rlnCoordinateX',
        'rlnCoordinateY', 
        'rlnCoordinateZ',
        'rlnAngleRot',
        'rlnAngleTilt',
        'rlnAnglePsi',
        'rlnImagePixelSize',
        'rlnLCCmax'
    ]
    
    # Keep only columns that exist in the dataframe
    available_columns = [col for col in output_columns if col in df_out.columns]
    missing_columns = [col for col in output_columns if col not in df_out.columns]
    
    if missing_columns:
        console.log(f"[yellow]Warning: Missing columns in output: {', '.join(missing_columns)}[/yellow]")
    
    df_out = df_out[available_columns].copy()
    
    return df_out


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert warp star to raw star for 1 tomogram',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --i input.star --o output.star --rlnTomoName TS_001.tomostar
        """
    )
    parser.add_argument(
        '--i', 
        type=str, 
        required=True,
        metavar='INPUT',
        help='Input warp star file path'
    )
    parser.add_argument(
        '--o', 
        type=str, 
        required=True,
        metavar='OUTPUT',
        help='Output raw star file path'
    )
    parser.add_argument(
        '--rlnTomoName', 
        type=str, 
        required=True,
        metavar='TOMO_NAME',
        help='Tomogram name to extract'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.i)
    if not input_path.exists():
        console.log(f"[bold red]Error: Input file '{args.i}' not found[/bold red]")
        sys.exit(1)
    
    try:
        console.log(f"[cyan]Converting RELION Warp file to Raw Star File[/cyan]")
        console.log(f"Reading {args.i}...")
        
        # Read input STAR file
        star = starfile.read(args.i, always_dict=True)
        console.log(f"[green]✓[/green] {args.i} read successfully")
        
        # Extract tomogram-specific particles
        console.log(f"Extracting particles for tomogram: {args.rlnTomoName}")
        df_tomo = extract_tomo_df(star, args.rlnTomoName)
        console.log(f"[green]✓[/green] Found {len(df_tomo)} particles")
        
        # Write output file
        console.log(f"Writing output to {args.o}...")
        starfile.write({'particles': df_tomo}, args.o, overwrite=True)
        console.log(f"[green]✓[/green] {args.o} written successfully")
        
    except ValueError as e:
        console.log(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except KeyError as e:
        console.log(f"[bold red]Error: Missing required column {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.log(f"[bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()