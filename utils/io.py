#!/usr/bin/env python
# coding: utf-8

"""
I/O utilities for STAR file processing.
@Builab 2025
"""

import os,sys
import glob
import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import starfile
from pathlib import Path

_RE = re.compile(r'^(.*_\d+)_\d+(?:\.\d+)?Apx$', re.IGNORECASE)

def sanitize_name(name: str) -> str:
    """Return sanitized name or original name if it doesn't match the pattern."""
    m = _RE.match(name)
    if m:
        return f"{m.group(1)}"
    return name

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> None:
    """
    Validate DataFrame is not None or empty, and has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    name : str
        Name of the DataFrame for error messages
    required_columns : list, optional
        List of required column names
    Returns:
        True if valid, False otherwise.        
    
    Raises
    ------
    ValueError
        If DataFrame is None, empty, or missing required columns
    """
    if df is None:
        raise ValueError(f"DataFrame is None")
        return False
        
    if df.empty:
        raise ValueError(f"DataFrame is empty")
        return False
        
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
            return False
            
    return True

def load_coordinates(
    file_path: str,
    angpix: float
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    """
    Load coordinates from STAR file into NumPy array.
    
    Args:
        file_path: Path to STAR file.
        angpix: Default pixel size if not found in file.
    
    Returns:
        Tuple of (coordinates array, tomogram name, detector pixel size).
        Returns (None, None, None) if file is invalid.
    """
    if not file_path.endswith(".star"):
        raise ValueError(f"Unsupported file format: {file_path}. Only .star files supported.")
    
    df = read_star(file_path)
    
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    if not validate_dataframe(df, ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']):
        return None, None, None
    
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    
    # Handle detector pixel size
    if 'rlnImagePixelSize' in df.columns:
        pixel_size = df['rlnImagePixelSize'].iloc[0]
    else:
        pixel_size = angpix
        print(f"  - rlnImagePixelSize not found, using --angpix: {angpix}")
    
    # Handle tomogram name (priority: rlnMicrographName > rlnTomoName > filename)
    tomo_name = df['rlnTomoName'].iloc[0]
        
    return coords, tomo_name, pixel_size


def read_star(file_path: str) -> pd.DataFrame:
    """
    Read STAR file (no optics group) into DataFrame.
    Take care of rlnTomoName from either rlnMicrographName
    Take care of rlnImagePixelSize from either rlnDetectorPixelSize
    
    Args:
        file_path: Path to STAR file.
    
    Returns:
        DataFrame containing STAR file data.
    """
    df = starfile.read(file_path)
    
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']

    # --- Step 1: Rename column if needed ---
    if 'rlnMicrographName' in df.columns and 'rlnTomoName' not in df.columns:
        df = df.rename(columns={'rlnMicrographName': 'rlnTomoName'})
        print('Rename rlnMicrographName to rlnTomoName')

    # **HIGHLIGHTED CHANGE: Add default rlnTomoName if missing**
    if 'rlnTomoName' not in df.columns:
        df['rlnTomoName'] = 'TS_001'
        print('rlnTomoName column not found - added with default value: TS_001')
        
    # --- Step 2: Remove trailing .tomostar first ---    
    if 'rlnTomoName' in df.columns:
    	# Sanitize tomoName
        df['rlnTomoName'] = df['rlnTomoName'].str.replace(r'\.tomostar$', '', case=False, regex=True)
        df['rlnTomoName'] = df['rlnTomoName'].apply(sanitize_name)
        
    # --- Step 3: Unify rlnImagePixelSize ---    
    if 'rlnDetectorPixelSize' in df.columns and 'rlnImagePixelSize' not in df.columns:
        df = df.rename(columns={'rlnDetectorPixelSize': 'rlnImagePixelSize'})
        print('Rename rlnDetectorPixelSize to rlnImagePixelSize')

    # STAR files round-trip rlnHelicalTubeID as float (e.g. "1.000000"); normalize
    # back to int so downstream code (e.g. connect's "{:4d}" formatting) works the
    # same whether the DataFrame came fresh from fitting or was reloaded from disk.
    if 'rlnHelicalTubeID' in df.columns:
        df['rlnHelicalTubeID'] = df['rlnHelicalTubeID'].astype(int)

    print(f'Read {file_path} and sanitize')

    return df


def write_star(df: pd.DataFrame, file_path: str, overwrite: bool = True) -> None:
    """
    Write DataFrame to STAR file.

    Args:
        df: DataFrame to write.
        file_path: Output file path.
        overwrite: Whether to overwrite existing file.
    """
    starfile.write(df, file_path, overwrite=overwrite)


def write_copick(
    df: pd.DataFrame,
    output_dir: str,
    object_name: str = "microtubule",
    user_id: str = "mtfit",
    session_id: str = "0",
) -> None:
    """
    Write particle picks to copick JSON format (one file per tomogram).

    Coordinates are converted from pixels to Ångström using rlnImagePixelSize
    (or rlnDetectorPixelSize as fallback). RELION ZYZ Euler angles are converted
    to 4x4 transformation matrices following the copick convention (inverted
    rotation, matching relion_df_to_picks). rlnHelicalTubeID is preserved as
    instance_id.

    Args:
        df: DataFrame with rlnCoordinateX/Y/Z, rlnAngleRot/Tilt/Psi,
            rlnHelicalTubeID, rlnTomoName, rlnImagePixelSize or rlnDetectorPixelSize.
        output_dir: Directory to write JSON files.
        object_name: Copick pickable object name.
        user_id: Copick user ID.
        session_id: Copick session ID.
    """
    from copick.models import CopickPicksFile, CopickPoint, CopickLocation
    from scipy.spatial.transform import Rotation

    os.makedirs(output_dir, exist_ok=True)

    tomo_col = next((c for c in ('rlnTomoName', 'rlnMicrographName') if c in df.columns), None)
    if tomo_col is None:
        raise ValueError("DataFrame must have rlnTomoName or rlnMicrographName column")

    pixel_col = next((c for c in ('rlnImagePixelSize', 'rlnDetectorPixelSize') if c in df.columns), None)
    has_angles = {'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'}.issubset(df.columns)

    for tomo_name, tomo_df in df.groupby(tomo_col):
        pixel_size = float(tomo_df[pixel_col].iloc[0]) if pixel_col else 1.0

        points = []
        for _, row in tomo_df.iterrows():
            x_ang = float(row['rlnCoordinateX']) * pixel_size
            y_ang = float(row['rlnCoordinateY']) * pixel_size
            z_ang = float(row['rlnCoordinateZ']) * pixel_size

            if has_angles:
                angles = [row['rlnAngleRot'], row['rlnAngleTilt'], row['rlnAnglePsi']]
                R = Rotation.from_euler("ZYZ", angles, degrees=True).inv().as_matrix()
            else:
                R = np.eye(3)

            transform = [
                [R[0, 0], R[0, 1], R[0, 2], x_ang],
                [R[1, 0], R[1, 1], R[1, 2], y_ang],
                [R[2, 0], R[2, 1], R[2, 2], z_ang],
                [0.0,     0.0,     0.0,     1.0  ],
            ]

            points.append(CopickPoint(
                location=CopickLocation(x=x_ang, y=y_ang, z=z_ang),
                transformation_=transform,
                instance_id=int(row.get('rlnHelicalTubeID', 0)),
                score=float(row.get('rlnMaxValueProbDistribution', 0.0)),
            ))

        picks_file = CopickPicksFile(
            pickable_object_name=object_name,
            user_id=user_id,
            session_id=session_id,
            run_name=str(tomo_name),
            voxel_spacing=pixel_size,
            unit="angstrom",
            points=points,
        )

        out_file = os.path.join(output_dir, f"{tomo_name}_{object_name}.json")
        with open(out_file, 'w') as f:
            f.write(picks_file.model_dump_json(indent=2))

        print(f"  Wrote {len(points)} picks → {out_file}")


def convert_to_relionwarp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert STAR file DataFrame to relionwarp format.
    
    Args:
        df: Input DataFrame with particle data.
    
    Returns:
        DataFrame in relionwarp format.
    """
    # Create output DataFrame with required columns
    output_df = pd.DataFrame()
    
    # Handle rlnTomoName - add .tomostar extension if not present
    if 'rlnTomoName' in df.columns:
    	# Sanitize tomoName
        output_df['rlnTomoName'] = df['rlnTomoName'].apply(
            lambda x: x if x.endswith('.tomostar') else f"{x}.tomostar"
        )
    else:
        raise ValueError("Input STAR file missing rlnTomoName column")
    
    # Copy coordinate columns
    for col in ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']:
        if col in df.columns:
            output_df[col] = df[col]
        else:
            raise ValueError(f"Input STAR file missing {col} column")
    
    # Copy angle columns
    for col in ['rlnAngleTilt', 'rlnAnglePsi', 'rlnAngleRot']:
        if col in df.columns:
            output_df[col] = df[col]
        else:
            raise ValueError(f"Input STAR file missing {col} column")
    
    # Set origin columns to 0
    output_df['rlnOriginXAngst'] = 0.0
    output_df['rlnOriginYAngst'] = 0.0
    output_df['rlnOriginZAngst'] = 0.0
    
    # Copy rlnHelicalTubeID if present
    if 'rlnHelicalTubeID' in df.columns:
        output_df['rlnHelicalTubeID'] = df['rlnHelicalTubeID']
    
    # Handle pixel size: prefer rlnImagePixelSize, fallback to rlnDetectorPixelSize
    if 'rlnImagePixelSize' in df.columns:
        output_df['rlnImagePixelSize'] = df['rlnImagePixelSize']
    elif 'rlnDetectorPixelSize' in df.columns:
        output_df['rlnImagePixelSize'] = df['rlnDetectorPixelSize']
    else:
        raise ValueError("Input STAR file missing both rlnImagePixelSize and rlnDetectorPixelSize columns")
    
    return output_df



def combine_star_files(input_patterns: list, output_file: str) -> None:
    """
    Combine multiple STAR files into a single relionwarp file.
    
    Args:
        input_patterns: List of file paths or glob patterns to match input STAR files.
        output_file: Output file path for combined relionwarp STAR file.
    """
    # Collect all input files (expanding patterns if needed)
    input_files = []
    for pattern in input_patterns:
        # Check if it's an exact file path
        if Path(pattern).exists():
            input_files.append(pattern)
        else:
            # Try as a glob pattern
            matched_files = glob.glob(pattern)
            if matched_files:
                input_files.extend(matched_files)
            else:
                print(f"Warning: No files found for pattern: {pattern}")
    
    # Remove duplicates and sort
    input_files = sorted(set(input_files))
    
    if not input_files:
        print(f"Error: No input files found")
        sys.exit(1)
    
    print(f"Found {len(input_files)} files to process")
    
    # List to store all DataFrames
    all_dfs = []
    
    # Read and convert each file
    for file_path in input_files:
        print(f"Processing: {file_path}")
        try:
            df = read_star(file_path)
            
            # Validate required columns
            required_cols = [
                'rlnTomoName',
                'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'
            ]
            validate_dataframe(df, required_cols)
            
            # Check for at least one pixel size column
            if 'rlnImagePixelSize' not in df.columns and 'rlnDetectorPixelSize' not in df.columns:
                raise ValueError("Missing both rlnImagePixelSize and rlnDetectorPixelSize columns")
            
            # Convert to relionwarp format
            converted_df = convert_to_relionwarp(df)
            all_dfs.append(converted_df)
            print(f"  - Added {len(converted_df)} particles")
            
        except Exception as e:
            print(f"  - Error processing {file_path}: {e}")
            print(f"  - Skipping this file...")
            continue
    
    if not all_dfs:
        print("Error: No valid STAR files were processed")
        sys.exit(1)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal particles combined: {len(combined_df)}")
    
    
    # Write output file
    print(f"Writing output to: {output_file}")
    write_star(combined_df, output_file, overwrite=True)
    print("Done!")

