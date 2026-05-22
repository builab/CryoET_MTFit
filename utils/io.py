#!/usr/bin/env python
# coding: utf-8

"""
I/O utilities for STAR file processing.
@Builab 2025
"""

import json
import os
import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import starfile

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
    
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    if not validate_dataframe(df, ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']):
        return None, None, None
    
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    
    # Handle detector pixel size
    if 'rlnDetectorPixelSize' in df.columns:
        detector_pixel_size = df['rlnDetectorPixelSize'].iloc[0]
    else:
        detector_pixel_size = angpix
        print(f"  - rlnDetectorPixelSize not found, using --angpix: {angpix}")
    
    # Handle tomogram name (priority: rlnMicrographName > rlnTomoName > filename)
    tomo_name = None
    
    if 'rlnMicrographName' in df.columns:
        tomo_name = df['rlnMicrographName'].iloc[0]
    elif 'rlnTomoName' in df.columns:
        tomo_name = df['rlnTomoName'].iloc[0]
        if tomo_name.endswith('.tomostar'):
            tomo_name = tomo_name[:-9]
            print(f"  - Removed .tomostar extension from rlnTomoName: {tomo_name}")
    
    if tomo_name is None:
        match = re.match(r"^(.+?_\d{2,3})", os.path.basename(file_path))
        if match:
            tomo_name = match.group(1)
        else:
            tomo_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"  - No rlnMicrographName or rlnTomoName found, using modified filename: {tomo_name}")
    
    return coords, tomo_name, detector_pixel_size


def read_star(file_path: str) -> pd.DataFrame:
    """
    Read STAR file into DataFrame.
    
    Args:
        file_path: Path to STAR file.
    
    Returns:
        DataFrame containing STAR file data.
    """
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    return df


def write_star(df: pd.DataFrame, file_path: str, overwrite: bool = True) -> None:
    """
    Write DataFrame to STAR file.

    Args:
        df: DataFrame to write.
        file_path: Output file path.
        overwrite: Whether to overwrite existing file.
    """
    out = df.copy()
    # ArtiaX requires rlnImagePixelSize; mirror from rlnDetectorPixelSize if present
    if 'rlnDetectorPixelSize' in out.columns and 'rlnImagePixelSize' not in out.columns:
        out['rlnImagePixelSize'] = out['rlnDetectorPixelSize']
    starfile.write(out, file_path, overwrite=overwrite)


def _euler_zyz_to_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """Convert RELION ZYZ Euler angles to 3x3 rotation matrix R = Rz(psi)·Ry(tilt)·Rz(rot)."""
    rot  = np.radians(rot_deg)
    tilt = np.radians(tilt_deg)
    psi  = np.radians(psi_deg)

    Rz1 = np.array([[ np.cos(rot), -np.sin(rot), 0],
                    [ np.sin(rot),  np.cos(rot), 0],
                    [ 0,            0,            1]])
    Ry  = np.array([[ np.cos(tilt), 0, np.sin(tilt)],
                    [ 0,            1, 0            ],
                    [-np.sin(tilt), 0, np.cos(tilt)]])
    Rz2 = np.array([[ np.cos(psi), -np.sin(psi), 0],
                    [ np.sin(psi),  np.cos(psi), 0],
                    [ 0,            0,            1]])
    return Rz2 @ Ry @ Rz1


def write_copick(
    df: pd.DataFrame,
    output_dir: str,
    object_name: str = "microtubule",
    user_id: str = "mtfit",
    session_id: str = "0",
) -> None:
    """
    Write particle picks to copick JSON format (one file per tomogram).

    Coordinates are converted from pixels to Ångström using rlnDetectorPixelSize.
    RELION ZYZ Euler angles are converted to 4x4 homogeneous transformation matrices.
    rlnHelicalTubeID is preserved as instance_id.

    Args:
        df: DataFrame with rlnCoordinateX/Y/Z, rlnAngleRot/Tilt/Psi,
            rlnHelicalTubeID, rlnTomoName or rlnMicrographName, rlnDetectorPixelSize.
        output_dir: Directory to write JSON files.
        object_name: Copick pickable object name.
        user_id: Copick user ID.
        session_id: Copick session ID.
    """
    os.makedirs(output_dir, exist_ok=True)

    tomo_col = next((c for c in ('rlnTomoName', 'rlnMicrographName') if c in df.columns), None)
    if tomo_col is None:
        raise ValueError("DataFrame must have rlnTomoName or rlnMicrographName column")

    has_pixel_size = 'rlnDetectorPixelSize' in df.columns

    for tomo_name, tomo_df in df.groupby(tomo_col):
        pixel_size = float(tomo_df['rlnDetectorPixelSize'].iloc[0]) if has_pixel_size else 1.0

        points = []
        for _, row in tomo_df.iterrows():
            x_ang = float(row['rlnCoordinateX']) * pixel_size
            y_ang = float(row['rlnCoordinateY']) * pixel_size
            z_ang = float(row['rlnCoordinateZ']) * pixel_size

            R = _euler_zyz_to_matrix(
                float(row.get('rlnAngleRot',  0.0)),
                float(row.get('rlnAngleTilt', 0.0)),
                float(row.get('rlnAnglePsi',  0.0)),
            )
            transform = [
                [R[0, 0], R[0, 1], R[0, 2], x_ang],
                [R[1, 0], R[1, 1], R[1, 2], y_ang],
                [R[2, 0], R[2, 1], R[2, 2], z_ang],
                [0.0,     0.0,     0.0,     1.0  ],
            ]

            points.append({
                "location":       {"x": x_ang, "y": y_ang, "z": z_ang},
                "transformation": transform,
                "instance_id":    int(row.get('rlnHelicalTubeID', 0)),
                "score":          float(row.get('rlnMaxValueProbDistribution', 0.0)),
            })

        picks = {
            "pickable_object_name": object_name,
            "user_id":              user_id,
            "session_id":           session_id,
            "run_name":             str(tomo_name),
            "voxel_spacing":        pixel_size,
            "unit":                 "angstrom",
            "points":               points,
        }

        out_file = os.path.join(output_dir, f"{tomo_name}_{object_name}.json")
        with open(out_file, 'w') as f:
            json.dump(picks, f, indent=2)

        print(f"  Wrote {len(points)} picks → {out_file}")

