#!/usr/bin/env python3
"""
Connect broken helical tubes using trajectory extrapolation.

This module provides functionality to:
- Detect potential connections between tube segments
- Extrapolate tube trajectories using polynomial fitting
- Merge connected tubes and resample them

@Builab 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# Constants
DEFAULT_MAX_EXTRAPOLATION_STEPS = 50
INTEGRATION_STEP_DIVISOR = 50.0
MIN_POINTS_FOR_FITTING = 2


class TubeInfo:
    """Container for helical tube metadata and coordinates."""
    
    def __init__(
        self,
        tube_id: int,
        coords: np.ndarray,
        tomo_name: str = "Unknown",
        angpix: Optional[float] = None
    ):
        self.tube_id = tube_id
        self.coords = coords  # In Angstroms
        self.n_points = len(coords)
        self.tomo_name = tomo_name
        self.angpix = angpix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backwards compatibility."""
        return {
            'tube_id': self.tube_id,
            'coords': self.coords,
            'n_points': self.n_points,
            'tomo_name': self.tomo_name,
            'angpix': self.angpix
        }

def extract_tube_info(
    df: pd.DataFrame,
    tube_id: int,
    angpix: float
) -> Optional[TubeInfo]:
    """
    Extract coordinates and metadata for a specific helical tube.
    
    Args:
        df: DataFrame containing particle data.
        tube_id: Unique tube identifier.
        angpix: Pixel size in Angstroms for coordinate conversion.
    
    Returns:
        TubeInfo object or None if tube has insufficient points.
    """
    tube_data = df[df['rlnHelicalTubeID'] == tube_id].copy()
    tube_data = tube_data.sort_index()
    
    # Convert coordinates from pixels to Angstroms
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    # Check for NaN or invalid values
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        print(f"  Warning: Tube {tube_id} contains NaN or infinite coordinates. Skipping.")
        return None
    
    if len(coords) < MIN_POINTS_FOR_FITTING:
        return None
    
    # Extract metadata with safe defaults
    tomo_name = (
        tube_data['rlnTomoName'].iloc[0]
        if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty
        else "Unknown"
    )
    
    return TubeInfo(tube_id, coords, tomo_name, angpix)


def calculate_average_step_size(coords: np.ndarray, n_points: int) -> float:
    """
    Calculate average distance between consecutive points.
    
    Args:
        coords: Coordinate array (N x 3).
        n_points: Number of points to use from the end.
    
    Returns:
        Average step size in Angstroms.
    """
    seed_coords = coords[-n_points:]
    step_distances = np.linalg.norm(seed_coords[1:] - seed_coords[:-1], axis=1)
    
    if len(step_distances) == 0:
        # Fallback to last two points
        return np.linalg.norm(coords[-1] - coords[-2]) if len(coords) >= 2 else 0.0
    
    return np.mean(step_distances)


def extrapolate_trajectory(
    coords: np.ndarray,
    min_seed: int,
    dist_extrapolate: float,
    poly_order: int
) -> Optional[np.ndarray]:
    """
    Fit polynomial to seed points and extrapolate trajectory forward. Fix NaN error
    
    Args:
        coords: Coordinate array (N x 3) in Angstroms.
        min_seed: Number of points to use for polynomial fitting.
        dist_extrapolate: Total distance to extrapolate in Angstroms.
        poly_order: Polynomial order for fitting.
    
    Returns:
        Extrapolated coordinates (M x 3) or None if fitting fails.
    """
    n_points = len(coords)
    
    # Validate input
    if n_points < poly_order + 1 or min_seed < poly_order + 1:
        return None
    
    # Check for NaN values
    if np.any(np.isnan(coords)):
        return None
    
    # Calculate number of extrapolation steps
    avg_step = calculate_average_step_size(coords, min_seed)
    
    if avg_step <= 0:
        n_steps = 10  # Default fallback
    else:
        n_steps = max(1, int(np.ceil(dist_extrapolate / avg_step)))
    
    n_extrapolate = min(n_steps, DEFAULT_MAX_EXTRAPOLATION_STEPS)
    
    # Prepare fitting parameters
    seed_start_idx = n_points - min_seed
    t_fit = np.arange(seed_start_idx, n_points)
    t_extrapolate = np.arange(n_points, n_points + n_extrapolate)
    
    extrapolated_coords = np.zeros((n_extrapolate, 3))
    
    # Fit and extrapolate each dimension independently
    for dim in range(3):
        y_fit = coords[seed_start_idx:, dim]
        
        # Double-check lengths match (defensive programming)
        if len(t_fit) != len(y_fit):
            return None
        
        try:
            coeffs = np.polyfit(t_fit, y_fit, poly_order)
            extrapolated_coords[:, dim] = np.polyval(coeffs, t_extrapolate)
        except (np.linalg.LinAlgError, ValueError):
            return None
    
    return extrapolated_coords


def detect_trajectory_overlap(
    extrapolated_coords: Optional[np.ndarray],
    target_coords: np.ndarray,
    overlap_threshold: float
) -> Tuple[bool, float]:
    """
    Check if extrapolated trajectory overlaps with target segment.
    
    Args:
        extrapolated_coords: Extrapolated trajectory coordinates.
        target_coords: Target segment coordinates to check against.
        overlap_threshold: Maximum distance for overlap detection in Angstroms.
    
    Returns:
        Tuple of (overlap_detected, minimum_distance).
    """
    if extrapolated_coords is None or len(target_coords) == 0:
        return False, float('inf')
    
    # Compute pairwise distances between all extrapolated and target points
    dist_matrix = np.linalg.norm(
        extrapolated_coords[:, None, :] - target_coords[None, :, :],
        axis=2
    )
    
    min_distance = np.min(dist_matrix)
    overlap_detected = min_distance <= overlap_threshold
    
    return overlap_detected, min_distance


def assess_connection_compatibility(
    tube1: TubeInfo,
    tube2: TubeInfo,
    end1: str,
    end2: str,
    overlap_threshold: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order: int
) -> Tuple[bool, float, bool, bool, float]:
    """
    Assess if two tube segments can be connected via trajectory extrapolation.
    
    Args:
        tube1: First tube information.
        tube2: Second tube information.
        end1: Connection point on tube1 ('start' or 'end').
        end2: Connection point on tube2 ('start' or 'end').
        overlap_threshold: Distance threshold for overlap in Angstroms.
        min_seed: Number of points for polynomial fitting.
        dist_extrapolate: Extrapolation distance in Angstroms.
        poly_order: Polynomial order for fitting.
    
    Returns:
        Tuple containing:
        - can_connect: Whether connection is possible
        - min_distance: Minimum distance between trajectories
        - reverse1: Whether tube1 needs reversal
        - reverse2: Whether tube2 needs reversal
        - end_to_end_distance: Simple endpoint distance
    """
    coords1 = tube1.coords
    coords2 = tube2.coords
    
    # Calculate direct endpoint distance
    endpoint1 = coords1[-1] if end1 == 'end' else coords1[0]
    endpoint2 = coords2[0] if end2 == 'start' else coords2[-1]
    end_to_end_distance = np.linalg.norm(endpoint1 - endpoint2)
    
    # Prepare tube1 for extrapolation (reverse if connecting from start)
    if end1 == 'end':
        fit_coords = coords1
        reverse1 = False
    else:
        fit_coords = np.flipud(coords1)
        reverse1 = True
    
    # Extrapolate tube1's trajectory
    extrapolated = extrapolate_trajectory(
        fit_coords, min_seed, dist_extrapolate, poly_order
    )
    
    if extrapolated is None:
        return False, float('inf'), False, False, end_to_end_distance
    
    # Prepare target region on tube2
    target_buffer_size = min_seed * 2
    
    if end2 == 'start':
        target_coords = coords2[:target_buffer_size]
        reverse2 = False
    else:
        target_coords = np.flipud(coords2[-target_buffer_size:])
        reverse2 = True
    
    # Check for trajectory overlap
    overlap_detected, min_distance = detect_trajectory_overlap(
        extrapolated, target_coords, overlap_threshold
    )
    
    if not overlap_detected:
        return False, min_distance, False, False, end_to_end_distance
    
    return True, min_distance, reverse1, reverse2, end_to_end_distance


def find_tube_connections(
    df: pd.DataFrame,
    angpix: float,
    overlap_threshold: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order: int
) -> List[Dict[str, Any]]:
    """
    Identify all possible connections between tube segments.
    
    Args:
        df: DataFrame containing particle data.
        angpix: Pixel size in Angstroms.
        overlap_threshold: Overlap detection threshold in Angstroms.
        min_seed: Number of points for polynomial fitting.
        dist_extrapolate: Extrapolation distance in Angstroms.
        poly_order: Polynomial order for fitting for seed.
    
    Returns:
        List of connection dictionaries with metadata.
    """
    tube_ids = df['rlnHelicalTubeID'].unique()
    
    # Extract information for all tubes
    tubes = {}
    for tube_id in tube_ids:
        tube_info = extract_tube_info(df, tube_id, angpix)
        if tube_info is not None:
            tubes[tube_id] = tube_info
    
    # Test all possible connection configurations
    connection_types = [
        ('end', 'start', 'Line1_end → Line2_start'),
        ('end', 'end', 'Line1_end → Line2_end (reverse Line2)'),
        ('start', 'start', 'Line1_start → Line2_start (reverse Line1)'),
        ('start', 'end', 'Line1_start → Line2_end (reverse both)'),
    ]
    
    connections = []
    tube_ids_list = list(tubes.keys())
    
    for i, tube_id1 in enumerate(tube_ids_list):
        for tube_id2 in tube_ids_list[i + 1:]:
            best_connection = None
            best_score = float('inf')
            
            # Try all connection configurations
            for end1, end2, description in connection_types:
                can_connect, min_dist, rev1, rev2, simple_dist = assess_connection_compatibility(
                    tubes[tube_id1],
                    tubes[tube_id2],
                    end1, end2,
                    overlap_threshold,
                    min_seed,
                    dist_extrapolate,
                    poly_order
                )
                if can_connect and min_dist < best_score:
                    best_score = min_dist
                    best_connection = {
                        'tube_id1': tube_id1,
                        'tube_id2': tube_id2,
                        'min_overlap_dist': min_dist,
                        'simple_end_distance': simple_dist,
                        'reverse1': rev1,
                        'reverse2': rev2,
                        'connection_type': description,
                        'n_points1': tubes[tube_id1].n_points,
                        'n_points2': tubes[tube_id2].n_points
                    }
            
            if best_connection is not None:
                connections.append(best_connection)
    
    return connections


class UnionFind:
    """Union-Find data structure for grouping connected tubes."""
    
    def __init__(self):
        self.parent = {}
    
    def find(self, x: int) -> int:
        """Find root of element with path compression."""
        if x not in self.parent:
            self.parent[x] = x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        """Unite two sets."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            self.parent[root_y] = root_x
    
    def get_groups(self, elements: List[int]) -> Dict[int, List[int]]:
        """Group elements by their root."""
        groups = {}
        for element in elements:
            root = self.find(element)
            if root not in groups:
                groups[root] = []
            groups[root].append(element)
        return groups


def merge_connected_tubes(
    df: pd.DataFrame,
    connections: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Merge tubes identified as connected using Union-Find algorithm.
    
    Args:
        df: DataFrame containing particle data.
        connections: List of connection dictionaries from find_tube_connections.
    
    Returns:
        Tuple containing:
        - Merged DataFrame with updated tube IDs
        - Mapping from old tube IDs to new tube IDs
    """
    uf = UnionFind()
    
    # Build connected components
    for conn in connections:
        uf.union(conn['tube_id1'], conn['tube_id2'])
    
    # Group tubes by connected components
    tube_ids = df['rlnHelicalTubeID'].unique()
    groups = uf.get_groups(tube_ids.tolist())
    
    # Create mapping to new consecutive tube IDs
    tube_id_mapping = {}
    for new_id, (_, tube_list) in enumerate(groups.items(), start=1):
        for tube_id in tube_list:
            tube_id_mapping[tube_id] = new_id
    
    # Apply mapping to DataFrame
    df_merged = df.copy()
    df_merged['rlnHelicalTubeID'] = df_merged['rlnHelicalTubeID'].map(tube_id_mapping)
    
    return df_merged, tube_id_mapping


def fit_and_resample_tube(
    tube_data: pd.DataFrame,
    poly_order: int,
    sample_step: float,
    angpix: float,
    tube_id: int
) -> List[Dict[str, Any]]:
    """
    Fit polynomial curve to tube and resample at regular intervals.
    
    Args:
        tube_data: DataFrame with tube particle data (coordinates in pixels).
        poly_order: Polynomial order for curve fitting.
        sample_step: Resampling step size in Angstroms.
        angpix: Pixel size in Angstroms.
        tube_id: Tube ID for output particles.
    
    Returns:
        List of resampled particle dictionaries (coordinates in pixels).
    """
    from .fit import resample
    
    # Convert coordinates from pixels to Angstroms
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    n_points = len(coords)
    
    # Validate sufficient points for fitting
    if n_points < poly_order + 1:
        print(f"  Warning: Tube {tube_id} has {n_points} points, "
              f"insufficient for polynomial order {poly_order}. Skipping.")
        return []
    
    # Extract metadata
    tomo_name = (
        tube_data['rlnTomoName'].iloc[0]
        if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty
        else "Unknown"
    )
    
    # Determine primary axis (dimension with largest range)
    ranges = np.ptp(coords, axis=0)
    
    if ranges[0] >= ranges[1]:  # X is primary axis
        independent_var = coords[:, 0]
        dependent_y = coords[:, 1]
        dependent_z = coords[:, 2]
        mode = 1
    else:  # Y is primary axis
        independent_var = coords[:, 1]
        dependent_y = coords[:, 0]
        dependent_z = coords[:, 2]
        mode = 2
    
    # Fit polynomials
    poly_y = np.poly1d(np.polyfit(independent_var, dependent_y, poly_order))
    poly_z = np.poly1d(np.polyfit(independent_var, dependent_z, poly_order))
    
    # Resample along fitted curve
    integration_step = sample_step / INTEGRATION_STEP_DIVISOR
    start = independent_var.min()
    end = independent_var.max()
    
    resampled_points = resample(
        poly_y, poly_z, start, end, mode,
        tube_id - 1,  # cluster_id is 0-indexed in original code
        tomo_name, sample_step, angpix,
        integration_step
    )
    
    # Convert coordinates back from Angstroms to pixels
    for point in resampled_points:
        point['rlnCoordinateX'] /= angpix
        point['rlnCoordinateY'] /= angpix
        point['rlnCoordinateZ'] /= angpix
    
    return resampled_points


def refit_and_resample_all_tubes(
    df: pd.DataFrame,
    poly_order: int,
    sample_step: float,
    angpix: float
) -> pd.DataFrame:
    """
    Renumber tube IDs consecutively and refit/resample all tubes.
    
    Args:
        df: DataFrame with merged tubes.
        poly_order: Polynomial order for curve fitting.
        sample_step: Resampling step size in Angstroms.
        angpix: Pixel size in Angstroms.
    
    Returns:
        DataFrame with resampled particles (coordinates in pixels).
    """
    print(f"\n{'='*60}")
    print("Post-Merging Refit & Resample")
    print(f"{'='*60}")
    print(f"  Polynomial Order: {poly_order}")
    print(f"  Resampling Step:  {sample_step:.2f} Å")
    
    # Renumber tube IDs consecutively
    unique_ids = df['rlnHelicalTubeID'].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    
    df_renumbered = df.copy()
    df_renumbered['rlnHelicalTubeID'] = df_renumbered['rlnHelicalTubeID'].map(id_mapping)
    
    # Resample each tube
    all_resampled = []
    
    for new_id in sorted(id_mapping.values()):
        tube_data = df_renumbered[df_renumbered['rlnHelicalTubeID'] == new_id]
        resampled = fit_and_resample_tube(tube_data, poly_order, sample_step, angpix, new_id)
        all_resampled.extend(resampled)
    
    if not all_resampled:
        print("  Warning: Resampling failed. Returning renumbered data without resampling.")
        return df_renumbered
    
    # Create output DataFrame
    df_output = pd.DataFrame(all_resampled)
        
    print(f"  ✓ Resampled {df_output['rlnHelicalTubeID'].nunique()} tubes "
          f"into {len(df_output)} particles")
    
    # Ensure standard column order
    required_cols = [
        'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
        'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
        'rlnHelicalTubeID', 'rlnTomoName', 'rlnImagePixelSize'
    ]
    
    
    # Fill missing columns with defaults
    for col in required_cols:
        if col not in df_output.columns:
            df_output[col] = 0.0
    
    return df_output[required_cols]

def connect_tubes_once(
    df: pd.DataFrame,
    angpix: float,
    overlap_threshold: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order_seed: int,
    poly_order_final: int,
    sample_step: float,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Single pass of tube connection pipeline.
    
    Executes one round of the tube connection pipeline:
    1. Find and merge connections at the given extrapolation distance
    2. Refit and resample merged tubes with polynomial curves
    
    Args:
        df: DataFrame with particle data (coordinates in pixels).
        angpix: Pixel size in Angstroms.
        overlap_threshold: Maximum distance in Angstroms to consider tubes connected.
        min_seed: Number of points to use for trajectory fitting.
        dist_extrapolate: Distance to extrapolate trajectories in Angstroms.
        poly_order_seed: Polynomial order for extrapolation fitting.
        poly_order_final: Polynomial order for final curve fitting.
        sample_step: Resampling step size in Angstroms.
        debug: If True, print detailed distance information for all connections.
    
    Returns:
        DataFrame with connected and resampled tubes (coordinates in pixels).
    """
    tubes_before = df['rlnHelicalTubeID'].nunique()
    
    # Find connections
    connections = find_tube_connections(
        df,
        angpix,
        overlap_threshold,
        min_seed,
        dist_extrapolate,
        poly_order_seed
    )

    # Debug mode: show top 5 closest tube pairs (regardless of connection threshold)
    if debug:
        print(f"\n    {'='*56}")
        print(f"    DEBUG: Top 5 closest tube pairs")
        print(f"    {'='*56}")
        
        # Get all unique tube IDs
        tube_ids = sorted(df['rlnHelicalTubeID'].unique())
        
        # Calculate pairwise distances between all tube endpoints
        min_distances = []
        for i, tube1 in enumerate(tube_ids):
            for tube2 in tube_ids[i+1:]:
                # Get endpoints for both tubes
                df1 = df[df['rlnHelicalTubeID'] == tube1].copy()
                df2 = df[df['rlnHelicalTubeID'] == tube2].copy()
                
                if len(df1) < 2 or len(df2) < 2:
                    continue
                
                # Calculate minimum distance between any endpoints
                coords1 = df1[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
                coords2 = df2[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
                
                # Get first and last points
                ends1 = np.array([coords1[0], coords1[-1]])
                ends2 = np.array([coords2[0], coords2[-1]])
                
                # Calculate all endpoint-to-endpoint distances
                min_dist = float('inf')
                for end1 in ends1:
                    for end2 in ends2:
                        dist = np.linalg.norm((end1 - end2) * angpix)
                        min_dist = min(min_dist, dist)
                
                min_distances.append({
                    'tube1': tube1,
                    'tube2': tube2,
                    'distance': min_dist
                })
        
        # Sort by distance and show top 5
        min_distances.sort(key=lambda x: x['distance'])
        for i, pair in enumerate(min_distances[:5], 1):
            threshold_marker = "✓" if pair['distance'] <= overlap_threshold else "✗"
            print(f"    {i}. Tubes {pair['tube1']:4d} ↔ {pair['tube2']:4d}: "
                  f"{pair['distance']:6.1f} Å {threshold_marker}")
        
        print(f"    {'='*56}")
        print(f"    (✓ = within threshold, ✗ = exceeds threshold)\n")
    
    if not connections:
        print(f"    No connections found")
        return df
    
    print(f"    Found {len(connections)} potential connections")
    
    # Debug mode: print all connection distances
    if debug:
        print(f"\n    {'='*56}")
        print(f"    DEBUG: All connection distances")
        print(f"    {'='*56}")
        for i, conn in enumerate(connections, 1):
            print(f"    {i:3d}. Tubes {conn['tube_id1']:4d} ↔ {conn['tube_id2']:4d}: "
                  f"{conn['min_overlap_dist']:6.1f} Å")
        print(f"    {'='*56}\n")
    else:
        # Show top 3 connections only
        for i, conn in enumerate(connections[:3], 1):
            print(f"      {i}. Tubes {conn['tube_id1']} ↔ {conn['tube_id2']}: "
                  f"{conn['min_overlap_dist']:.1f} Å")
        if len(connections) > 3:
            print(f"      ... and {len(connections) - 3} more")
    
    # Merge connected tubes
    df_merged, _ = merge_connected_tubes(df, connections)
    
    tubes_after = df_merged['rlnHelicalTubeID'].nunique()
    merges = tubes_before - tubes_after
    
    print(f"    Merged {merges} tube groups")
    print(f"    Remaining tubes: {tubes_after}")
    
    # Refit and resample
    print(f"    Refitting and resampling...")
    df_final = refit_and_resample_all_tubes(
        df_merged,
        poly_order_final,
        sample_step,
        angpix
    )
    
    return df_final


def connect_tubes(
    df: pd.DataFrame,
    angpix: float,
    overlap_threshold: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order_seed: int,
    poly_order_final: int,
    sample_step: float,
    max_iterations: int = 2,
    dist_scale: float = 1.0,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Complete tube connection pipeline with multiple rounds.
    
    Executes the full pipeline for connecting broken helical tubes:
    1. Runs multiple rounds of connection attempts
    2. Each round: find connections → merge → refit → resample
    3. The refitting between rounds can reveal new connection opportunities
    
    Args:
        df: DataFrame with particle data (coordinates in pixels).
        angpix: Pixel size in Angstroms.
        overlap_threshold: Maximum distance in Angstroms to consider tubes connected.
        min_seed: Number of points to use for trajectory fitting.
        dist_extrapolate: Distance to extrapolate trajectories in Angstroms.
        poly_order_seed: Polynomial order for extrapolation fitting.
        poly_order_final: Polynomial order for final curve fitting.
        sample_step: Resampling step size in Angstroms.
        max_iterations: Maximum number of connection iterations (default: 2).
        dist_scale: Scaling factor for extrapolation distance per iteration (default: 1.0).
        debug: If True, print detailed distance information for all connections.
    
    Returns:
        DataFrame with connected and resampled tubes (coordinates in pixels).
    """
    print("\n" + "="*60)
    print("TUBE CONNECTION PIPELINE")
    print("="*60)
        
    tubes_initial = df['rlnHelicalTubeID'].nunique()
    particles_initial = len(df)
    

    print(f"\nInitial data: {tubes_initial} tubes, {particles_initial} particles")
    print(f"\nConnection parameters:")
    print(f"  Pixel Size:     {angpix:.1f} Å/pixel")
    print(f"  Overlap threshold:     {overlap_threshold:.1f} Å")
    print(f"  Extrapolation distance: {dist_extrapolate:.1f} Å")
    print(f"  Distance scaling:      {dist_scale}x per iteration")
    print(f"  Max iterations:        {max_iterations}")
    print(f"  Seed points:           {min_seed}")
    print(f"  Seed poly order:       {poly_order_seed}")
    print(f"  Final poly order:      {poly_order_final}")
    print(f"  Resampling step:       {sample_step:.1f} Å")
    if debug:
        print(f"  Debug mode:            ON")
    
    # Multiple iterations of connection
    df_current = df.copy()
    total_merges = 0
    current_dist = dist_extrapolate
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations} (extrapolation: {current_dist:.1f} Å)")
        print(f"{'='*60}")
        
        tubes_before = df_current['rlnHelicalTubeID'].nunique()
        
        # Run one connection pass
        df_current = connect_tubes_once(
            df_current,
            angpix,
            overlap_threshold,
            min_seed,
            current_dist,
            poly_order_seed,
            poly_order_final,
            sample_step,
            debug=debug
        )
        

        tubes_after = df_current['rlnHelicalTubeID'].nunique()
        merges_this_iter = tubes_before - tubes_after
        total_merges += merges_this_iter
        
        # Check if we should continue
        if merges_this_iter == 0:
            if iteration == 1:
                print(f"\n✓ No connections needed - tubes are well separated")
            else:
                print(f"\n✓ Converged (no new merges in iteration {iteration})")
            break
        
        # Scale distance for next iteration
        if iteration < max_iterations:
            current_dist *= dist_scale
    
    # Final summary
    tubes_final = df_current['rlnHelicalTubeID'].nunique()
    particles_final = len(df_current)
    tubes_merged = tubes_initial - tubes_final
    
    
    print("\n" + "="*60)
    print("CONNECTION PIPELINE COMPLETE")
    print("="*60)
    print(f"  Total iterations: {min(iteration, max_iterations)}")
    print(f"  Tubes merged:     {tubes_merged}")
    print(f"  Final tubes:      {tubes_final}")
    print(f"  Final particles:  {particles_final} "
          f"({particles_final - particles_initial:+d})")
    print("="*60 + "\n")
    
    return df_current