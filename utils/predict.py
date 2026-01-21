#!/usr/bin/env python3
"""
Predict particle angles from template geometry.

This module provides a three-stage pipeline:
1. LCC filtering - Keep high-confidence template particles
2. Angle mapping - Transfer angles from template to input via spatial proximity
3. Filament snapping - Adjust outlier angles to filament median

@Builab 2025
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Optional, Dict, Tuple

from .io import validate_dataframe

# Column name constants
COORD_COLUMNS = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
ANGLE_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]

# Default parameters
DEFAULT_LCC_KEEP_PERCENT = 80.0
DEFAULT_MAPPING_RADIUS = 100.0  # Angstroms
DEFAULT_MAX_NEIGHBORS = 8
DEFAULT_SNAP_MAX_DELTA = 20.0  # degrees
DEFAULT_MIN_FILAMENT_POINTS = 5

def normalize_angle(angle: float) -> float:
    """Normalize angle to range [-180, 180]."""
    return ((angle + 180) % 360) - 180

def circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate circular mean of angles in degrees.
    
    Args:
        angles: Array of angles in degrees.
        weights: Optional weights for each angle.
    
    Returns:
        Circular mean in degrees (0-360).
    """
    angles_rad = np.deg2rad(angles)
    
    if weights is None:
        sin_mean = np.sin(angles_rad).mean()
        cos_mean = np.cos(angles_rad).mean()
    else:
        weights_norm = weights / (weights.sum() + 1e-12)
        sin_mean = (np.sin(angles_rad) * weights_norm).sum()
        cos_mean = (np.cos(angles_rad) * weights_norm).sum()
    
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return (np.rad2deg(mean_rad) + 360.0) % 360.0


def circular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate signed circular difference between two angles.
    
    Args:
        angle1: First angle in degrees.
        angle2: Second angle in degrees.
    
    Returns:
        Difference in degrees, wrapped to [-180, 180).
    """
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180


def filter_by_lcc(
    df_input: pd.DataFrame,
    df_template: pd.DataFrame,
    angpix: float,
    neighbor_radius: float,
    keep_percentage: float = DEFAULT_LCC_KEEP_PERCENT
) -> pd.DataFrame:
    """
    Filter template particles by Local Cross-Correlation scores.
    
    For each tube in the input, identifies template particles within the
    neighborhood and retains only those with highest LCC scores.
    
    Algorithm:
    1. For each tube ID in input data
    2. Find all template particles within neighbor_radius of any input particle
    3. Sort template particles by LCC score (descending)
    4. Keep top keep_percentage of those particles
    
    Args:
        df_input: Input DataFrame with coordinates and tube IDs.
        df_template: Template DataFrame with coordinates, tube IDs, and LCC scores.
        angpix: Pixel size in Angstroms for coordinate conversion.
        neighbor_radius: Search radius in Angstroms.
        keep_percentage: Percentage of top LCC particles to retain (0-100).
    
    Returns:
        Filtered template DataFrame with only high-LCC particles.
    
    Raises:
        ValueError: If required columns are missing or parameters invalid.
    """
    # Validate inputs
    validate_dataframe(df_input, COORD_COLUMNS + ['rlnHelicalTubeID'])
    validate_dataframe(df_template, COORD_COLUMNS)
    
    has_lcc = 'rlnLCCmax' in df_template.columns
    	
    if not 0 < keep_percentage <= 100:
        raise ValueError(
            f"keep_percentage must be in (0, 100], got {keep_percentage}"
        )
    
    # Convert radius to pixel units
    radius_px = neighbor_radius / angpix
    
    print(f"\n{'='*60}")
    print("LCC FILTERING")
    print(f"{'='*60}")
    print(f"  Radius: {neighbor_radius:.1f} Å ({radius_px:.2f} px)")
    if has_lcc:
        print(f"  Keeping top {keep_percentage}% by LCC score")
    else:
        print(f"  ⚠ No rlnLCCmax column - keeping all neighbors")
            
    # Build spatial index for template particles
    template_coords = df_template[COORD_COLUMNS].to_numpy(dtype=float)
    kdtree = cKDTree(template_coords)
    
    # Process each tube
    tube_ids = df_input['rlnHelicalTubeID'].unique()
    print(f"  Processing {len(tube_ids)} tubes...")
    
    indices_to_keep = set()
    
    for tube_id in tube_ids:
        # Get input particles for this tube
        input_tube = df_input[df_input['rlnHelicalTubeID'] == tube_id]
        input_coords = input_tube[COORD_COLUMNS].to_numpy(dtype=float)
        
        # Find template particles within radius of any input particle
        neighbor_lists = kdtree.query_ball_point(input_coords, r=radius_px)
        
        # Collect unique neighbor indices
        neighbor_indices = set()
        for neighbors in neighbor_lists:
            neighbor_indices.update(neighbors)
        
        if not neighbor_indices:
            continue
        
        # Get neighbor particles and sort by LCC
        neighbor_mask = np.zeros(len(df_template), dtype=bool)
        neighbor_mask[list(neighbor_indices)] = True
        neighbors_df = df_template[neighbor_mask]
        
        if 'rlnLCCmax' in df_template.columns:
            # Keep top percentage by LCC score
            neighbors_sorted = neighbors_df.sort_values('rlnLCCmax', ascending=False)
            n_keep = max(1, int(np.ceil(len(neighbors_sorted) * keep_percentage / 100.0)))
            top_particles = neighbors_sorted.iloc[:n_keep]
        else:
            top_particles = neighbors_df
            
        indices_to_keep.update(top_particles.index)
    
    # Create filtered DataFrame
    df_filtered = df_template.loc[list(indices_to_keep)].copy()
    
    reduction = 100 * (1 - len(df_filtered) / len(df_template))
    print(f"  ✓ Filtered: {len(df_template)} → {len(df_filtered)} particles "
          f"({reduction:.1f}% reduction)")
    
    return df_filtered


def map_angles_from_template(
    df_input: pd.DataFrame,
    df_template: pd.DataFrame,
    angpix: float = 14.0,
    search_radius: float = DEFAULT_MAPPING_RADIUS,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    weight_by_distance: bool = True
) -> pd.DataFrame:
    """
    Map angles from template to input particles based on spatial proximity.
    
    For each input particle:
    1. Find nearby template particles within search_radius
    2. Compute circular mean of their angles (optionally weighted by distance)
    3. Assign computed angles to input particle
    
    Args:
        df_input: Input DataFrame with coordinates.
        df_template: Template DataFrame with coordinates and angles.
        angpix: Pixel size in Angstroms.
        search_radius: Search radius in Angstroms.
        max_neighbors: Maximum number of neighbors to consider.
        weight_by_distance: Whether to weight angles by inverse distance.
    
    Returns:
        Input DataFrame with mapped angles.
    
    Raises:
        ValueError: If required columns are missing.
    """
    # Validate columns
    validate_dataframe(df_input, COORD_COLUMNS)
    validate_dataframe(df_template, COORD_COLUMNS + ANGLE_COLUMNS)
    
    print(f"\n{'='*60}")
    print("ANGLE MAPPING")
    print(f"{'='*60}")
    print(f"  Search radius: {search_radius:.1f} Å")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  Distance weighting: {'enabled' if weight_by_distance else 'disabled'}")
    
    # Extract coordinates
    input_coords = df_input[COORD_COLUMNS].to_numpy(dtype=float)
    template_coords = df_template[COORD_COLUMNS].to_numpy(dtype=float)
    radius_px = search_radius / angpix
    
    # Initialize output with angles
    df_output = df_input.copy()
    for angle_col in ANGLE_COLUMNS:
        df_output[angle_col] = 0.0
    
    fallback_counts = {col: 0 for col in ANGLE_COLUMNS}
    
    # Process each input particle
    for i, particle_coord in enumerate(input_coords):
        # Calculate distances to all template particles
        distances = np.linalg.norm(template_coords - particle_coord, axis=1)
        neighbor_order = np.argsort(distances)
        
        # Get k nearest neighbors
        nearest_k = neighbor_order[:max(1, max_neighbors)]
        neighbor_distances = distances[nearest_k]
        
        # Check which neighbors are within radius
        within_radius = neighbor_distances <= radius_px
        
        if not np.any(within_radius):
            # No neighbors in radius - use closest one as fallback
            within_radius = np.zeros_like(neighbor_distances, dtype=bool)
            within_radius[0] = True
            used_fallback = True
        else:
            used_fallback = False
        
        # Select neighbors to use
        selected_neighbors = nearest_k[within_radius]
        selected_distances = neighbor_distances[within_radius]
        
        # Compute distance weights if requested
        if weight_by_distance and len(selected_distances) > 1:
            weights = 1.0 / (selected_distances + 1e-9)
        else:
            weights = None
        
        # Map each angle using circular mean
        for angle_col in ANGLE_COLUMNS:
            template_angles = df_template[angle_col].iloc[selected_neighbors].to_numpy(dtype=float)
            mapped_angle = circular_mean(template_angles, weights)
            df_output.at[i, angle_col] = mapped_angle
            
            if used_fallback:
                fallback_counts[angle_col] += 1
    
    # Report fallback usage
    if any(count > 0 for count in fallback_counts.values()):
        total_fallbacks = fallback_counts[ANGLE_COLUMNS[0]]
        print(f"  ⚠ {total_fallbacks} particles required fallback "
              f"(no neighbors within radius)")
    
    print(f"  ✓ Mapped angles to {len(df_output)} particles")
    
    return df_output


def snap_angles_to_filament_median(
    df: pd.DataFrame,
    max_delta: float = DEFAULT_SNAP_MAX_DELTA,
    min_points: int = DEFAULT_MIN_FILAMENT_POINTS
) -> pd.DataFrame:
    """
    Adjust outlier angles to filament median values.
    
    For each filament (tube):
    1. Calculate circular median for each angle
    2. Identify particles with angles > max_delta from median
    3. Snap those outliers to the median value
    
    Args:
        df: DataFrame with particles, angles, and tube IDs.
        max_delta: Maximum angular deviation in degrees before snapping.
        min_points: Minimum points required per filament for snapping.
    
    Returns:
        DataFrame with adjusted angles.
    """
    if 'rlnHelicalTubeID' not in df.columns:
        print("[Snap] No rlnHelicalTubeID column; treating all as one filament")
        df = df.copy()
        df['rlnHelicalTubeID'] = 1
    
    print(f"\n{'='*60}")
    print("FILAMENT ANGLE SNAPPING")
    print(f"{'='*60}")
    print(f"  Max deviation: {max_delta}°")
    print(f"  Min points per filament: {min_points}")
    
    df_output = df.copy()
    snap_counts = {col: 0 for col in ANGLE_COLUMNS}
    
    # Process each filament
    for tube_id, tube_group in df_output.groupby('rlnHelicalTubeID'):
        if len(tube_group) < min_points:
            continue
        
        # Process each angle column
        for angle_col in ANGLE_COLUMNS:
            angles = tube_group[angle_col].to_numpy(dtype=float)
            
            # Calculate median using circular mean
            median_angle = circular_mean(angles)
            
            # Find outliers
            deviations = np.abs([
                circular_difference(angle, median_angle) for angle in angles
            ])
            outlier_mask = deviations > max_delta
            
            # Snap outliers to median
            if np.any(outlier_mask):
                outlier_indices = tube_group.index[outlier_mask]
                df_output.loc[outlier_indices, angle_col] = median_angle
                snap_counts[angle_col] += int(outlier_mask.sum())
    
    # Report results
    total_snapped = sum(snap_counts.values())
    if total_snapped > 0:
        print(f"  ✓ Snapped {total_snapped} angles across all filaments")
        for angle_col, count in snap_counts.items():
            if count > 0:
                print(f"    - {angle_col}: {count}")
    else:
        print(f"  ✓ No outliers detected")
    
    return df_output

def fit_polynomial(x, y, order=2):
    """
    Fit polynomial to data.
    
    Returns:
        Tuple of (fitted_y, residuals, rmse, poly_object)
    """
    # Ensure there is enough data for fitting (at least order + 1 points)
    if len(x) <= order:
        return np.full_like(y, np.nan), np.full_like(y, np.nan), np.nan, None

    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)
    fitted_y = poly(x)
    residuals = y - fitted_y
    rmse = np.sqrt(np.mean(residuals**2))
    return fitted_y, residuals, rmse, poly


def robust_mad_outlier_detection(residuals, threshold=3.5):
    """
    Detects outliers using the Modified Z-score based on the Median Absolute Deviation (MAD).
    """
    if len(residuals) < 5: 
        return np.zeros_like(residuals, dtype=bool)

    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    if mad == 0:
        return np.abs(residuals - median_residual) > 1e-6 

    # 0.6745 is the correction factor for consistency with Gaussian Z-score
    modified_z_score = 0.6745 * (residuals - median_residual) / mad
    
    is_outlier = np.abs(modified_z_score) > threshold
    return is_outlier


def iterative_fit_and_detect(particle_indices, angles, order=2, n_iterations=2):
    """
    Performs iterative polynomial fitting and MAD outlier detection (2 steps).
    
    Returns:
        Tuple: (final_fitted_angles, total_outliers_mask)
    """
    
    N = len(angles)
    total_outliers_mask = np.zeros(N, dtype=bool)
    current_inlier_indices = np.arange(N)
    
    final_fitted_angles = np.full(N, np.nan)
    final_rmse = np.nan
    final_poly = None
    
    for i in range(n_iterations):
        
        # 1. Get current inlier data
        x_in = particle_indices[current_inlier_indices]
        y_in = angles[current_inlier_indices]
        
        # Skip if not enough data remains
        if len(x_in) <= order:
            break
            
        # 2. Fit polynomial to inliers and get the poly object
        _, _, _, poly = fit_polynomial(x_in, y_in, order)
        
        # 3. Calculate fitted line and residuals for ALL original points
        fitted_angles_all = poly(particle_indices)
        residuals_all = angles - fitted_angles_all
        
        # 4. Detect outliers among the current inliers
        residuals_for_detection = residuals_all[current_inlier_indices]
        is_outlier_in_subset = robust_mad_outlier_detection(residuals_for_detection)
        
        # 5. Map detected outliers back to the original index space
        outlier_indices_original = current_inlier_indices[is_outlier_in_subset]
        
        # 6. Break if no new outliers found
        if len(outlier_indices_original) == 0:
            final_fitted_angles = fitted_angles_all
            final_poly = poly
            final_rmse = np.sqrt(np.mean((angles[~total_outliers_mask] - fitted_angles_all[~total_outliers_mask])**2))
            break
            
        # 7. Update the total outlier mask
        total_outliers_mask[outlier_indices_original] = True
        
        # 8. Update the current inlier indices for the next iteration
        current_inlier_indices = np.where(~total_outliers_mask)[0]
        
        # 9. Record final results if this is the last iteration
        if i == n_iterations - 1:
            final_fitted_angles = fitted_angles_all
            final_poly = poly
            final_rmse = np.sqrt(np.mean((angles[~total_outliers_mask] - fitted_angles_all[~total_outliers_mask])**2))

    # Return only the essential items for correction
    return total_outliers_mask, final_poly


def smooth_angles(df):
    """
    Reads star df, performs iterative angle correction, and returns the corrected df.
    Prints the total number of detected and corrected outlier particles.
    """    
    df_corrected = df.copy()
    grouped = df.groupby('rlnHelicalTubeID')
    
    angle_names = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    
    # Initialize counter
    total_outliers_corrected = 0 

    # --- Iterative Correction Loop ---
    for tube_id, tube_data in grouped:
        particle_indices = np.arange(len(tube_data))
        tube_masks = {}
        tube_polys = {}
        
        # 1. Run iterative fit and detection for all three angles
        for angle_col in angle_names:
            angles = tube_data[angle_col].values
            
            # Get the mask and polynomial object
            total_outliers_mask, final_poly = \
                iterative_fit_and_detect(particle_indices, angles, order=2, n_iterations=2)
            
            tube_masks[angle_col] = total_outliers_mask
            tube_polys[angle_col] = final_poly

        # 2. Global Outlier Identification (Flag if outlier in ANY angle)
        global_outlier_mask = np.zeros(len(tube_data), dtype=bool)
        for angle in angle_names:
             global_outlier_mask = global_outlier_mask | tube_masks[angle]

        # 3. Extrapolation and Data Replacement
        if global_outlier_mask.any():
            # Count outliers for this tube and add to total
            outlier_count = global_outlier_mask.sum()
            total_outliers_corrected += outlier_count
            
            indices_in_tube = tube_data.index
            outlier_indices_original_df = indices_in_tube[global_outlier_mask]
            outlier_indices_local = particle_indices[global_outlier_mask]
            
            for angle in angle_names:
                poly = tube_polys[angle]
                # Calculate new (extrapolated) values using the final robust fit
                extrapolated_values = poly(outlier_indices_local)
                
                # Replace the values in the corrected DataFrame copy
                df_corrected.loc[outlier_indices_original_df, angle] = extrapolated_values
	
    # Print summary
    print(f"  Total outlier found and corrected: {total_outliers_corrected}")
    
    return df_corrected


def predict_angles(
    df_input: pd.DataFrame,
    df_template: Optional[pd.DataFrame] = None,
    angpix: float = 14.0,
    neighbor_radius: float = 100.0,
    lcc_keep_percent: float = 80.0,
    snap_max_delta: float = 20.0,
    snap_min_points: int = 5,
    direction: int = 0
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    
    df_working = df_input.copy()
    intermediates = {}

    # --- New: Direction Flip Logic ---
    if direction == 1:
        print("  Applying direction flip (180 - rlnAnglePsi)")
        # Normalize ensures the resulting angle stays in [-180, 180]
        df_working['rlnAnglePsi'] = df_working['rlnAnglePsi'].apply(lambda x: normalize_angle(x + 180))
        
        # Flip Tilt (Reverse direction along filament)
        # Note: Tilt is typically in range [0, 180]
        df_working['rlnAngleTilt'] = df_working['rlnAngleTilt'].apply(lambda x: 180.0 - x)

    # --- Step 1 & 2: Template Mapping (if available) ---
    if df_template is not None:
        df_filtered = filter_by_lcc(df_working, df_template, angpix, neighbor_radius, lcc_keep_percent)
        intermediates['filtered'] = df_filtered
        
        df_working = map_angles_from_template(df_working, df_filtered, angpix, neighbor_radius)
        intermediates['mapped'] = df_working

    # --- Step 3: Refinement & Smoothing ---
    df_final = snap_angles_to_filament_median(df_working, snap_max_delta, snap_min_points)
    
    # Final cleanup before smoothing
    df_final['rlnAnglePsi'] = normalize_angle(df_final['rlnAnglePsi'].values)
    df_final['rlnAngleRot'] = normalize_angle(df_final['rlnAngleRot'].values)
    
    df_corrected = smooth_angles(df_final) #
    
    return df_corrected, intermediates