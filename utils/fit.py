#!/usr/bin/env python
# coding: utf-8

"""
Fit lines on scatter data.

This module provides functionality to:
- Fit lines on scatter points from template matching
- Resample points at specific interval

Code based on https://github.com/PengxinChai/multi-curve-fitting

@Builab 2025
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

PRECOMPUTE_DISTANCE_THRESHOLD=3000
EVALUATION_STEP_SIZE = 40.0 # In Angstrom

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points (2D or 3D)."""
    return np.linalg.norm(p1 - p2)


def find_seed(
    i: int,
    j: int,
    coords: np.ndarray,
    assigned_clusters: np.ndarray,
    min_seed: int,
    max_distance_to_line: float,
    min_distance_in_extension_seed: float,
    max_distance_in_extension_seed: float
) -> List[int]:
    """
    Find an initial seed of collinear points starting from points i and j.
    
    Returns:
        List of indices forming a valid seed, or empty list if no valid seed found.
    """
    if assigned_clusters[i] != -1 or assigned_clusters[j] != -1:
        return []

    seed_indices = [i, j]
    p1, p2 = coords[i, :2], coords[j, :2]
    k1, k2 = coords[i, 2], coords[j, 2]

    # Line equation: ax + by + c = 0
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -p1[0] * p2[1] + p2[0] * p1[1]
    norm_factor = np.sqrt(a**2 + b**2)
    
    if norm_factor == 0:
        return []

    # Check Z-slice distance for initial pair
    if abs(k1 - k2) > max_distance_to_line:
        return []
    
    k_avg = (k1 + k2) / 2
    
    unassigned_mask = (assigned_clusters == -1)
    unassigned_indices = np.where(unassigned_mask)[0]
    
    potential_points_mask = np.ones(len(coords), dtype=bool)
    potential_points_mask[seed_indices] = False

    while True:
        found_new_point = False
        
        # Vectorized distance calculation
        candidate_mask = unassigned_mask & potential_points_mask
        candidate_coords = coords[candidate_mask]
        
        if len(candidate_coords) == 0:
            break
        
        dist_to_line = np.abs(a * candidate_coords[:, 0] + 
                             b * candidate_coords[:, 1] + c) / norm_factor
        delta_z = np.abs(candidate_coords[:, 2] - k_avg)
        
        line_candidates_mask = ((dist_to_line < max_distance_to_line) & 
                               (delta_z < max_distance_to_line))
        
        candidate_indices = unassigned_indices[potential_points_mask[unassigned_mask]][line_candidates_mask]

        # Pre-compute all distances at once for better performance
        if len(candidate_indices) > 0:
            seed_coords = coords[seed_indices, :2]
            candidate_points = coords[candidate_indices, :2]
            
            # Broadcasting: (n_candidates, 1, 2) - (1, n_seeds, 2) -> (n_candidates, n_seeds)
            dist_matrix = np.linalg.norm(
                candidate_points[:, np.newaxis, :] - seed_coords[np.newaxis, :, :], 
                axis=2
            )
            min_distances = np.min(dist_matrix, axis=1)
            
            # Find valid candidates all at once
            valid_mask = ((min_distances > min_distance_in_extension_seed) & 
                         (min_distances < max_distance_in_extension_seed))
            valid_indices = candidate_indices[valid_mask]
            
            if len(valid_indices) > 0:
                # Take the first valid point
                k = valid_indices[0]
                seed_indices.append(k)
                potential_points_mask[k] = False
                
                if len(seed_indices) >= min_seed:
                    return seed_indices
                
                found_new_point = True

        if not found_new_point:
            break
            
    return seed_indices if len(seed_indices) >= min_seed else []


def angle_evaluate(
    poly: np.poly1d,
    point: float,
    mode: int,
    angpix: float,
    max_angle_change_per_4nm: float,
    integration_step: float = 0.1 
) -> int:
    """
    Evaluate curvature of polynomial fit.
    
    Returns:
        1 if curvature is acceptable, 0 otherwise.
    """
    evaluation_step = EVALUATION_STEP_SIZE / angpix
    step = integration_step
    
    def get_next_pos(val: float) -> Tuple[float, float]:
        next_val = val + step
        return (next_val, poly(next_val)) if mode == 1 else (poly(next_val), next_val)

    def get_slope(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return np.inf if abs(dx) < 1e-4 else dy / dx
    
    # Calculate initial position and angle
    current_pos = (point, poly(point)) if mode == 1 else (poly(point), point)
    next_pos = get_next_pos(point)
    angle_one = math.degrees(math.atan(get_slope(current_pos, next_pos)))

    # Move along curve for evaluation_step distance
    accumulation = 0.0
    while accumulation < evaluation_step:
        next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
        accumulation += distance(np.array(current_pos), np.array(next_pos))
        current_pos = next_pos
    
    # Calculate final angle
    final_next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
    angle_two = math.degrees(math.atan(get_slope(current_pos, final_next_pos)))
    
    return 1 if abs(angle_two - angle_one) < max_angle_change_per_4nm else 0


def resample(
    poly_xy: np.poly1d,
    poly_k: np.poly1d,
    start: float,
    end: float,
    mode: int,
    cluster_id: int,
    tomo_name: str,
    sample_step: float,
    angpix: float,
    integration_step: float = 0.1 
) -> List[Dict[str, Any]]:
    """Resample points along fitted 3D curve at specified step size."""
    resampled_points = []
    accumulation = 0.0
    current_val = start

    def get_coords(val: float) -> np.ndarray:
        if mode == 1:  # y=f(x)
            x, y = val, poly_xy(val)
        else:  # x=f(y)
            y, x = val, poly_xy(val)
        k = poly_k(val)
        return np.array([x, y, k])

    current_pos = get_coords(current_val)

    while current_val < end:
        next_val = current_val + integration_step
        next_pos = get_coords(next_val)
        
        dist = distance(current_pos, next_pos)
        accumulation += dist
        
        if accumulation >= sample_step:
            accumulation = 0.0
            
            # Calculate angles
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            d_xy = np.sqrt(dx**2 + dy**2)
            
            # Angle in XY plane (rlnAngleYX)
            # @Builab proper Relion angle calculation
            angle_yx = math.degrees(math.atan2(-dy, dx))
            
            # Angle with respect to XY plane (rlnAngleZXY)
            dz = next_pos[2] - current_pos[2]
            angle_zxy = math.degrees(math.atan2(dz, d_xy)) if d_xy != 0 else 0

            point_data = {
                'rlnCoordinateX': current_pos[0],
                'rlnCoordinateY': current_pos[1],
                'rlnCoordinateZ': current_pos[2],
                'rlnAngleRot': 0,
                'rlnAngleTilt': angle_zxy + 90,
                'rlnAnglePsi': angle_yx,
                'rlnHelicalTubeID': cluster_id + 1,
                'rlnTomoName': tomo_name,
                'rlnImagePixelSize': angpix
            }
                
            resampled_points.append(point_data)

        current_pos = next_pos
        current_val = next_val
        
    return resampled_points


def seed_extension(
    seed_indices: List[int],
    coords: np.ndarray,
    assigned_clusters: np.ndarray,
    cluster_id: int,
    tomo_name: str,
    poly_order: int,
    poly_order_seed: int,
    seed_evaluation_constant: float,
    angpix: float,
    max_angle_change_per_4nm: float,
    max_distance_to_curve: float,
    min_distance_in_extension: float,
    max_distance_in_extension: float,
    min_number_growth: int,
    sample_step: float,
    integration_step: float = 0.1 
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Extend seed by iteratively fitting polynomial and adding nearby points.
    
    Returns:
        Tuple of (final cluster indices, resampled points) or ([], []) on failure.
    """
    cluster_indices = list(seed_indices)
    
    # Determine fitting mode
    cluster_coords = coords[cluster_indices]
    delta_x = np.ptp(cluster_coords[:, 0])
    delta_y = np.ptp(cluster_coords[:, 1])
    mode = 1 if delta_x >= delta_y else 0  # 1: y=f(x), 0: x=f(y)

    # Set up variables for fitting
    if mode == 1:
        ind_vars = cluster_coords[:, 0]
        dep_vars_xy = cluster_coords[:, 1]
    else:
        ind_vars = cluster_coords[:, 1]
        dep_vars_xy = cluster_coords[:, 0]
    dep_vars_k = cluster_coords[:, 2]

    # --- Seed Evaluation ---
    # 1. Fit with lower order polynomial
    poly_seed_xy = np.poly1d(np.polyfit(ind_vars, dep_vars_xy, poly_order_seed))
    
    # 2. Check fitting error
    errors = np.abs(poly_seed_xy(ind_vars) - dep_vars_xy)
    if np.any(errors >= seed_evaluation_constant):
        return [], []

    # 3. Check curvature
    poly_xy_final = np.poly1d(np.polyfit(ind_vars, dep_vars_xy, poly_order))
    mid_point = (np.min(ind_vars) + np.max(ind_vars)) / 2
    if not angle_evaluate(poly_xy_final, mid_point, mode, angpix, max_angle_change_per_4nm):
        return [], []

    # --- Curve Growth ---
    MAX_ITERATIONS = 100  # Prevent infinite loops
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        grew = False
        unassigned_indices = np.where(assigned_clusters == -1)[0]
        
        if len(unassigned_indices) == 0:
            break
        
        # Re-fit polynomial with all current cluster points
        all_cluster_coords = coords[cluster_indices]
        if mode == 1:
            ind_vars_all = all_cluster_coords[:, 0]
            dep_vars_xy_all = all_cluster_coords[:, 1]
        else:
            ind_vars_all = all_cluster_coords[:, 1]
            dep_vars_xy_all = all_cluster_coords[:, 0]
        dep_vars_k_all = all_cluster_coords[:, 2]

        poly_xy_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_xy_all, poly_order))
        poly_k_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_k_all, poly_order))

        # Vectorize the curve distance computation
        unassigned_coords = coords[unassigned_indices]
        ind_vars_unassigned = unassigned_coords[:, 0] if mode == 1 else unassigned_coords[:, 1]
        dep_vars_unassigned = unassigned_coords[:, 1] if mode == 1 else unassigned_coords[:, 0]
        
        # Calculate distances to curve for all unassigned points at once
        dist_to_curve_xy = np.abs(poly_xy_growth(ind_vars_unassigned) - dep_vars_unassigned)
        dist_to_curve_k = np.abs(poly_k_growth(ind_vars_unassigned) - unassigned_coords[:, 2])
        
        # Filter points close to curve
        close_to_curve_mask = ((dist_to_curve_xy < max_distance_to_curve) & 
                               (dist_to_curve_k < max_distance_to_curve))
        close_indices = unassigned_indices[close_to_curve_mask]
        
        if len(close_indices) > 0:
            # Compute all distances at once using broadcasting
            cluster_coords_2d = coords[cluster_indices, :2]
            close_coords_2d = coords[close_indices, :2]
            
            # Shape: (n_close, n_cluster)
            dist_matrix = np.linalg.norm(
                close_coords_2d[:, np.newaxis, :] - cluster_coords_2d[np.newaxis, :, :],
                axis=2
            )
            min_distances = np.min(dist_matrix, axis=1)
            
            # Find valid points
            valid_mask = ((min_distances > min_distance_in_extension) & 
                         (min_distances < max_distance_in_extension))
            valid_indices = close_indices[valid_mask]
            
            # Add all valid points at once
            if len(valid_indices) > 0:
                cluster_indices.extend(valid_indices)
                assigned_clusters[valid_indices] = -2  # Provisional assignment
                grew = True

        if not grew:
            break

    # --- Final Evaluation and Resampling ---
    if len(cluster_indices) - len(seed_indices) >= min_number_growth:
        print(f"  - Seed extension successful. Cluster {cluster_id} found with "
              f"{len(cluster_indices)} points.")
        
        final_coords = coords[cluster_indices]
        
        if mode == 1:
            ind = final_coords[:, 0]
            dep_xy = final_coords[:, 1]
            dep_k = final_coords[:, 2]
        else:
            ind = final_coords[:, 1]
            dep_xy = final_coords[:, 0]
            dep_k = final_coords[:, 2]
        
        poly_final_xy = np.poly1d(np.polyfit(ind, dep_xy, poly_order))
        poly_final_k = np.poly1d(np.polyfit(ind, dep_k, poly_order))
        
        resampled_data = resample(
            poly_final_xy, poly_final_k, np.min(ind), np.max(ind),
            mode, cluster_id, tomo_name, sample_step, angpix
        )
        return cluster_indices, resampled_data
    else:
        # Revert provisional assignments
        for idx in cluster_indices:
            if assigned_clusters[idx] == -2:
                assigned_clusters[idx] = -1
        return [], []


def fit_curves(
    coords: np.ndarray,
    tomo_name: str,
    angpix: float,
    poly_order: int,
    sample_step: float,
    min_seed: int,
    max_distance_to_line: float,
    min_distance_in_extension_seed: float,
    max_distance_in_extension_seed: float,
    poly_order_seed: int,
    seed_evaluation_constant: float,
    max_angle_change_per_4nm: float,
    max_distance_to_curve: float,
    min_distance_in_extension: float,
    max_distance_in_extension: float,
    min_number_growth: int,
    cluster_id_offset: int = 0,
    integration_step: float = 0.1 
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Core computational engine for curve fitting (I/O-free).

    Args:
        coords: Array of shape (N, 3) with X, Y, Z coordinates.
        tomo_name: Tomogram name for output.
        angpix: Pixel size in Angstroms.
        poly_order: Polynomial order for fitting.
        sample_step: Resampling step size (in pixels).
        min_seed: Minimum number of points for valid seed.
        max_distance_to_line: Max distance from seed line (in pixels).
        min_distance_in_extension_seed: Min distance between seed neighbors (in pixels).
        max_distance_in_extension_seed: Max distance between seed neighbors (in pixels).
        poly_order_seed: Polynomial order for seed evaluation.
        seed_evaluation_constant: Max fitting error for seed.
        max_angle_change_per_4nm: Max curvature change (degrees).
        max_distance_to_curve: Max distance from curve during growth (in pixels).
        min_distance_in_extension: Min distance between neighbors during growth (in pixels).
        max_distance_in_extension: Max distance between neighbors during growth (in pixels).
        min_number_growth: Min points to add during growth.
        cluster_id_offset: Offset for cluster IDs.
        integration_step: Integration step for curve length (in pixels).

    Returns:
        Tuple of (resampled DataFrame, cluster assignments, cluster count).
    """
    total_number = len(coords)
    assigned_clusters = np.full(total_number, -1, dtype=int)
    cluster_id_counter = 0
    all_resampled_points = []

    # Pre-compute 2D distance matrix for efficiency (if dataset is not too large)
    # Only compute if reasonable memory footprint (< ~10M distances)
    use_precomputed_distances = total_number < PRECOMPUTE_DISTANCE_THRESHOLD
    dist_matrix_2d = None
    
    if use_precomputed_distances:
        # Compute pairwise 2D distances once
        coords_2d = coords[:, :2]
        # Using broadcasting: shape (N, 1, 2) - (1, N, 2) -> (N, N)
        diff = coords_2d[:, np.newaxis, :] - coords_2d[np.newaxis, :, :]
        dist_matrix_2d = np.sqrt(np.sum(diff**2, axis=2))

    # Main clustering loop
    for i in range(total_number):
        if assigned_clusters[i] != -1:
            continue
        
        # Get distances from point i to all other points
        if use_precomputed_distances:
            distances_from_i = dist_matrix_2d[i]
        else:
            distances_from_i = np.linalg.norm(coords[:, :2] - coords[i, :2], axis=1)
        
        # Find potential partners within distance range
        potential_partners = np.where(
            (assigned_clusters == -1) &
            (distances_from_i > min_distance_in_extension_seed) &
            (distances_from_i < max_distance_in_extension_seed) &
            (np.arange(total_number) > i)  # Only check j > i to avoid duplicates
        )[0]
        
        for j in potential_partners:
            seed_indices = find_seed(
                i, j, coords, assigned_clusters,
                min_seed, max_distance_to_line,
                min_distance_in_extension_seed, max_distance_in_extension_seed
            )
            
            if seed_indices:
                current_cluster_id = cluster_id_counter + cluster_id_offset
                final_indices, resampled_data = seed_extension(
                    seed_indices, coords, assigned_clusters,
                    current_cluster_id, tomo_name,
                    poly_order, poly_order_seed, seed_evaluation_constant,
                    angpix, max_angle_change_per_4nm,
                    max_distance_to_curve, min_distance_in_extension,
                    max_distance_in_extension, min_number_growth,
                    sample_step
                )
                
                if final_indices:
                    for idx in final_indices:
                        assigned_clusters[idx] = current_cluster_id
                    all_resampled_points.extend(resampled_data)
                    cluster_id_counter += 1
                    break  # Move to next i since this point is now assigned
    
    df_resam = pd.DataFrame(all_resampled_points)
    return df_resam, assigned_clusters, cluster_id_counter

