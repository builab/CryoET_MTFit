"""
Scoring utilities for evaluating fitted tube quality against original data points.
"""

import pandas as pd
import numpy as np


def calculate_tube_scores(fitted_df: pd.DataFrame, original_df: pd.DataFrame, 
                         distance_threshold: float, verbose: bool = True) -> dict:
    """
    Calculate coverage and spreading scores for fitted tubes against original scatter points.
    
    Coverage score: What fraction of fitted points have at least one original point nearby?
    Spreading score: How concentrated are the original points around fitted points? (1.0 = ideal one-to-one)
    
    Args:
        fitted_df: DataFrame with rlnHelicalTubeID (fitted tubes) and coordinates
        original_df: DataFrame with original scatter points (no rlnHelicalTubeID) and coordinates
        distance_threshold: Maximum distance (in Angstroms) to consider a point as "covered"
        verbose: If True, print progress information
    
    Returns:
        Dictionary mapping tube_id (str) to dict with:
            - 'coverage': float (0.0 to 1.0)
            - 'spreading': float or None (1.0 = ideal, lower = more clustering)
            - 'avg_matches': float (average original points per fitted point)
            - 'total_fitted_points': int
            - 'fitted_points_with_coverage': int
    
    Raises:
        ValueError: If required columns are missing from either DataFrame
    """
    # Validate required columns
    required_coords = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    
    for col in required_coords:
        if col not in fitted_df.columns:
            raise ValueError(f"Fitted DataFrame is missing required column: {col}")
        if col not in original_df.columns:
            raise ValueError(f"Original DataFrame is missing required column: {col}")
    
    if 'rlnHelicalTubeID' not in fitted_df.columns:
        raise ValueError("Fitted DataFrame must have 'rlnHelicalTubeID' column")
    
    if 'rlnHelicalTubeID' in original_df.columns:
        raise ValueError("Original DataFrame should not have 'rlnHelicalTubeID' column (should be scatter points)")
    
    if 'rlnImagePixelSize' not in fitted_df.columns:
        raise ValueError("Fitted DataFrame is missing 'rlnImagePixelSize' column")
    
    if 'rlnImagePixelSize' not in original_df.columns:
        raise ValueError("Original DataFrame is missing 'rlnImagePixelSize' column")
    
    # Get pixel size to convert threshold from Angstroms to pixels
    pixel_size = fitted_df['rlnImagePixelSize'].iloc[0]
    distance_threshold_pixels = distance_threshold / pixel_size
    
    if verbose:
        print(f"  Pixel size: {pixel_size} Angstroms/pixel")
        print(f"  Distance threshold: {distance_threshold} Angstroms = {distance_threshold_pixels:.2f} pixels")
    
    # Extract coordinates from original points
    original_coords = original_df[required_coords].values
    
    scores = {}
    tube_ids = fitted_df['rlnHelicalTubeID'].unique()
    
    for tube_id in tube_ids:
        # Get all points for this tube
        tube_df = fitted_df[fitted_df['rlnHelicalTubeID'] == tube_id]
        tube_coords = tube_df[required_coords].values
        
        # Track coverage and spreading metrics
        points_with_coverage = 0
        total_matched_original_points = 0
        
        for tube_point in tube_coords:
            # Calculate distances from this tube point to all original points (in pixels)
            distances = np.sqrt(np.sum((original_coords - tube_point)**2, axis=1))
            
            # Find all original points within threshold
            within_threshold = distances <= distance_threshold_pixels
            num_matches = np.sum(within_threshold)
            
            if num_matches > 0:
                points_with_coverage += 1
                total_matched_original_points += num_matches
        
        # Calculate coverage score
        total_fitted_points = len(tube_coords)
        coverage_score = points_with_coverage / total_fitted_points if total_fitted_points > 0 else 0.0
        
        # Calculate spreading score
        if points_with_coverage > 0:
            avg_matches = total_matched_original_points / points_with_coverage
            spreading_score = 1.0 / avg_matches  # Ideal = 1.0 (one-to-one), lower = more spreading
        else:
            avg_matches = 0.0
            spreading_score = None
        
        scores[str(tube_id)] = {
            'coverage': coverage_score,
            'spreading': spreading_score,
            'avg_matches': avg_matches,
            'total_fitted_points': total_fitted_points,
            'fitted_points_with_coverage': points_with_coverage
        }
    
    return scores


def print_tube_scores(scores: dict, title: str = "Tube Scores"):
    """
    Print tube scores in a formatted way.
    
    Args:
        scores: Dictionary returned by calculate_tube_scores()
        title: Title to display above the scores
    """
    print(f"\n{title}:")
    for tube_id, score_dict in scores.items():
        print(f"  Tube ID {tube_id}:")
        print(f"    Coverage Score = {score_dict['coverage']:.2f} "
              f"({score_dict['fitted_points_with_coverage']}/{score_dict['total_fitted_points']} fitted points)")
        if score_dict['spreading'] is not None:
            print(f"    Spreading Score = {score_dict['spreading']:.2f} "
                  f"(avg {score_dict['avg_matches']:.2f} original points per fitted point)")
        else:
            print(f"    Spreading Score = N/A (no matching points)")


def filter_tubes_by_threshold(fitted_df: pd.DataFrame, scores: dict, 
                              coverage_threshold: float = None,
                              spreading_threshold: float = None) -> pd.DataFrame:
    """
    Filter tubes based on coverage and/or spreading score thresholds.
    
    Args:
        fitted_df: Original DataFrame with rlnHelicalTubeID
        scores: Dictionary returned by calculate_tube_scores()
        coverage_threshold: Minimum coverage score to keep (0.0 to 1.0), or None to skip
        spreading_threshold: Minimum spreading score to keep (0.0 to 1.0), or None to skip
    
    Returns:
        Filtered DataFrame containing only tubes that pass the thresholds
    """
    tube_ids_to_keep = []
    
    for tube_id, score_dict in scores.items():
        keep = True
        
        # Check coverage threshold
        if coverage_threshold is not None:
            if score_dict['coverage'] < coverage_threshold:
                keep = False
        
        # Check spreading threshold
        if spreading_threshold is not None:
            if score_dict['spreading'] is None or score_dict['spreading'] < spreading_threshold:
                keep = False
        
        if keep:
            tube_ids_to_keep.append(tube_id)
    
    # Filter the dataframe
    # Convert tube IDs to the same type as in the dataframe
    fitted_df_copy = fitted_df.copy()
    fitted_df_copy['rlnHelicalTubeID'] = fitted_df_copy['rlnHelicalTubeID'].astype(str)
    
    filtered_df = fitted_df_copy[fitted_df_copy['rlnHelicalTubeID'].isin(tube_ids_to_keep)]
    
    return filtered_df


def get_summary_statistics(scores: dict) -> dict:
    """
    Calculate summary statistics across all tubes.
    
    Args:
        scores: Dictionary returned by calculate_tube_scores()
    
    Returns:
        Dictionary with summary statistics
    """
    coverage_scores = [s['coverage'] for s in scores.values()]
    spreading_scores = [s['spreading'] for s in scores.values() if s['spreading'] is not None]
    
    summary = {
        'num_tubes': len(scores),
        'coverage_mean': np.mean(coverage_scores) if coverage_scores else 0.0,
        'coverage_std': np.std(coverage_scores) if coverage_scores else 0.0,
        'coverage_min': np.min(coverage_scores) if coverage_scores else 0.0,
        'coverage_max': np.max(coverage_scores) if coverage_scores else 0.0,
        'spreading_mean': np.mean(spreading_scores) if spreading_scores else 0.0,
        'spreading_std': np.std(spreading_scores) if spreading_scores else 0.0,
        'spreading_min': np.min(spreading_scores) if spreading_scores else 0.0,
        'spreading_max': np.max(spreading_scores) if spreading_scores else 0.0,
        'tubes_with_no_coverage': sum(1 for s in scores.values() if s['spreading'] is None)
    }
    
    return summary