#!/usr/bin/env python3
"""
Corrects Euler angles (Rot, Tilt, Psi) in a RELION star file using 2-step 
iterative polynomial fitting and robust outlier detection.

For any particle identified as an outlier in ANY of the three angles, ALL three 
angles are replaced by the value extrapolated from the final, robust polynomial fit.

The output is a new STAR file named [input_filename]_corrected.star.

Usage:
    # Corrects angles and prints RMSE to console
    python star_angle_corrector.py input.star
    
    # Corrects angles and saves RMSE to CSV file
    python star_angle_corrector.py input.star --out_rmse tube_quality.csv

@Builab 2025
"""

import argparse
import numpy as np
import starfile
from pathlib import Path
import pandas as pd


def read_star(file_path: str):
    """Read STAR file into DataFrame."""
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    return df


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
        Tuple: (final_fitted_angles, total_outliers_mask, final_rmse, final_poly_object)
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

    return final_fitted_angles, total_outliers_mask, final_rmse, final_poly


def process_star_file(star_path, rmse_output_path=None):
    """
    Reads star file, performs iterative correction, and saves outputs.
    """
    print(f"Reading star file: {star_path}")
    df = read_star(star_path)
    
    # --- Validation ---
    required_cols = ['rlnHelicalTubeID', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Total particles: {len(df)}")
    print(f"Total tubes: {df['rlnHelicalTubeID'].nunique()}")
    
    # --- Setup Output ---
    output_path = f"{Path(star_path).stem}_corrected.star"
    df_corrected = df.copy()
    grouped = df.groupby('rlnHelicalTubeID')
    
    # --- Data Structures for Correction and Reporting ---
    tube_metrics = []
    angle_names = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']

    # --- Iterative Correction Loop ---
    for tube_id, tube_data in grouped:
        particle_indices = np.arange(len(tube_data))
        tube_metric = {'rlnHelicalTubeID': tube_id}
        tube_masks = {}
        tube_polys = {}
        
        # 1. Run iterative fit and detection for all three angles
        for angle_col in angle_names:
            angles = tube_data[angle_col].values
            
            # Note: We only use the mask and polynomial from this call
            _, total_outliers_mask, final_rmse, final_poly = \
                iterative_fit_and_detect(particle_indices, angles, order=2, n_iterations=2)
            
            tube_metric[f'RMSE_{angle_col}'] = final_rmse
            tube_masks[angle_col] = total_outliers_mask
            tube_polys[angle_col] = final_poly

        tube_metrics.append(tube_metric)
        
        # 2. Global Outlier Identification (Flag if outlier in ANY angle)
        global_outlier_mask = np.zeros(len(tube_data), dtype=bool)
        for angle in angle_names:
             global_outlier_mask = global_outlier_mask | tube_masks[angle]

        # 3. Extrapolation and Data Replacement
        if global_outlier_mask.any():
            indices_in_tube = tube_data.index
            outlier_indices_original_df = indices_in_tube[global_outlier_mask]
            outlier_indices_local = particle_indices[global_outlier_mask]
            
            for angle in angle_names:
                poly = tube_polys[angle]
                # Calculate new (extrapolated) values using the final robust fit
                extrapolated_values = poly(outlier_indices_local)
                
                # Replace the values in the corrected DataFrame copy
                df_corrected.loc[outlier_indices_original_df, angle] = extrapolated_values

    # --- Output STAR File ---
    starfile.write(df_corrected, output_path, overwrite=True)
    print(f"\nExtrapolated STAR file saved to: {output_path}")

    # --- RMSE Reporting ---
    if tube_metrics:
        metrics_df = pd.DataFrame(tube_metrics)
        metrics_df.columns = ['TubeID', 'RMSE_Rot_deg', 'RMSE_Tilt_deg', 'RMSE_Psi_deg']

        if rmse_output_path:
            # Save to CSV
            metrics_df.to_csv(rmse_output_path, index=False, float_format='%.2f')
            print(f"\nIndividual Tube RMSE saved to CSV: {rmse_output_path}")
        else:
            # Print to console
            print("\n--- Individual Tube RMSE (Tube Quality Check) ---")
            metrics_df.columns = ['TubeID', 'RMSE_Rot (°)', 'RMSE_Tilt (°)', 'RMSE_Psi (°)']
            
            for col in [c for c in metrics_df.columns if c.startswith('RMSE_')]:
                metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
            print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Corrects Euler angles for helical tubes using iterative polynomial fitting.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python star_angle_corrector.py particles.star --out_rmse tube_quality.csv
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Input star file')
    parser.add_argument('--out_rmse', type=str, default=None,
                       help='Output file to save individual tube RMSE values (CSV format). If not specified, prints to console.')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    try:
        process_star_file(args.input, args.out_rmse)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())