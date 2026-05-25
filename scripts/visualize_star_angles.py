#!/usr/bin/env python3
"""
Visualize Euler angles (Rot, Tilt, Psi) and inter-particle distances for helical tubes from RELION star file.

This script creates a multi-panel visualization showing how the three Euler angles
and inter-particle distances vary along each helical tube. Each tube is colored consistently across all subplots.

The --fit_line option now implements 2-step iterative polynomial fitting and 
robust outlier detection (MAD Z-score > 3.5), highlighting outliers from each 
step with different shades of red.

The --histogram option plots distributions of angles across all particles regardless of tube ID.

The output logic is:
- Plot: If --output is provided, saves to file. Otherwise, attempts to display interactively (plt.show()).
- RMSE: If --out_rmse is provided, saves to CSV. Otherwise, prints the table to the console.

Usage:
    python visualize_angles.py input.star --fit_line [--output plot.png] [--out_rmse rmse.csv]
    python visualize_angles.py input.star --histogram [--output hist.png]

@Builab 2025
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import starfile
from pathlib import Path
import pandas as pd


# Constants for Outlier Colors
COLOR_OUTLIER_ITER_1 = '#990000' # Deep Red (most egregious)
COLOR_OUTLIER_ITER_2 = '#ff6666' # Lighter Red (detected after refinement)


def read_star(file_path: str):
    """
    Read STAR file into DataFrame.
    """
    data = starfile.read(file_path)
    if isinstance(data, dict):
        # Has optics block
        if 'particles' in data:
            df = data['particles']
            optics = data.get('optics', None)
            return df, optics
        else:
            return data, None
    else:
        # No optics block
        return data, None


def get_pixel_size(df, optics):
    """
    Get pixel size from optics block or from particle data.
    """
    if optics is not None and 'rlnImagePixelSize' in optics.columns:
        return optics['rlnImagePixelSize'].iloc[0]
    elif 'rlnImagePixelSize' in df.columns:
        return df['rlnImagePixelSize'].iloc[0]
    else:
        raise ValueError("Could not find rlnImagePixelSize in optics or particles block")


def calculate_distances(tube_data, pixel_size):
    """
    Calculate Euclidean distances between consecutive particles in Angstrom.
    
    Args:
        tube_data: DataFrame for a single tube
        pixel_size: Pixel size in Angstrom
    
    Returns:
        Array of distances (length = n_particles - 1)
    """
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    
    # Calculate differences between consecutive points
    diffs = np.diff(coords, axis=0)
    
    # Calculate Euclidean distances in pixels
    distances_pixels = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Convert to Angstrom
    distances_angstrom = distances_pixels * pixel_size
    
    return distances_angstrom


def fit_polynomial(x, y, order=2):
    """
    Fit polynomial to data and calculate residuals.
    
    Args:
        x: X coordinates (particle indices)
        y: Y coordinates (angle values)
        order: Polynomial order (default: 2)
    
    Returns:
        Tuple of (fitted_y, residuals, rmse)
    """
    # Ensure there is enough data for fitting
    if len(x) <= order:
        return np.full_like(y, np.nan), np.full_like(y, np.nan), np.nan

    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)
    fitted_y = poly(x)
    residuals = y - fitted_y
    # Calculate RMSE only on the points used for the current fit
    rmse = np.sqrt(np.mean(residuals**2))
    return fitted_y, residuals, rmse


def robust_mad_outlier_detection(residuals, threshold=3.5):
    """
    Detects outliers using the Modified Z-score based on the Median Absolute Deviation (MAD).
    
    Args:
        residuals: 1D numpy array of residuals.
        threshold: Modified Z-score threshold (default 3.5).
        
    Returns:
        1D boolean array where True indicates an outlier.
    """
    # Need enough points to reliably calculate MAD (5 is a typical minimum)
    if len(residuals) < 5: 
        return np.zeros_like(residuals, dtype=bool)

    # 1. Calculate MAD (Median Absolute Deviation)
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    if mad == 0:
        # If MAD is 0, check if any point deviates from the median
        return np.abs(residuals - median_residual) > 1e-6 

    # 2. Calculate Modified Z-score (0.6745 is the correction factor)
    modified_z_score = 0.6745 * (residuals - median_residual) / mad
    
    # 3. Apply Threshold
    is_outlier = np.abs(modified_z_score) > threshold
    return is_outlier


def iterative_fit_and_detect(particle_indices, angles, order=2, n_iterations=2):
    """
    Performs iterative polynomial fitting and MAD outlier detection.
    
    Args:
        particle_indices: Full array of x-coordinates (particle indices).
        angles: Full array of y-coordinates (angle values).
        order: Polynomial order.
        n_iterations: Number of iterations to perform.
        
    Returns:
        Tuple: (final_fitted_angles, total_outliers_mask, final_rmse, outliers_iter_1_mask)
    """
    
    N = len(angles)
    # Mask to track all outliers found across all iterations
    total_outliers_mask = np.zeros(N, dtype=bool)
    # Mask to specifically track outliers from the first iteration
    outliers_iter_1_mask = np.zeros(N, dtype=bool)
    
    # Indices of points currently considered inliers for fitting
    current_inlier_indices = np.arange(N)
    
    final_fitted_angles = np.full(N, np.nan)
    final_rmse = np.nan
    
    for i in range(n_iterations):
        
        # 1. Get current inlier data
        x_in = particle_indices[current_inlier_indices]
        y_in = angles[current_inlier_indices]
        
        # Skip if not enough data remains
        if len(x_in) <= order:
            break
            
        # 2. Fit polynomial to inliers
        coeffs = np.polyfit(x_in, y_in, order)
        poly = np.poly1d(coeffs)
        
        # 3. Calculate fitted line and residuals for ALL original points
        fitted_angles_all = poly(particle_indices)
        residuals_all = angles - fitted_angles_all
        
        # 4. Detect outliers among the current inliers
        # We only look for outliers among the points that HAVEN'T been excluded yet
        residuals_for_detection = residuals_all[current_inlier_indices]
        is_outlier_in_subset = robust_mad_outlier_detection(residuals_for_detection)
        
        # 5. Map detected outliers back to the original index space
        outlier_indices_original = current_inlier_indices[is_outlier_in_subset]
        
        if len(outlier_indices_original) == 0:
            # No new outliers found, stop iterating
            final_fitted_angles = fitted_angles_all
            final_rmse = np.sqrt(np.mean((angles[~total_outliers_mask] - fitted_angles_all[~total_outliers_mask])**2))
            break
            
        # 6. Update the total outlier mask and the iteration 1 mask
        total_outliers_mask[outlier_indices_original] = True
        if i == 0:
            outliers_iter_1_mask[outlier_indices_original] = True
            
        # 7. Update the current inlier indices for the next iteration
        current_inlier_indices = np.where(~total_outliers_mask)[0]
        
        # If this is the last iteration, record the final fit and RMSE on the final inliers
        if i == n_iterations - 1:
            final_fitted_angles = fitted_angles_all
            final_rmse = np.sqrt(np.mean((angles[~total_outliers_mask] - fitted_angles_all[~total_outliers_mask])**2))

    return final_fitted_angles, total_outliers_mask, final_rmse, outliers_iter_1_mask


def plot_angle_histograms(star_path, output_path=None):
    """
    Plot histograms of Euler angles across all particles.
    
    Args:
        star_path: Path to the STAR file
        output_path: Optional output file path for saving the plot
    """
    print(f"Reading star file: {star_path}")
    df, optics = read_star(star_path)
    
    required_cols = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Total particles: {len(df)}")
    if 'rlnHelicalTubeID' in df.columns:
        print(f"Total tubes: {df['rlnHelicalTubeID'].nunique()}")
    
    # Create figure with 1x3 horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fig.suptitle(f'Euler Angle Distributions (All Particles)\n{Path(star_path).name}', 
                 fontsize=14, fontweight='bold')
    
    angle_names = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    angle_labels = ['Rot (°)', 'Tilt (°)', 'Psi (°)']
    
    # Plot histograms for each angle
    for ax_idx, (angle_col, angle_label) in enumerate(zip(angle_names, angle_labels)):
        angles = df[angle_col].values
        
        # Calculate statistics
        mean_val = np.mean(angles)
        median_val = np.median(angles)
        std_val = np.std(angles)
        
        # Plot histogram
        n, bins, patches = axes[ax_idx].hist(angles, bins=50, color='steelblue', 
                                             alpha=0.7, edgecolor='black')
        
        # Add vertical lines for mean and median
        axes[ax_idx].axvline(mean_val, color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {mean_val:.1f}°')
        axes[ax_idx].axvline(median_val, color='orange', linestyle='--', 
                            linewidth=2, label=f'Median: {median_val:.1f}°')
        
        # Add statistics text box
        textstr = f'n = {len(angles)}\nμ = {mean_val:.2f}°\nσ = {std_val:.2f}°'
        axes[ax_idx].text(0.02, 0.98, textstr, transform=axes[ax_idx].transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[ax_idx].set_xlabel(angle_label, fontsize=11)
        axes[ax_idx].set_ylabel('Count', fontsize=11)
        axes[ax_idx].legend(loc='upper right', fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print("\n=== Overall Statistics (All Particles) ===")
    print(f"\nEuler Angles:")
    for angle_col, angle_label in zip(angle_names, angle_labels):
        angles = df[angle_col].values
        print(f"  {angle_label:12s}: μ={np.mean(angles):7.2f}°  σ={np.std(angles):7.2f}°  "
              f"median={np.median(angles):7.2f}°  range=[{np.min(angles):7.2f}, {np.max(angles):7.2f}]")


def plot_tube_angles(star_path, output_path=None, fit_line=False, rmse_output_path=None):
    """
    Plot Euler angles and inter-particle distances for all helical tubes in a star file, 
    with optional iterative fit and outlier detection.
    Always uses 2x2 layout for the 4 plots (3 angles + distances).
    """
    print(f"Reading star file: {star_path}")
    df, optics = read_star(star_path)
    
    required_cols = ['rlnHelicalTubeID', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for distance plotting requirements
    dist_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    missing_dist_cols = [col for col in dist_cols if col not in df.columns]
    if missing_dist_cols:
        raise ValueError(f"Missing required columns for distance calculation: {missing_dist_cols}")
    
    try:
        pixel_size = get_pixel_size(df, optics)
        print(f"Pixel size: {pixel_size} Å")
    except ValueError as e:
        raise ValueError(f"Cannot plot distances: {e}")
    
    print(f"Total particles: {len(df)}")
    print(f"Total tubes: {df['rlnHelicalTubeID'].nunique()}")
    
    grouped = df.groupby('rlnHelicalTubeID')
    n_tubes = len(grouped)
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_tubes, 20)))
    if n_tubes > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_tubes))
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    if fit_line:
        fig.suptitle(f'Euler Angle Iterative Polynomial Fit & Outlier Detection (Order 2, 2-Step MAD Z > 3.5)\n{Path(star_path).name}', 
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Helical Tube Analysis\n{Path(star_path).name}', 
                     fontsize=14, fontweight='bold')
    
    angle_names = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    angle_labels = ['Rot (°)', 'Tilt (°)', 'Psi (°)']
    
    tube_metrics = []
    tube_statistics = []
    
    # Flags to ensure legend entries are only added once
    outlier_legend_iter_1_added = False
    outlier_legend_iter_2_added = False

    # Plot each tube
    for idx, (tube_id, tube_data) in enumerate(grouped):
        color = colors[idx % len(colors)]
        particle_indices = np.arange(len(tube_data))
        tube_metric = {'rlnHelicalTubeID': tube_id}
        tube_stat = {'rlnHelicalTubeID': tube_id}
        
        # Calculate statistics for angles
        for angle_col in angle_names:
            angles = tube_data[angle_col].values
            tube_stat[f'{angle_col}_min'] = np.min(angles)
            tube_stat[f'{angle_col}_max'] = np.max(angles)
            tube_stat[f'{angle_col}_avg'] = np.mean(angles)
        
        # Plot angles
        for ax_idx, (angle_col, angle_label) in enumerate(zip(angle_names, angle_labels)):
            angles = tube_data[angle_col].values
            
            if fit_line:
                # Perform 2-step iterative detection
                fitted_angles, total_outliers_mask, final_rmse, outliers_iter_1_mask = \
                    iterative_fit_and_detect(particle_indices, angles, order=2, n_iterations=2)
                
                tube_metric[f'RMSE_{angle_col}'] = final_rmse
                
                # Identify points for plotting
                inlier_mask = ~total_outliers_mask
                outlier_iter_2_mask = total_outliers_mask & ~outliers_iter_1_mask # Outliers found in step 2 only
                
                inlier_indices = particle_indices[inlier_mask]
                inlier_angles = angles[inlier_mask]
                
                outlier_iter_1_indices = particle_indices[outliers_iter_1_mask]
                outlier_iter_1_angles = angles[outliers_iter_1_mask]
                
                outlier_iter_2_indices = particle_indices[outlier_iter_2_mask]
                outlier_iter_2_angles = angles[outlier_iter_2_mask]

                # 1. Plot fitted line (final, robust fit)
                axes[ax_idx].plot(particle_indices, fitted_angles, 
                                color=color, alpha=0.9, linewidth=2.0, linestyle='-',
                                label=f'Tube {tube_id}' if ax_idx == 0 and n_tubes <= 10 else None)
                
                # 2. Plot inliers (original angles)
                axes[ax_idx].scatter(inlier_indices, inlier_angles, 
                                   color=color, alpha=0.5, s=10)
                                   
                # 3. Plot 1st Iteration Outliers (Deep Red)
                label_1 = 'Outlier (Iter 1 - Deep Red)' if ax_idx == 0 and not outlier_legend_iter_1_added else None
                axes[ax_idx].scatter(outlier_iter_1_indices, outlier_iter_1_angles, 
                                   color=COLOR_OUTLIER_ITER_1, marker='o', alpha=1.0, s=25, zorder=5,
                                   label=label_1)
                if label_1:
                    outlier_legend_iter_1_added = True
                
                # 4. Plot 2nd Iteration Outliers (Lighter Red)
                label_2 = 'Outlier (Iter 2 - Light Red)' if ax_idx == 0 and not outlier_legend_iter_2_added else None
                axes[ax_idx].scatter(outlier_iter_2_indices, outlier_iter_2_angles, 
                                   color=COLOR_OUTLIER_ITER_2, marker='s', alpha=1.0, s=20, zorder=4,
                                   label=label_2)
                if label_2:
                    outlier_legend_iter_2_added = True
            
            else:
                # Plot raw angles
                axes[ax_idx].plot(particle_indices, angles, 
                                color=color, alpha=0.7, linewidth=1.5,
                                label=f'Tube {tube_id}' if ax_idx == 0 else None)
                axes[ax_idx].scatter(particle_indices, angles, 
                                   color=color, alpha=0.5, s=10)
        
        # Plot distances
        distances = calculate_distances(tube_data, pixel_size)
        # Distance indices are between particles (n-1 points)
        distance_indices = particle_indices[:-1] + 0.5  # Plot at midpoints
        
        # Calculate statistics for distances
        tube_stat['Distance_min'] = np.min(distances)
        tube_stat['Distance_max'] = np.max(distances)
        tube_stat['Distance_avg'] = np.mean(distances)
        
        if fit_line:
            # Perform 2-step iterative detection with linear fit (order 1)
            fitted_distances, total_outliers_mask_dist, final_rmse_dist, outliers_iter_1_mask_dist = \
                iterative_fit_and_detect(distance_indices, distances, order=1, n_iterations=2)
            
            tube_metric['RMSE_Distance'] = final_rmse_dist
            
            # Identify points for plotting
            inlier_mask_dist = ~total_outliers_mask_dist
            outlier_iter_2_mask_dist = total_outliers_mask_dist & ~outliers_iter_1_mask_dist
            
            inlier_indices_dist = distance_indices[inlier_mask_dist]
            inlier_distances = distances[inlier_mask_dist]
            
            outlier_iter_1_indices_dist = distance_indices[outliers_iter_1_mask_dist]
            outlier_iter_1_distances = distances[outliers_iter_1_mask_dist]
            
            outlier_iter_2_indices_dist = distance_indices[outlier_iter_2_mask_dist]
            outlier_iter_2_distances = distances[outlier_iter_2_mask_dist]
            
            # Plot fitted line (linear fit)
            axes[3].plot(distance_indices, fitted_distances, 
                        color=color, alpha=0.9, linewidth=2.0, linestyle='-',
                        label=f'Tube {tube_id}' if n_tubes <= 10 else None)
            
            # Plot inliers
            axes[3].scatter(inlier_indices_dist, inlier_distances, 
                           color=color, alpha=0.5, s=10)
            
            # Plot outliers (use same legend logic as angles)
            axes[3].scatter(outlier_iter_1_indices_dist, outlier_iter_1_distances, 
                           color=COLOR_OUTLIER_ITER_1, marker='o', alpha=1.0, s=25, zorder=5)
            
            axes[3].scatter(outlier_iter_2_indices_dist, outlier_iter_2_distances, 
                           color=COLOR_OUTLIER_ITER_2, marker='s', alpha=1.0, s=20, zorder=4)
        else:
            # Plot raw distances
            axes[3].plot(distance_indices, distances, 
                        color=color, alpha=0.7, linewidth=1.5,
                        label=f'Tube {tube_id}' if n_tubes <= 10 else None)
            axes[3].scatter(distance_indices, distances, 
                           color=color, alpha=0.5, s=10)
            axes[3].set_ylim(0, 100)

        
        if fit_line:
            tube_metrics.append(tube_metric)
        
        tube_statistics.append(tube_stat)

    
    # Format angle subplots
    for ax_idx, (angle_col, angle_label) in enumerate(zip(angle_names, angle_labels)):
        axes[ax_idx].set_xlabel('Particle Index (within tube)', fontsize=11)
        axes[ax_idx].set_ylabel(angle_label, fontsize=11)
        
        axes[ax_idx].grid(True, alpha=0.3)
        
        # Set y-axis limits 
        if angle_col == 'rlnAngleTilt':
            axes[ax_idx].set_ylim(-10, 190)
        else:
            axes[ax_idx].set_ylim(-190, 190)
        
        # Add a horizontal line at y=0
        y_min, y_max = axes[ax_idx].get_ylim()
        if y_min < 0 < y_max:
            axes[ax_idx].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Format distance subplot
    axes[3].set_xlabel('Particle Index (within tube)', fontsize=11)
    axes[3].set_ylabel('Inter-particle Distance (Å)', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    if n_tubes <= 10 or fit_line:
        # Combine tube colors and the two outlier markers in the legend
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    # --- PLOT SAVING LOGIC: Save if output_path is provided, otherwise show (interactive) ---
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        # Reverting to interactive mode for local execution as requested
        plt.show() 
    
    plt.close()
    
    # --- PRINT STATISTICS FOR EACH TUBE ---
    if tube_statistics:
        stats_df = pd.DataFrame(tube_statistics)
        print("\n=== Statistics per Helical Tube ===")
        
        # Reorder columns for better readability
        cols = ['rlnHelicalTubeID', 
                'rlnAngleRot_min', 'rlnAngleRot_max', 'rlnAngleRot_avg',
                'rlnAngleTilt_min', 'rlnAngleTilt_max', 'rlnAngleTilt_avg',
                'rlnAnglePsi_min', 'rlnAnglePsi_max', 'rlnAnglePsi_avg',
                'Distance_min', 'Distance_max', 'Distance_avg']
        stats_df = stats_df[cols]
        stats_df.columns = ['TubeID', 
                           'Rot_min', 'Rot_max', 'Rot_avg',
                           'Tilt_min', 'Tilt_max', 'Tilt_avg',
                           'Psi_min', 'Psi_max', 'Psi_avg',
                           'Dist_min(Å)', 'Dist_max(Å)', 'Dist_avg(Å)']
        
        # Format numerical columns to 2 decimal places
        for col in stats_df.columns:
            if col != 'TubeID':
                stats_df[col] = stats_df[col].map('{:.2f}'.format)
        
        print(stats_df.to_string(index=False))
    
    # --- RMSE REPORTING LOGIC: Save to CSV if path provided, otherwise print to console ---
    if fit_line and tube_metrics:
        metrics_df = pd.DataFrame(tube_metrics)
        metrics_df.columns = ['TubeID', 'RMSE_Rot_deg', 'RMSE_Tilt_deg', 'RMSE_Psi_deg', 'RMSE_Distance_Ang']

        if rmse_output_path:
            # Save to CSV
            metrics_df.to_csv(rmse_output_path, index=False, float_format='%.2f')
            print(f"\nIndividual Tube RMSE saved to CSV: {rmse_output_path}")
        else:
            # Print to console
            print("\n--- Individual Tube RMSE (Tube Quality Check) ---")
            metrics_df.columns = ['TubeID', 'RMSE_Rot (°)', 'RMSE_Tilt (°)', 'RMSE_Psi (°)', 'RMSE_Distance (Å)']
            
            # Format to 2 decimal places for printing
            for col in [c for c in metrics_df.columns if c.startswith('RMSE_')]:
                metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
            print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Euler angles and inter-particle distances for helical tubes from RELION star file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display fit/outliers plot interactively AND print RMSE to console
  python visualize_star_angles.py particles.star --fit_line
  
  # Save fit/outliers plot to file AND save RMSE to CSV
  python visualize_star_angles.py particles.star --fit_line --output fit_plot.png --out_rmse tube_quality.csv
  
  # Plot angles and inter-particle distances without fitting
  python visualize_star_angles.py particles.star
  
  # Plot histograms of angle distributions (all particles)
  python visualize_star_angles.py particles.star --histogram --output hist.png
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Input star file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot file (PNG, PDF, SVG). If not specified, displays interactively (plt.show()).')
    parser.add_argument('--fit_line', action='store_true',
                       help='Fit polynomial (order 2), plot the fitted line, use 2-step iterative outlier detection, and calculate individual tube RMSE.')
    parser.add_argument('--histogram', action='store_true',
                       help='Plot histograms of angle distributions across all particles (regardless of tube ID). Cannot be used with --fit_line.')
    parser.add_argument('--out_rmse', type=str, default=None,
                       help='Output file to save individual tube RMSE values (CSV format). If not specified, prints to console.')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    # Check for incompatible options
    if args.fit_line and args.histogram:
        parser.error("--fit_line and --histogram cannot be used together. Fitting is done per-tube, while histograms aggregate all particles.")
    
    if args.histogram and args.out_rmse:
        parser.error("--out_rmse cannot be used with --histogram (RMSE is only calculated with --fit_line)")
    
    try:
        if args.histogram:
            plot_angle_histograms(args.input, args.output)
        else:
            plot_tube_angles(args.input, args.output, args.fit_line, args.out_rmse)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())