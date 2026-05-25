import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

# Import scoring utilities
from .scoring import calculate_tube_scores, print_tube_scores



def visualize_star_df(df: pd.DataFrame, input_file: str = "", output_path: str = None):
    """
    Visualizes the 3D coordinates (rlnCoordinateX,Y,Z) from a loaded STAR file DataFrame.

    If 'rlnHelicalTubeID' is present, points belonging to the same ID are
    connected by a colored line. Otherwise, a simple scatter plot is generated.

    Args:
        df: A pandas DataFrame containing STAR file data, including 
            'rlnCoordinateX', 'rlnCoordinateY', and 'rlnCoordinateZ'.
        input_file: Name of the input file for display purposes.
        output_path: Optional path to save the plot as an interactive HTML file.
                     If None, the plot attempts to open in a web browser.
    """
    # Check for mandatory coordinate columns
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Error: The input DataFrame is missing required coordinate columns: {', '.join(missing_cols)}")
        print("Visualization aborted.")
        return

    # --- Plotting Logic ---
    if 'rlnHelicalTubeID' in df.columns:
        print(f"Detected 'rlnHelicalTubeID' with {df['rlnHelicalTubeID'].nunique()} unique tubes.")
        print("Plotting coordinates as continuous 3D lines.")

        # Ensure the Tube ID is treated as categorical (string) for discrete coloring
        df['rlnHelicalTubeID'] = df['rlnHelicalTubeID'].astype(str)

        fig = go.Figure()
        
        # Define the color sequence for plotting
        colors = px.colors.qualitative.Plotly
        tube_ids = df['rlnHelicalTubeID'].unique()
        
        for i, tube_id in enumerate(tube_ids):
            # Select all rows belonging to the current tube ID
            # Use .copy() to avoid SettingWithCopyWarning
            tube_df = df[df['rlnHelicalTubeID'] == tube_id].copy()
            
            # Select a color for the current tube, cycling through the palette
            color = colors[i % len(colors)]
            
            # 1. Scatter trace for individual particle visibility and hover information
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8),
                name=f'Tube ID: {tube_id}',
                # Display the Tube ID on hover for each particle
                hovertext=[f'Tube ID: {tube_id}' for _ in range(len(tube_df))],
                hoverinfo='text',
            ))

            # 2. Line trace to connect the points in the order they appear in the file
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False, # Hide the line trace from the legend
                hoverinfo='skip' # Do not show hover text for the line segments
            ))

        title = f"3D Microtubule Traces <br>{input_file}"
        
    else:
        print("No 'rlnHelicalTubeID' found. Plotting as a simple 3D scatter plot.")
        
        # Simple scatter plot using Plotly Express
        fig = px.scatter_3d(
            df, 
            x='rlnCoordinateX', 
            y='rlnCoordinateY', 
            z='rlnCoordinateZ',
            opacity=0.8,
            title= f"3D Scatter Plot of All Coordinates<br>{input_file}",
        )
        # Update point size for visibility
        fig.update_traces(marker=dict(size=4, color='#3b82f6'))
        title = f"3D Scatter Plot of All Microtubule Coordinates<br>{input_file}"
        
    # --- Common Layout Setup ---
    fig.update_layout(
        scene=dict(
            xaxis_title='rlnCoordinateX',
            yaxis_title='rlnCoordinateY',
            zaxis_title='rlnCoordinateZ',
            # Ensures a sensible aspect ratio for structural data
            aspectmode='data' 
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )
    
    # --- Output Logic ---
    if output_path:
        print(f"Saving interactive 3D plot to file: {output_path}")
        fig.write_html(output_path)
        print("Save complete. Transfer the file to your local machine to view it in a browser.")
    else:
        # Display the interactive plot (default for local execution)
        print("Displaying interactive 3D plot in browser window.")
        fig.show()


def calculate_tube_coverage_scores(main_df: pd.DataFrame, overlay_df: pd.DataFrame, 
                                   distance_threshold: float) -> dict:
    """
    Calculate coverage and spreading scores for each tube in main_df based on proximity to points in overlay_df.
    
    Coverage score: What fraction of fitted points have at least one original point nearby?
    Spreading score: How concentrated are the original points around fitted points? (1.0 = ideal one-to-one)
    
    Args:
        main_df: DataFrame with rlnHelicalTubeID (fitted tubes)
        overlay_df: DataFrame without rlnHelicalTubeID (original scatter points)
        distance_threshold: Maximum distance (in Angstroms) to consider a point as "covered"
    
    Returns:
        Dictionary mapping tube_id to dict with 'coverage', 'spreading', and 'avg_matches'
    """
    # Get pixel size to convert threshold from Angstroms to pixels
    if 'rlnImagePixelSize' not in main_df.columns:
        print("Error: rlnImagePixelSize not found in main file. Cannot calculate distances.")
        return {}
    
    if 'rlnImagePixelSize' not in overlay_df.columns:
        print("Error: rlnImagePixelSize not found in overlay file. Cannot calculate distances.")
        return {}
    
    # Use pixel size from main file (should be the same in both files)
    pixel_size = main_df['rlnImagePixelSize'].iloc[0]
    
    # Convert distance threshold from Angstroms to pixels
    distance_threshold_pixels = distance_threshold / pixel_size
    
    print(f"  Pixel size: {pixel_size} Angstroms/pixel")
    print(f"  Distance threshold: {distance_threshold} Angstroms = {distance_threshold_pixels:.2f} pixels")
    
    # Extract coordinates from overlay (original points)
    overlay_coords = overlay_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    
    scores = {}
    tube_ids = main_df['rlnHelicalTubeID'].unique()
    
    for tube_id in tube_ids:
        # Get all points for this tube
        tube_df = main_df[main_df['rlnHelicalTubeID'] == tube_id]
        tube_coords = tube_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        
        # Track coverage and spreading metrics
        points_with_coverage = 0
        total_matched_original_points = 0
        matched_original_indices = set()
        
        for tube_point in tube_coords:
            # Calculate distances from this tube point to all original points (in pixels)
            distances = np.sqrt(np.sum((overlay_coords - tube_point)**2, axis=1))
            
            # Find all original points within threshold
            within_threshold = distances <= distance_threshold_pixels
            num_matches = np.sum(within_threshold)
            
            if num_matches > 0:
                points_with_coverage += 1
                total_matched_original_points += num_matches
                
                # Track which original points were matched (to avoid double-counting)
                matched_indices = np.where(within_threshold)[0]
                matched_original_indices.update(matched_indices)
        
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
            'avg_matches': avg_matches
        }
    
    return scores


def visualize_overlay_star_dfs(main_df: pd.DataFrame, overlay_df: pd.DataFrame, 
                                main_filename: str, overlay_filename: str, 
                                output_path: str = None, distance_threshold: float = None):
    """
    Visualizes two STAR files overlaid in the same 3D plot.
    The main file is displayed normally, and the overlay file is shown with transparency.

    Args:
        main_df: Main pandas DataFrame containing STAR file data.
        overlay_df: Overlay pandas DataFrame containing STAR file data.
        main_filename: Filename of the main file.
        overlay_filename: Filename of the overlay file.
        output_path: Optional path to save the plot as an interactive HTML file.
                     If None, the plot attempts to open in a web browser.
        distance_threshold: Optional distance threshold for calculating coverage scores.
                           Only used when main has rlnHelicalTubeID and overlay doesn't.
    """
    # Check for mandatory coordinate columns in both DataFrames
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    
    for df, name in [(main_df, main_filename), (overlay_df, overlay_filename)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: File '{name}' is missing required columns: {', '.join(missing_cols)}")
            print("Visualization aborted.")
            return
    
    print(f"--- Creating overlay visualization ---")
    
    # Check if we should calculate coverage scores
    has_main_tubes = 'rlnHelicalTubeID' in main_df.columns
    has_overlay_tubes = 'rlnHelicalTubeID' in overlay_df.columns
    
    calculate_scores = (
        distance_threshold is not None and
        has_main_tubes and
        not has_overlay_tubes and
        calculate_tube_scores is not None
    )
    
    if calculate_scores:
        print(f"--- Calculating tube coverage and spreading scores (threshold: {distance_threshold} Angstroms) ---")
        try:
            scores = calculate_tube_scores(main_df, overlay_df, distance_threshold, verbose=True)
            print_tube_scores(scores, title="Tube Coverage and Spreading Scores")
        except Exception as e:
            print(f"Error calculating scores: {e}")
            print("Continuing with visualization...")
    
    fig = go.Figure()
    
    # --- Plot Main File (Normal Opacity) ---
    print(f"Main file '{main_filename}': ", end="")
    if 'rlnHelicalTubeID' in main_df.columns:
        print(f"Detected {main_df['rlnHelicalTubeID'].nunique()} tubes with rlnHelicalTubeID")
        
        main_df['rlnHelicalTubeID'] = main_df['rlnHelicalTubeID'].astype(str)
        colors = px.colors.qualitative.Plotly
        tube_ids = main_df['rlnHelicalTubeID'].unique()
        
        for i, tube_id in enumerate(tube_ids):
            tube_df = main_df[main_df['rlnHelicalTubeID'] == tube_id].copy()
            color = colors[i % len(colors)]
            
            # Scatter trace with normal opacity
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8),
                name=f'{main_filename} - Tube {tube_id}',
                legendgroup='main',
                hovertext=[f'{main_filename}<br>Tube ID: {tube_id}' for _ in range(len(tube_df))],
                hoverinfo='text',
            ))
            
            # Line trace
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup='main',
                hoverinfo='skip'
            ))
    else:
        print("No rlnHelicalTubeID found, plotting as scatter")
        
        fig.add_trace(go.Scatter3d(
            x=main_df['rlnCoordinateX'],
            y=main_df['rlnCoordinateY'],
            z=main_df['rlnCoordinateZ'],
            mode='markers',
            marker=dict(size=4, color='#3b82f6', opacity=0.8),
            name=main_filename,
            legendgroup='main',
            hovertext=[main_filename for _ in range(len(main_df))],
            hoverinfo='text',
        ))
    
    # --- Plot Overlay File (With Transparency) ---
    print(f"Overlay file '{overlay_filename}': ", end="")
    overlay_color = '#3b82f6'  # Same blue color as main scatter
    overlay_opacity = 0.3  # Transparency for overlay
    overlay_line_opacity = 0.4
    
    if 'rlnHelicalTubeID' in overlay_df.columns:
        print(f"Detected {overlay_df['rlnHelicalTubeID'].nunique()} tubes with rlnHelicalTubeID")
        
        overlay_df['rlnHelicalTubeID'] = overlay_df['rlnHelicalTubeID'].astype(str)
        tube_ids = overlay_df['rlnHelicalTubeID'].unique()
        
        for tube_id in tube_ids:
            tube_df = overlay_df[overlay_df['rlnHelicalTubeID'] == tube_id].copy()
            
            # Scatter trace with transparency
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='markers',
                marker=dict(size=4, color=overlay_color, opacity=overlay_opacity),
                name=f'{overlay_filename} - Tube {tube_id}',
                legendgroup='overlay',
                hovertext=[f'{overlay_filename}<br>Tube ID: {tube_id}' for _ in range(len(tube_df))],
                hoverinfo='text',
            ))
            
            # Line trace with transparency
            fig.add_trace(go.Scatter3d(
                x=tube_df['rlnCoordinateX'],
                y=tube_df['rlnCoordinateY'],
                z=tube_df['rlnCoordinateZ'],
                mode='lines',
                line=dict(color=overlay_color, width=2),
                opacity=overlay_line_opacity,
                showlegend=False,
                legendgroup='overlay',
                hoverinfo='skip'
            ))
    else:
        print("No rlnHelicalTubeID found, plotting as scatter")
        
        fig.add_trace(go.Scatter3d(
            x=overlay_df['rlnCoordinateX'],
            y=overlay_df['rlnCoordinateY'],
            z=overlay_df['rlnCoordinateZ'],
            mode='markers',
            marker=dict(size=4, color=overlay_color, opacity=overlay_opacity),
            name=overlay_filename,
            legendgroup='overlay',
            hovertext=[overlay_filename for _ in range(len(overlay_df))],
            hoverinfo='text',
        ))
    
    # --- Layout Setup ---
    title = f"Overlay: {main_filename} (solid) + {overlay_filename} (transparent)"
    
    fig.update_layout(
        scene=dict(
            xaxis_title='rlnCoordinateX',
            yaxis_title='rlnCoordinateY',
            zaxis_title='rlnCoordinateZ',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # --- Output Logic ---
    if output_path:
        print(f"Saving interactive 3D overlay plot to file: {output_path}")
        fig.write_html(output_path)
        print("Save complete. Transfer the file to your local machine to view it in a browser.")
    else:
        print("Displaying interactive 3D overlay plot in browser window.")
        fig.show()