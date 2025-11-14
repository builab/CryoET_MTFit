import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys


def visualize_star_df(df: pd.DataFrame, input_file: str = "", output_path: str = None):
    """
    Visualizes the 3D coordinates (rlnCoordinateX,Y,Z) from a loaded STAR file DataFrame.

    If 'rlnHelicalTubeID' is present, points belonging to the same ID are
    connected by a colored line. Otherwise, a simple scatter plot is generated.

    Args:
        df: A pandas DataFrame containing STAR file data, including 
            'rlnCoordinateX', 'rlnCoordinateY', and 'rlnCoordinateZ'.
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
