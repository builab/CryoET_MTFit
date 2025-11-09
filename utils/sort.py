#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geom/Sort for ReLAX – cross-section extraction, ellipse fit, ordering, and plotting (nm-scaled).

"""

import argparse
import numpy as np
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt

from .io import validate_dataframe

from collections import defaultdict
from typing import Dict, Tuple, Optional
from scipy.linalg import lstsq
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')


class NumpyJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles NumPy integer types by converting 
    them to standard Python integers.
    """
    def default(self, obj):
        # Check if the object is any NumPy integer type
        if isinstance(obj, np.integer):
            return int(obj)
        # Check for NumPy floats, just in case (e.g., np.float64)
        elif isinstance(obj, np.floating):
            return float(obj)
        # Let the base class default method handle other types
        return super().default(obj)
        
# ----------------------------- Basic utilities -----------------------------

def normalize_angle(angle):
    """Normalize angle to range -180..180."""
    return (angle + 180) % 360 - 180

def px_to_nm(arr_px, pixel_size_A):
    """Convert pixels to nanometers."""
    return np.asarray(arr_px, float) * (float(pixel_size_A) / 10.0)


# ----------------------------- Geometry functions -----------------------------

def calculate_perpendicular_distance(point, plane_normal, reference_point):
    """Calculate perpendicular distance from point to plane."""
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)


def find_cross_section_points(data, plane_normal, reference_point):
    """For each tube, pick the point closest to the plane."""
    cross_section = []
    for tube_id, group in data.groupby('rlnHelicalTubeID', sort=False):
        pts = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(float)
        dists = np.array([calculate_perpendicular_distance(p, plane_normal, reference_point) for p in pts])
        closest = group.iloc[int(np.argmin(dists))].copy()
        cross_section.append(closest)
    return pd.DataFrame(cross_section)


def find_shortest_tube(data):
    """Find the shortest tube and return its ID and midpoint."""
    shortest_len, shortest_mid, shortest_id = float('inf'), None, None
    for tube_id, g in data.groupby('rlnHelicalTubeID', sort=False):
        pts = g[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        length = float(np.linalg.norm(mx - mn))
        if length < shortest_len:
            shortest_len, shortest_mid, shortest_id = length, (mn + mx) / 2.0, tube_id
    return shortest_id, shortest_mid


def calculate_normal_vector(tube_points, window_size=3):
    """Calculate normal vector from local average of segment vectors."""
    n = tube_points.shape[0]
    mid = n // 2
    s = max(mid - window_size, 0)
    e = min(mid + window_size, n - 1)
    vecs = [tube_points[i + 1] - tube_points[i] for i in range(s, e)]
    avg = np.mean(vecs, axis=0)
    nv = avg / np.linalg.norm(avg)
    return nv if nv[2] >= 0 else -nv  # enforce pointing +z


def process_cross_section(data):
    """Find cross-section based on shortest tube's midpoint."""
    shortest_id, midpoint = find_shortest_tube(data)
    tube_pts = data.loc[data['rlnHelicalTubeID'] == shortest_id, 
                        ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].to_numpy(float)
    nvec = calculate_normal_vector(tube_pts)
    return find_cross_section_points(data, nvec, midpoint)


def rotate_cross_section(cross_section):
    """Rotate cross-section into Z plane using median Psi and Tilt."""
    rotated = cross_section.copy()
    psi = 90 - float(np.nanmedian(rotated['rlnAnglePsi']))
    tilt = float(np.nanmedian(rotated['rlnAngleTilt']))

    psi_rad, tilt_rad = np.radians(psi), np.radians(tilt)
    Rz = np.array([[np.cos(-psi_rad), -np.sin(-psi_rad), 0],
                   [np.sin(-psi_rad),  np.cos(-psi_rad), 0],
                   [0, 0, 1]])
    Ry = np.array([[1, 0, 0],
                   [0, np.cos(-tilt_rad), -np.sin(-tilt_rad)],
                   [0, np.sin(-tilt_rad),  np.cos(-tilt_rad)]])

    for idx, row in rotated.iterrows():
        v = np.array([row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']], float)
        v = Ry @ (Rz @ v)
        rotated.at[idx, 'rlnCoordinateX'] = v[0]
        rotated.at[idx, 'rlnCoordinateY'] = v[1]
        rotated.at[idx, 'rlnCoordinateZ'] = v[2]
    
    rotated[['rlnAngleTilt','rlnAnglePsi']] = 0
    return rotated


def enforce_consistent_orientation(df):
    """Ensure cross-section is clockwise and right-facing."""
    points = df[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    # Calculate signed area
    area = sum((points[(i+1)%len(points)][0] - points[i][0]) * 
               (points[(i+1)%len(points)][1] + points[i][1]) for i in range(len(points)))
    
    out = df.copy()
    if area > 0:  # counterclockwise → flip Y
        out['rlnCoordinateY'] = -out['rlnCoordinateY']
    if out['rlnCoordinateX'].mean() < 0:  # mostly on left → flip X
        out['rlnCoordinateX'] = -out['rlnCoordinateX']
    return out


# ----------------------------- Ellipse fitting -----------------------------

def fit_ellipse(x, y):
    """Least-squares ellipse fit. Returns dict with parameters."""
    x, y = np.asarray(x).ravel().astype(float), np.asarray(y).ravel().astype(float)
    
    if len(x) < 5:
        raise ValueError("Not enough points to fit an ellipse")

    # De-bias for stability
    mx, my = np.mean(x), np.mean(y)
    x, y = x - mx, y - my

    X = np.column_stack([x**2, x*y, y**2, x, y])
    a, _, _, _ = lstsq(X, -np.ones_like(x), lapack_driver='gelsy')
    A, B, C, D, E = a

    if B**2 - 4*A*C >= 0:
        raise ValueError("Invalid ellipse (discriminant >= 0)")

    orientation = 0.5 * np.arctan2(B, (A - C))
    cphi, sphi = np.cos(orientation), np.sin(orientation)

    A_r = A * cphi**2 - B * cphi * sphi + C * sphi**2
    C_r = A * sphi**2 + B * cphi * sphi + C * cphi**2
    D_r, E_r = D * cphi - E * sphi, D * sphi + E * cphi

    if A_r < 0 or C_r < 0:
        A_r, C_r, D_r, E_r = -A_r, -C_r, -D_r, -E_r

    X0 = mx - D_r / (2 * A_r)
    Y0 = my - E_r / (2 * C_r)
    F = 1 + (D_r**2) / (4 * A_r) + (E_r**2) / (4 * C_r)
    a_len, b_len = np.sqrt(F / A_r), np.sqrt(F / C_r)

    return {'a': a_len, 'b': b_len, 'phi': orientation, 'X0': X0, 'Y0': Y0,
            'long_axis': 2*max(a_len, b_len), 'short_axis': 2*min(a_len, b_len)}


def ellipse_points(center, axes, angle, num_points=200):
    """Generate points along an ellipse."""
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.array([axes[0]*np.cos(t), axes[1]*np.sin(t)])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    e_rot = R @ ellipse
    return e_rot + np.array([[center[0]], [center[1]]])


def angle_along_ellipse(center, axes, angle, points):
    """Return parameter angle along ellipse for given points."""
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    a, b = axes
    angles = []
    for px, py in points:
        xt, yt = px - center[0], py - center[1]
        xr = xt * cos_a - yt * sin_a
        yr = xt * sin_a + yt * cos_a
        angles.append(np.arctan2(yr / b, xr / a) + angle)
    return np.array(angles)


def calculate_rot_angles_simple(rotated_cross_section):
    """Calculate rlnAngleRot using shortest path ordering."""
    df = rotated_cross_section.copy()
    pts = df[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    
    # Start with first point
    unvisited = list(range(len(pts)))
    order = [unvisited.pop(0)]
    current_pos = pts[order[0]]
    
    # Greedy shortest path: always go to nearest unvisited point
    while unvisited:
        distances = [np.linalg.norm(pts[i] - current_pos) for i in unvisited]
        nearest_idx = unvisited[np.argmin(distances)]
        order.append(nearest_idx)
        unvisited.remove(nearest_idx)
        current_pos = pts[nearest_idx]
    
    # Assign angles based on order (descending from 180 to -180)
    n = len(order)
    angles = np.linspace(180, -180, n, endpoint=False)
    
    # Map angles to original dataframe order
    angle_map = {df.index[order[i]]: angles[i] for i in range(n)}
    df['rlnAngleRot'] = df.index.map(angle_map)
    
    return df, None


def calculate_rot_angles_ellipse(rotated_cross_section):
    """Fit ellipse and compute rlnAngleRot for each point."""
    df = rotated_cross_section.copy()
    pts = df[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    
    ellipse = fit_ellipse(pts[:,0], pts[:,1])
    center = [ellipse['X0'], ellipse['Y0']]
    axes = [ellipse['a'], ellipse['b']]
    phi = ellipse['phi']

    ang = angle_along_ellipse(center, axes, phi, pts)
    ang = np.degrees(ang) - 270
    df['rlnAngleRot'] = np.vectorize(normalize_angle)(ang)
    
    return df, ellipse


# ----------------------------- Ordering and propagation -----------------------------

def get_tube_order_from_rot(cross_section):
    """Order tubes by descending rlnAngleRot."""
    return cross_section.sort_values('rlnAngleRot', ascending=False)['rlnHelicalTubeID'].tolist()


def renumber_tube_ids(df_all, sorted_ids, cs_df):
    """Map old HelicalTubeID to new 1..N order."""
    mapping = {orig: i + 1 for i, orig in enumerate(sorted_ids)}
    df_out = df_all.copy()
    cs_out = cs_df.copy()
    df_out['rlnHelicalTubeID'] = df_out['rlnHelicalTubeID'].map(mapping)
    cs_out['rlnHelicalTubeID'] = cs_out['rlnHelicalTubeID'].map(mapping)
    return df_out, cs_out, mapping


# ----------------------------- Plotting -----------------------------
def plot_multiple_cilia(cross_sections_dict, pixel_size_A, ellipse_params_dict, out_png):
    """
    Plot cross-sections for multiple cilia in subplots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    cilia_ids = sorted(cross_sections_dict.keys())
    n_cilia = len(cilia_ids)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_cilia, figsize=(6*n_cilia, 6))
    if n_cilia == 1:
        axes = [axes]
    
    for idx, cilia_id in enumerate(cilia_ids):
        ax = axes[idx]
        cs = cross_sections_dict[cilia_id]
        ellipse_params = ellipse_params_dict[cilia_id]
        
        if cs is None:
            ax.text(0.5, 0.5, f'Cilium {cilia_id}\n(>9 tubes, not sorted)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cilium {cilia_id}')
            continue
        
        # Determine column names (handle different naming conventions)
        if 'x_rot' in cs.columns:
            x_col, y_col = 'x_rot', 'y_rot'
        elif 'rlnCoordinateX' in cs.columns:
            x_col, y_col = 'rlnCoordinateX', 'rlnCoordinateY'
        else:
            x_col, y_col = 'x', 'y'
        
        rot_col = 'rot_angle' if 'rot_angle' in cs.columns else 'rlnAngleRot'
        
        # Plot cross-section
        ax.scatter(cs[x_col], cs[y_col], c=cs[rot_col], cmap='hsv', 
                  s=100, edgecolors='black', linewidths=1)
        
        # Add tube labels
        for _, row in cs.iterrows():
            doublet_num = int(row['rlnHelicalTubeID'] % 10)
            if doublet_num == 0:
                doublet_num = 10
            ax.text(row[x_col], row[y_col], str(doublet_num), 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Plot ellipse if available
        if ellipse_params is not None:
            try:
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(xy=(ellipse_params['center_x'], ellipse_params['center_y']),
                                width=2*ellipse_params['a'], height=2*ellipse_params['b'],
                                angle=np.degrees(ellipse_params['theta']),
                                edgecolor='red', fc='None', lw=2, linestyle='--')
                ax.add_patch(ellipse)
            except (KeyError, TypeError):
                # Skip ellipse if parameters are missing
                pass
        
        ax.set_aspect('equal')
        ax.set_xlabel(f'X (Å)')
        ax.set_ylabel(f'Y (Å)')
        ax.set_title(f'Cilium {cilia_id}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[info] Saved multi-cilium plot to {out_png}")
    

def plot_cross_section(cross_section_px, pixel_size_A, ellipse_params=None, output_png=None):
    """Plot cross-section with optional ellipse fit."""
    cs = cross_section_px.reset_index(drop=True).copy()
    Xnm = px_to_nm(cs['rlnCoordinateX'], pixel_size_A)
    Ynm = px_to_nm(cs['rlnCoordinateY'], pixel_size_A)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Xnm, Ynm, c=cs['rlnHelicalTubeID'], cmap='viridis', 
                         s=100, edgecolors='k', zorder=3)

    # Draw connecting lines in circular order
    if 'rlnAngleRot' in cs.columns:
        order = np.argsort(cs['rlnAngleRot'].to_numpy())[::-1]
    else:
        xc, yc = float(np.mean(Xnm)), float(np.mean(Ynm))
        order = np.argsort(np.arctan2(Ynm - yc, Xnm - xc))
    
    xs = Xnm[order].tolist() + [Xnm[order[0]]]
    ys = Ynm[order].tolist() + [Ynm[order[0]]]
    plt.plot(xs, ys, 'k-', alpha=0.5, zorder=1)

    # Annotate tube IDs
    for i in range(len(cs)):
        plt.annotate(str(cs['rlnHelicalTubeID'][i]), (Xnm[i], Ynm[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    # Draw ellipse if provided
    if ellipse_params:
        center = [ellipse_params['X0'], ellipse_params['Y0']]
        axes = [ellipse_params['a'], ellipse_params['b']]
        angle = ellipse_params['phi']
        fitted = ellipse_points(center, axes, angle)
        
        ex_nm = px_to_nm(fitted[0], pixel_size_A)
        ey_nm = px_to_nm(fitted[1], pixel_size_A)
        plt.plot(ex_nm, ey_nm, 'r--', label='Fitted Ellipse', linewidth=2, zorder=2)
        
        distortion = ellipse_params['a'] / ellipse_params['b']
        mx_nm = float(np.mean(px_to_nm(cs['rlnCoordinateX'], pixel_size_A)))
        my_nm = float(np.mean(px_to_nm(cs['rlnCoordinateY'], pixel_size_A)))
        plt.text(mx_nm, my_nm, f"Distortion: {distortion:.2f}",
                fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend()

    plt.xlabel('X (nm)', fontsize=12)
    plt.ylabel('Y (nm)', fontsize=12)
    plt.title('Cross-Section: Tube Positions and Ellipse Fit', fontsize=14)
    plt.colorbar(scatter, label='Tube ID')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    
# ----------------------------- Main pipeline -----------------------------

def sort_doublet_order(df, pixel_size_A, fit_method='ellipse', out_png=None):
    """
    Main processing pipeline. Returns (df_out, cross_sections, ellipse_params_dict).
    Handles multiple cilia by processing each separately.
    """
    validate_dataframe(df, ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ','rlnHelicalTubeID'])
    
    # Check if rlnCiliaGroup exists
    if 'rlnCiliaGroup' not in df.columns:
        print("[WARNING] rlnCiliaGroup column not found. Processing as single cilium.")
        df = df.copy()
        df['rlnCiliaGroup'] = 1
    
    # Get unique cilia groups
    cilia_groups = sorted(df['rlnCiliaGroup'].unique())
    n_cilia = len(cilia_groups)
    
    print(f"[info] Found {n_cilia} cilia group(s)")
    
    # Initialize output containers
    df_out = df.copy()
    all_cross_sections = {}
    all_ellipse_params = {}
    all_mappings = {}  # Store mappings for all cilia
    
    # Process each cilium separately
    for cilia_id in cilia_groups:
        print(f"\n[info] Processing Cilium {cilia_id}")
        
        # Extract data for this cilium
        cilia_mask = df['rlnCiliaGroup'] == cilia_id
        df_cilia = df[cilia_mask].copy()
        
        # Check tube count for this cilium
        n_tubes = df_cilia['rlnHelicalTubeID'].nunique()
        if n_tubes > 9:
            print(f"[WARNING] Cilium {cilia_id} has {n_tubes} tubes (> 9). Skipping sorting for this cilium.")
            all_cross_sections[cilia_id] = None
            all_ellipse_params[cilia_id] = None
            continue
            
        print(f"[info] Cilium {cilia_id}: Processing {n_tubes} tubes using {fit_method} method")
        
        # Process cross-section
        cs = process_cross_section(df_cilia)
        cs_rot = rotate_cross_section(cs)
        cs_rot = enforce_consistent_orientation(cs_rot)
        
        # Calculate rotation angles based on method
        if fit_method.lower() == 'ellipse':
            cs_with_rot, ellipse_params = calculate_rot_angles_ellipse(cs_rot)
        else:  # simple method
            cs_with_rot, ellipse_params = calculate_rot_angles_simple(cs_rot)
        
        # Get tube order
        order_ids = get_tube_order_from_rot(cs_with_rot)
        
        # Store cross-section and ellipse params
        all_cross_sections[cilia_id] = cs_with_rot.copy()
        all_ellipse_params[cilia_id] = ellipse_params
        
        # Renumber
        doublet_mapping = {tube_id: idx + 1 for idx, tube_id in enumerate(order_ids)}
        all_mappings[cilia_id] = doublet_mapping
            
        print(f"[info] Cilium {cilia_id} renumbering: {doublet_mapping}")
    
    # Apply all renumbering at once to avoid conflicts
    if len(all_mappings) > 0:
        print(f"\n[info] Applying tube ID renumbering...")
        
        # First pass: assign temporary negative IDs to avoid conflicts
        temp_mapping = {}
        for cilia_id, doublet_mapping in all_mappings.items():
            for old_id, doublet_num in doublet_mapping.items():
                temp_id = -1000 - old_id  # Use negative temporary IDs
                df_out.loc[df_out['rlnHelicalTubeID'] == old_id, 'rlnHelicalTubeID'] = temp_id
                temp_mapping[temp_id] = (cilia_id, doublet_num)
                
                # Update cross-section dataframe
                all_cross_sections[cilia_id].loc[
                    all_cross_sections[cilia_id]['rlnHelicalTubeID'] == old_id, 
                    'rlnHelicalTubeID'
                ] = temp_id
        
        # Second pass: assign final IDs
        for temp_id, (cilia_id, doublet_num) in temp_mapping.items():
            new_tube_id = (cilia_id - 1) * 10 + doublet_num
            df_out.loc[df_out['rlnHelicalTubeID'] == temp_id, 'rlnHelicalTubeID'] = new_tube_id
            
            # Update cross-section dataframe
            all_cross_sections[cilia_id].loc[
                all_cross_sections[cilia_id]['rlnHelicalTubeID'] == temp_id, 
                'rlnHelicalTubeID'
            ] = new_tube_id
        
        print(f"[info] Renumbering complete!")
    
    # Plot if requested
    if out_png:
        if n_cilia == 1:
            # Single cilium - use existing plot function
            cilia_id = cilia_groups[0]
            if all_cross_sections[cilia_id] is not None:
                plot_cross_section(all_cross_sections[cilia_id], pixel_size_A, 
                                 all_ellipse_params[cilia_id], out_png)
        else:
            # Multiple cilia - plot each separately with cilium ID in filename
            for cilia_id in cilia_groups:
                if all_cross_sections[cilia_id] is not None:
                    # Create output filename with cilium ID
                    base_name = out_png.rsplit('.', 1)[0]
                    ext = out_png.rsplit('.', 1)[1] if '.' in out_png else 'png'
                    cilia_png = f"{base_name}_cilium{cilia_id}.{ext}"
                    
                    plot_cross_section(all_cross_sections[cilia_id], pixel_size_A,
                                     all_ellipse_params[cilia_id], cilia_png)
    
    return df_out


def compute_tube_medians(df):
    """Compute median values for each tube."""
    # Get pixel size (angpix) from the first row (should be same for all)
    angpix = df['rlnImagePixelSize'].iloc[0]
    
    # Create a copy to avoid modifying original dataframe
    df_scaled = df.copy()
    
    # Convert coordinates from pixels to Angstroms
    df_scaled['rlnCoordinateX'] = df['rlnCoordinateX'] * angpix
    df_scaled['rlnCoordinateY'] = df['rlnCoordinateY'] * angpix
    df_scaled['rlnCoordinateZ'] = df['rlnCoordinateZ'] * angpix
    
    # Compute medians on scaled coordinates
    tube_stats = df_scaled.groupby('rlnHelicalTubeID').agg({
        'rlnAngleRot': 'median',
        'rlnAngleTilt': 'median',
        'rlnAnglePsi': 'median',
        'rlnCoordinateX': 'median',
        'rlnCoordinateY': 'median',
        'rlnCoordinateZ': 'median',
        'rlnHelicalTubeID': 'count'  # Number of points per tube
    }).rename(columns={'rlnHelicalTubeID': 'n_points'})
    
    return tube_stats

def angular_difference(angle1, angle2):
    """Compute minimum angular difference considering periodicity."""
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

def classify_cilia_groups(tube_stats, n_cilia=None, tilt_psi_threshold=15, coord_threshold=100):
    """
    Classify tubes into cilia groups based on Tilt/Psi angles and coordinates.
    
    Parameters:
    -----------
    tube_stats : DataFrame with median values per tube
    n_cilia : int or None, expected number of cilia (if None, auto-detect)
    tilt_psi_threshold : float, angular difference threshold for Tilt/Psi (degrees)
    coord_threshold : float, spatial distance threshold for parallel cilia
    """
    n_tubes = len(tube_stats)
    tube_ids = tube_stats.index.values
    
    # Compute pairwise angular differences for Tilt and Psi
    tilt_vals = tube_stats['rlnAngleTilt'].values
    psi_vals = tube_stats['rlnAnglePsi'].values
    
    # Create distance matrix based on Tilt and Psi
    angular_dist = np.zeros((n_tubes, n_tubes))
    for i in range(n_tubes):
        for j in range(i+1, n_tubes):
            tilt_diff = angular_difference(tilt_vals[i], tilt_vals[j])
            psi_diff = angular_difference(psi_vals[i], psi_vals[j])
            # Combined angular distance
            angular_dist[i, j] = np.sqrt(tilt_diff**2 + psi_diff**2)
            angular_dist[j, i] = angular_dist[i, j]
    
    # If n_cilia is specified, use it directly
    if n_cilia is not None:
        print(f"Using specified number of cilia: {n_cilia}")
        
        if n_cilia == 1:
            cilia_labels = np.zeros(n_tubes, dtype=int)
        else:
            # Try angular clustering first
            clustering = AgglomerativeClustering(n_clusters=n_cilia, linkage='average')
            angle_labels = clustering.fit_predict(angular_dist)
            
            # Check if angular clustering gives reasonable results
            unique, counts = np.unique(angle_labels, return_counts=True)
            min_tubes_per_cilium = counts.min()
            
            # If angular clustering gives very unbalanced groups, use spatial clustering
            if min_tubes_per_cilium < 3 or counts.max() / counts.min() > 5:
                print("Angular clustering unbalanced, trying spatial clustering...")
                coords = tube_stats[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
                coords_normalized = (coords - coords.mean(axis=0)) / coords.std(axis=0)
                
                spatial_clustering = AgglomerativeClustering(n_clusters=n_cilia, linkage='average')
                cilia_labels = spatial_clustering.fit_predict(coords_normalized)
            else:
                cilia_labels = angle_labels
    else:
        # Auto-detect mode
        print("Auto-detecting number of cilia...")
        
        # Try to cluster based on angles first
        clustering = AgglomerativeClustering(n_clusters=None, 
                                             distance_threshold=tilt_psi_threshold*np.sqrt(2),
                                             linkage='average')
        
        if n_tubes > 1:
            angle_labels = clustering.fit_predict(angular_dist)
            n_angle_clusters = len(np.unique(angle_labels))
        else:
            angle_labels = np.array([0])
            n_angle_clusters = 1
        
        print(f"Angular clustering found {n_angle_clusters} potential cilia groups")
        
        # If angles suggest parallel cilia, use spatial clustering
        if n_angle_clusters == 1 and n_tubes > 9:
            print("Angles suggest parallel cilia or single cilium. Using spatial clustering...")
            coords = tube_stats[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
            
            # Normalize coordinates for clustering
            coords_normalized = (coords - coords.mean(axis=0)) / coords.std(axis=0)
            
            # Try clustering into 2 groups
            spatial_clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
            spatial_labels = spatial_clustering.fit_predict(coords_normalized)
            
            # Check if spatial clustering makes sense (both groups should have reasonable size)
            unique, counts = np.unique(spatial_labels, return_counts=True)
            if counts.min() >= 4:  # At least 4 tubes per cilium
                print(f"Spatial clustering: Group sizes = {counts}")
                cilia_labels = spatial_labels
            else:
                print("Spatial clustering suggests single cilium")
                cilia_labels = angle_labels
        else:
            cilia_labels = angle_labels
    
    # Create cilia group assignments
    cilia_groups = {}
    for tube_id, label in zip(tube_ids, cilia_labels):
        if label not in cilia_groups:
            cilia_groups[label] = []
        cilia_groups[label].append(tube_id)
    
    return cilia_groups, cilia_labels

def classify_doublet_microtubules(tube_stats, cilia_groups, rot_threshold=10, enforce_9_doublets=False):
    """
    Within each cilium, group tubes by rlnAngleRot to identify doublet microtubules.
    
    Parameters:
    -----------
    tube_stats : DataFrame with median values per tube
    cilia_groups : dict mapping cilium label to list of tube IDs
    rot_threshold : float, angular difference threshold for Rot (degrees)
    enforce_9_doublets : bool, if True, force exactly 9 doublets per cilium
    """
    doublet_assignments = {}
    
    for cilium_id, tube_list in cilia_groups.items():
        print(f"\nProcessing Cilium {cilium_id + 1} with {len(tube_list)} tubes:")
        
        rot_vals = tube_stats.loc[tube_list, 'rlnAngleRot'].values
        
        #print(f"  DEBUG: Tube IDs: {tube_list}")
        #print(f"  DEBUG: Rot Values: {rot_vals.tolist()}")
        
        if len(tube_list) == 1:
            doublet_assignments[tube_list[0]] = (cilium_id + 1, 1)
            continue
        
        # Get Rot values for tubes in this cilium
        rot_vals = tube_stats.loc[tube_list, 'rlnAngleRot'].values
        
        # Compute pairwise Rot differences
        n = len(tube_list)
        rot_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                rot_diff = angular_difference(rot_vals[i], rot_vals[j])
                rot_dist[i, j] = rot_diff
                rot_dist[j, i] = rot_diff
        
        #print(f"  DEBUG: Rot Distance Matrix (first 5x5):")
        #print(rot_dist)
        # Cluster based on Rot angle
        if enforce_9_doublets and len(tube_list) >= 9:
            # Force exactly 9 clusters
            # affinity=precomputed is very important
            clustering = AgglomerativeClustering(n_clusters=9, linkage='average', affinity='precomputed')
            doublet_labels = clustering.fit_predict(rot_dist)
            print(f"  Enforcing 9 doublets (from {len(tube_list)} tubes)")
        else:
            # Auto-detect number of doublets
            clustering = AgglomerativeClustering(n_clusters=None,
                                                distance_threshold=rot_threshold,
                                                linkage='single', affinity='precomputed')
            doublet_labels = clustering.fit_predict(rot_dist)
            n_doublets = len(np.unique(doublet_labels))
            print(f"  Found {n_doublets} potential doublet groups")
            
            # Warn if not 9 doublets
            if n_doublets != 9:
                print(f"  ⚠️  WARNING: Expected 9 doublets but found {n_doublets}!")
                print(f"      Consider adjusting rot_threshold (current: {rot_threshold}°)")
                if n_doublets > 9:
                    print(f"      → Try increasing rot_threshold to merge more tubes")
                else:
                    print(f"      → Try decreasing rot_threshold or check for missing tubes")
        
        # Assign doublet IDs (store as tuple: cilium_group, doublet_number)
        for tube_id, doublet_label in zip(tube_list, doublet_labels):
            doublet_assignments[tube_id] = (cilium_id + 1, doublet_label + 1)
        
        # Print detailed doublet assignments with Rot angles
        doublet_dict = {}
        for tube_id, (cil_id, dbl_id) in doublet_assignments.items():
            if tube_id in tube_list:
                doublet_key = f"C{cil_id}_D{dbl_id}"
                if doublet_key not in doublet_dict:
                    doublet_dict[doublet_key] = []
                rot = tube_stats.loc[tube_id, 'rlnAngleRot']
                doublet_dict[doublet_key].append((tube_id, rot))
        
        print(f"  Doublet assignments (with rlnAngleRot):")
        for doublet_id in sorted(doublet_dict.keys()):
            tubes_info = doublet_dict[doublet_id]
            tubes_str = ', '.join([f"Tube {tid} ({rot:.1f}°)" for tid, rot in tubes_info])
            print(f"    {doublet_id}: {tubes_str}")
    
    return doublet_assignments

def create_grouped_dataframe(df, doublet_assignments):
    """
    Create output dataframe with rlnCiliaGroup and renumbered rlnHelicalTubeID.
    
    Parameters:
    -----------
    df : original DataFrame with all particles
    doublet_assignments : dict mapping original tube IDs to (cilium_group, doublet_id) tuples
    
    Returns:
    --------
    output_df : DataFrame with added rlnCiliaGroup and renumbered rlnHelicalTubeID
    """
    # Create a copy of the original dataframe
    output_df = df.copy()
    
    # Create mapping from original tube ID to new values
    tube_id_mapping = {}
    
    for orig_tube_id, (cilium_group, doublet_id) in doublet_assignments.items():
        # Calculate new tube ID: (CiliaGroup - 1) * 10 + DoubletID
        new_tube_id = (cilium_group - 1) * 10 + doublet_id
        tube_id_mapping[orig_tube_id] = new_tube_id
    
    # Renumber rlnHelicalTubeID first
    output_df['rlnHelicalTubeID'] = output_df['rlnHelicalTubeID'].map(tube_id_mapping)
    
    # Add rlnCiliaGroup column
    output_df['rlnCiliaGroup'] = output_df['rlnHelicalTubeID'].apply(lambda x: (x // 10) + 1)
    
    # Sort by rlnHelicalTubeID (small to large) and then by rlnCoordinateY (small to large)
    output_df = output_df.sort_values(by=['rlnHelicalTubeID', 'rlnCoordinateY'], 
                                       ascending=[True, True]).reset_index(drop=True)

    
    return output_df, tube_id_mapping

def group_cilia_and_doublets(
    df, 
    n_cilia=None, 
    tilt_psi_threshold=10, 
    coord_threshold=900, 
    rot_threshold=8, 
    enforce_9_doublets=False,
    export_json=None  # ADD THIS PARAMETER
):
    """
    Main function to classify ciliary tubes and output results.
    
    Parameters:
    -----------
    df : dataframe of the star file
    n_cilia : int or None, expected number of cilia (if None, auto-detect)
    tilt_psi_threshold : float, angular threshold for Tilt/Psi similarity (degrees)
    coord_threshold : float, spatial distance threshold (Angstroms)
    rot_threshold : float, angular threshold for Rot similarity (degrees, default 8)
    enforce_9_doublets : bool, force exactly 9 doublets per cilium
    export_json : str or None, path to export grouping as JSON (default: None)
    """
    print("="*80)
    print("CILIA TUBE CLASSIFICATION")
    print("="*80)
    
    # Load data
    print(f"Loaded {len(df)} particles from {df['rlnHelicalTubeID'].nunique()} tubes")
    
    # Compute median values per tube
    tube_stats = compute_tube_medians(df)
    print(f"\nTube statistics computed for {len(tube_stats)} tubes")
    print(f"Points per tube range: {tube_stats['n_points'].min()} - {tube_stats['n_points'].max()}")
    
    # Classify into cilia groups
    print("\n" + "-"*80)
    print("STEP 1: Classifying tubes into cilia groups")
    print("-"*80)
    cilia_groups, cilia_labels = classify_cilia_groups(
        tube_stats, 
        n_cilia,
        tilt_psi_threshold, 
        coord_threshold
    )
    
    print(f"\nFound {len(cilia_groups)} cilia group(s):")
    for cilium_id, tubes in cilia_groups.items():
        print(f"  Cilium {cilium_id + 1}: {len(tubes)} tubes - {tubes}")
    
    # Classify doublet microtubules within each cilium
    print("\n" + "-"*80)
    print("STEP 2: Classifying doublet microtubules within each cilium")
    print("-"*80)
    doublet_assignments = classify_doublet_microtubules(
        tube_stats, 
        cilia_groups, 
        rot_threshold,
        enforce_9_doublets
    )
        
    # Print final results
    print("\n" + "="*80)
    print("FINAL CLASSIFICATION RESULTS")
    print("="*80)
    
    for cilium_id, tubes in cilia_groups.items():
        print(f"\nCilium {cilium_id + 1}:")
        doublets_in_cilium = {}
        for tube_id in tubes:
            cil_grp, dbl_id = doublet_assignments[tube_id]
            doublet_key = f"C{cil_grp}_D{dbl_id}"
            if doublet_key not in doublets_in_cilium:
                doublets_in_cilium[doublet_key] = []
            doublets_in_cilium[doublet_key].append(tube_id)
        
        for doublet_key, tube_list in sorted(doublets_in_cilium.items()):
            print(f"  {doublet_key}: Tubes {tube_list}")
        
    # ========== ADD THIS BLOCK: Export JSON if requested ==========
    if export_json:        
        export_automatic_grouping(
            doublet_assignments=doublet_assignments,
            output_json=export_json,
            method='automatic'
        )
    # ==============================================================
    
    # Create output dataframe with modified columns
    print("\n" + "-"*80)
    print("STEP 3: Creating output dataframe")
    print("-"*80)
    output_df, tube_id_mapping = create_grouped_dataframe(df, doublet_assignments)
    
    print(f"\nOutput dataframe created with {len(output_df)} particles")
    print(f"New columns added: rlnCiliaGroup")
    print(f"rlnHelicalTubeID renumbered according to: (CiliaGroup - 1) * 10 + DoubletID")
    
    # Display tube ID mapping
    print("\nTube ID mapping:")
    for orig_id in sorted(tube_id_mapping.keys()):
        new_id = tube_id_mapping[orig_id]
        cil_grp, dbl_num = doublet_assignments[orig_id]
        print(f"  Original Tube {orig_id} → New Tube {new_id} (Cilium {cil_grp}, Doublet {dbl_num})")
        
    return output_df

def export_automatic_grouping(
    doublet_assignments: Dict[int, Tuple[int, int]],
    output_json: str,
    method: str = 'automatic'
) -> None:
    """
    Export automatic grouping results to JSON file for record keeping.
    ... [docstring content remains the same] ...
    """
    
    # Group tubes by cilium
    cilia_groups = defaultdict(lambda: {'tubes': [], 'doublet_order': []})
    
    # NOTE: The keys (tube_id) and values (cilium_id, doublet_num) 
    # are likely the source of the numpy.int64.
    for tube_id in sorted(doublet_assignments.keys()):
        cilium_id, doublet_num = doublet_assignments[tube_id]
        cilia_groups[cilium_id]['tubes'].append(tube_id)
        cilia_groups[cilium_id]['doublet_order'].append(doublet_num)
    
    # Create JSON structure
    output_data = {}
    
    # Add metadata
    output_data['_metadata'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'method': method,
        'n_cilia': len(cilia_groups),
        'n_tubes': len(doublet_assignments),
        'note': 'Automatically generated grouping. Can be edited and reused for manual grouping.'
    }
    
    # Add cilium data
    for cilium_id in sorted(cilia_groups.keys()):
        cilium_key = f"cilium_{cilium_id}"
        output_data[cilium_key] = {
            'tubes': cilia_groups[cilium_id]['tubes'],
            'doublet_order': cilia_groups[cilium_id]['doublet_order']
        }
    
    # 2. FIX: Write to file using the custom encoder
    with open(output_json, 'w') as f:
        # Pass the custom encoder class to the 'cls' argument
        json.dump(output_data, f, indent=2, cls=NumpyJSONEncoder)
    
    print(f"\n✓ Automatic grouping exported to: {output_json}")
    print(f"  Method: {method}")
    print(f"  Cilia: {len(cilia_groups)}")
    print(f"  Tubes: {len(doublet_assignments)}")
    print(f"\nThis file can be:")
    print(f"  1. Used as a record of the automatic grouping")
    print(f"  2. Edited manually and reused with --manual flag")
    print(f"  3. Shared with collaborators for reproducibility")

def load_manual_groups_json(json_file: str) -> Dict[int, Tuple[int, int]]:
    """
    Load manual grouping from JSON file.
    
    Supports both 1-to-1 and many-to-1 mapping (multiple tubes can share same doublet).
    
    JSON Format (1-to-1):
    {
      "cilium_1": {
        "tubes": [1, 5, 9, 13, 17, 21, 25, 29, 33],
        "doublet_order": [1, 2, 3, 4, 5, 6, 7, 8, 9]
      }
    }
    
    JSON Format (many-to-1, e.g., 10 tubes in 9 doublets):
    {
      "cilium_1": {
        "tubes": [1, 5, 9, 13, 17, 21, 25, 29, 33, 37],
        "doublet_order": [1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
      }
    }
    
    Parameters:
    -----------
    json_file : str
        Path to JSON file with manual grouping specification
    
    Returns:
    --------
    dict : mapping {original_tube_id: (cilium_group, doublet_number)}
    
    Raises:
    -------
    FileNotFoundError : if JSON file doesn't exist
    ValueError : if JSON format is invalid
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manual grouping file not found: {json_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file}: {e}")
    
    assignments = {}
    
    for cilium_name, info in data.items():
        # Skip instruction/comment keys
        if cilium_name.startswith('_'):
            continue
            
        # Extract cilium ID from key (e.g., "cilium_1" -> 1)
        if not cilium_name.startswith('cilium_'):
            raise ValueError(f"Invalid cilium key: {cilium_name}. Must start with 'cilium_'")
        
        try:
            cilium_id = int(cilium_name.split('_')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid cilium key format: {cilium_name}. Expected format: 'cilium_N'")
        
        # Validate required fields
        if 'tubes' not in info:
            raise ValueError(f"Missing 'tubes' field for {cilium_name}")
        
        tubes = info['tubes']
        
        # If doublet_order not specified, use sequential order (1-to-1)
        if 'doublet_order' in info:
            order = info['doublet_order']
            
            # Validate lengths match (required for explicit doublet_order)
            if len(tubes) != len(order):
                raise ValueError(
                    f"{cilium_name}: Number of tubes ({len(tubes)}) must match "
                    f"doublet_order length ({len(order)})"
                )
        else:
            order = list(range(1, len(tubes) + 1))
            print(f"  Note: No doublet_order specified for {cilium_name}, using sequential 1-to-1 mapping")
        
        # Validate doublet numbers are in valid range
        if not all(1 <= d <= 10 for d in order):
            raise ValueError(
                f"{cilium_name}: Doublet numbers must be between 1 and 10, got {order}"
            )
        
        # Check for duplicate tube IDs across cilia
        for tube_id in tubes:
            if tube_id in assignments:
                raise ValueError(
                    f"Tube {tube_id} is assigned to multiple cilia "
                    f"(at least {cilium_name} and another)"
                )
        
        # Detect many-to-1 mapping
        unique_doublets = len(set(order))
        if unique_doublets < len(tubes):
            print(f"  Note: {cilium_name} has many-to-1 mapping: "
                  f"{len(tubes)} tubes mapped to {unique_doublets} doublets")
            
            # Show which doublets have multiple tubes
            from collections import Counter
            doublet_counts = Counter(order)
            for doublet_num, count in sorted(doublet_counts.items()):
                if count > 1:
                    tube_ids = [tubes[i] for i, d in enumerate(order) if d == doublet_num]
                    print(f"    Doublet {doublet_num}: {count} tubes {tube_ids}")
        
        # Create assignments
        for tube_id, doublet_num in zip(tubes, order):
            assignments[tube_id] = (cilium_id, doublet_num)
    
    return assignments


def validate_manual_assignments(df: pd.DataFrame, assignments: Dict[int, Tuple[int, int]]) -> None:
    """
    Validate that manual assignments are compatible with the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with particles
    assignments : dict
        Manual tube assignments
    
    Raises:
    -------
    ValueError : if validation fails
    """
    # Get actual tube IDs from dataframe
    actual_tubes = set(df['rlnHelicalTubeID'].unique())
    assigned_tubes = set(assignments.keys())
    
    # Check for tubes in JSON that don't exist in data
    extra_tubes = assigned_tubes - actual_tubes
    if extra_tubes:
        raise ValueError(
            f"Manual grouping contains tube IDs not found in data: {sorted(extra_tubes)}\n"
            f"Available tubes: {sorted(actual_tubes)}"
        )
    
    # Check for tubes in data not assigned in JSON
    missing_tubes = actual_tubes - assigned_tubes
    if missing_tubes:
        raise ValueError(
            f"Manual grouping missing tube IDs present in data: {sorted(missing_tubes)}\n"
            f"All tubes must be assigned to a cilium group."
        )
    
    print(f"  ✓ Validated {len(assigned_tubes)} tube assignments")


def apply_manual_groups(df: pd.DataFrame, assignments: Dict[int, Tuple[int, int]]) -> pd.DataFrame:
    """
    Apply manual tube grouping to dataframe.
    
    Tubes assigned to the same (cilium_group, doublet_number) will be assigned 
    the SAME new rlnHelicalTubeID, thereby combining them into a single group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with particles
    assignments : dict
        Mapping {original_tube_id: (cilium_group, doublet_number)}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with updated rlnHelicalTubeID and rlnCiliaGroup columns
    """
    
    output_df = df.copy()
    
    # Create new tube ID mapping
    # This dictionary will map original tube IDs to their NEW (shared) HelicalTubeID.
    tube_id_mapping = {}
    
    print("\nℹ️ Applying Manual Grouping: Many-to-one mappings will result in shared IDs.")
    
    for tube_id, (cilium_group, doublet_id) in assignments.items():
        # --- MODIFIED LOGIC START ---
        
        # Calculate the new rlnHelicalTubeID. 
        # This ID is based ONLY on the cilium and doublet number, 
        # meaning all tubes assigned here will get the same ID.
        # ID calculation: (CiliumGroup - 1) * 10 + DoubletNumber
        # This formula ensures IDs are unique across different (Cilium, Doublet) pairs.
        new_tube_id = (cilium_group - 1) * 10 + doublet_id
        
        tube_id_mapping[tube_id] = new_tube_id
        
        # --- MODIFIED LOGIC END ---

    # Apply mapping
    # This will replace the original rlnHelicalTubeID (particle-level) with the 
    # new group ID (doublet-level).
    output_df['rlnHelicalTubeID'] = output_df['rlnHelicalTubeID'].map(tube_id_mapping)
    
    # Add rlnCiliaGroup column
    # The new CiliaGroup is derived directly from the new HelicalTubeID:
    # CiliaGroup = floor((HelicalTubeID - 1) / 10) + 1
    output_df['rlnCiliaGroup'] = output_df['rlnHelicalTubeID'].apply(
        lambda x: ((x - 1) // 10) + 1
    )
    
    # Sort by tube ID and then by Y coordinate for consistency
    output_df = output_df.sort_values(
        by=['rlnHelicalTubeID', 'rlnCoordinateY'],  
        ascending=[True, True]
    ).reset_index(drop=True)
    
    print(f"✓ Manual grouping applied to {len(output_df)} particles.")
    
    return output_df

def print_manual_grouping_summary(assignments: Dict[int, Tuple[int, int]]) -> None:
    """
    Print a formatted summary of manual tube grouping, reflecting the 
    new grouping logic where tubes assigned to the same doublet receive 
    the SAME output ID (no offset).
    
    Parameters:
    -----------
    assignments : dict
        Mapping {original_tube_id: (cilium_group, doublet_number)}
    """
    
    print("\n" + "="*80)
    print("MANUAL GROUPING SUMMARY")
    print("="*80)
    
    # Group by cilium
    cilia = defaultdict(list)
    # Also track doublet usage for many-to-1 detection
    doublet_usage = defaultdict(list)
    
    for tube_id, (cilium_id, doublet_num) in assignments.items():
        # Store for printing
        cilia[cilium_id].append((tube_id, doublet_num))
        # Store for checking many-to-one
        doublet_usage[(cilium_id, doublet_num)].append(tube_id)
    
    # Print each cilium
    for cilium_id in sorted(cilia.keys()):
        # Sort by doublet, then tube
        tubes = sorted(cilia[cilium_id], key=lambda x: (x[1], x[0])) 
        print(f"\nCilium {cilium_id}:")
        
        # Group by doublet for clearer display
        doublet_groups = defaultdict(list)
        for tube_id, doublet_num in tubes:
            doublet_groups[doublet_num].append(tube_id)
        
        for doublet_num in sorted(doublet_groups.keys()):
            tube_ids = doublet_groups[doublet_num]
            
            # Calculate the SHARED new ID once for this doublet group
            new_tube_id = (cilium_id - 1) * 10 + doublet_num
            
            if len(tube_ids) == 1:
                # 1-to-1 mapping
                tube_id = tube_ids[0]
                print(f"  Doublet {doublet_num:2d}: Tube {tube_id:3d} → New ID {new_tube_id:3d}")
            else:
                # Many-to-1 mapping
                # Tubes are now COMBINED under a single ID
                
                # Format tube IDs for a concise list
                tubes_list_str = ", ".join(map(str, tube_ids))
                
                print(f"  Doublet {doublet_num:2d}: {len(tube_ids)} tubes ({tubes_list_str})")
                print(f"    ↳ ALL tubes assigned New ID **{new_tube_id:3d}** (Combined Group)")

    print("\n" + "="*80)
    
    # Print warning/note about the COMBINED grouping
    has_many_to_one = any(len(tubes) > 1 for tubes in doublet_usage.values())
    if has_many_to_one:
        print("\nℹ️ Note: Many-to-1 mappings detected (multiple tubes per doublet).")
        print("  All tubes in a many-to-one mapping are assigned the **same** New ID.")
        print("  This means the particles from these tubes will be combined into one group.")
        print("="*80)


def manual_group_and_sort(
    df: pd.DataFrame,
    manual_json: str,
    angpix: float,
    fit_method: str = 'ellipse',
    out_png: Optional[str] = None
) -> pd.DataFrame:
    """
    Main function for manual grouping and sorting of ciliary tubes.
    
    This function:
    1. Loads manual grouping from JSON file
    2. Validates assignments against input data
    3. Applies manual grouping (renumbers tube IDs)
    4. Optionally sorts doublets within each cilium using ellipse fitting
    5. Optionally generates visualization plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with particle coordinates and tube IDs
    manual_json : str
        Path to JSON file with manual grouping specification
    angpix : float
        Pixel size in Angstroms
    fit_method : str, optional
        Method for sorting doublets: 'ellipse' or 'simple' (default: 'ellipse')
        Only used if you want to re-sort within the manually grouped cilia
    out_png : str, optional
        Output PNG file for visualization (default: None)
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with updated tube IDs and cilium groups
    
    Example JSON format:
    -------------------
    {
      "cilium_1": {
        "tubes": [1, 5, 9, 13, 17, 21, 25, 29, 33],
        "doublet_order": [1, 2, 3, 4, 5, 6, 7, 8, 9]
      },
      "cilium_2": {
        "tubes": [2, 6, 10, 14, 18, 22, 26, 30, 34]
      }
    }
    
    Usage:
    ------
    df_sorted = manual_group_and_sort(
        df=input_df,
        manual_json='manual_groups.json',
        angpix=14.0,
        fit_method='ellipse',
        out_png='output.png'
    )
    """
    print("\n" + "="*80)
    print("MANUAL TUBE GROUPING AND SORTING")
    print("="*80)
    print(f"  JSON file: {manual_json}")
    print(f"  Pixel size: {angpix} Å")
    print(f"  Fit method: {fit_method}")
    
    # Step 1: Load manual grouping
    print("\n[1/4] Loading manual grouping from JSON...")
    try:
        assignments = load_manual_groups_json(manual_json)
    except Exception as e:
        raise ValueError(f"Failed to load manual grouping: {e}")
    
    n_cilia = len(set(cil_id for cil_id, _ in assignments.values()))
    print(f"  ✓ Loaded grouping for {n_cilia} cilia, {len(assignments)} tubes")
    
    # Step 2: Validate assignments
    print("\n[2/4] Validating manual assignments...")
    validate_manual_assignments(df, assignments)
    
    # Step 3: Apply manual grouping
    print("\n[3/4] Applying manual grouping...")
    df_grouped = apply_manual_groups(df, assignments)
    print(f"  ✓ Grouped {len(df_grouped)} particles into {n_cilia} cilia")
    
    # Print summary
    print_manual_grouping_summary(assignments)
    
    # Step 4: Optional - Sort doublets within each cilium using geometric methods
    # Note: This step can be skipped if doublet_order is already specified correctly
    print("\n[4/4] Sorting doublets (optional geometric refinement)...")
    print("  Note: Since doublet order is manually specified, geometric sorting is skipped.")
    print("  If you want to re-sort using ellipse fitting, use sort_doublet_order() separately.")
    
    df_output = df_grouped
    
    # Generate plot if requested
    if out_png:
        print(f"\n[Plotting] Generating visualization...")
        # You can add plotting here if needed
        # For now, we skip plotting in manual mode since the order is user-defined
        print(f"  Note: Plotting skipped in manual mode. Use automatic mode for plots.")
    
    print("\n" + "="*80)
    print("MANUAL GROUPING COMPLETE")
    print("="*80)
    print(f"  Output: {len(df_output)} particles")
    print(f"  Cilia: {df_output['rlnCiliaGroup'].nunique()}")
    print(f"  Tubes: {df_output['rlnHelicalTubeID'].nunique()}")
    print("="*80 + "\n")
    
    return df_output


def export_grouping_template(df: pd.DataFrame, output_json: str) -> None:
    """
    Export a template JSON file for manual grouping based on current tube IDs.
    
    This is helpful to generate a starting point that users can edit.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with tube IDs
    output_json : str
        Path to output JSON template file
    
    Example:
    --------
    # Generate template for manual editing
    export_grouping_template(df, 'manual_groups_template.json')
    # Then edit manual_groups_template.json to assign tubes to cilia
    """
    tube_ids = sorted(df['rlnHelicalTubeID'].unique())
    
    # Create template with all tubes in cilium_1 by default
    template = {
        "cilium_1": {
            "tubes": tube_ids,
            "doublet_order": list(range(1, len(tube_ids) + 1))
        }
    }
    
    # Add comment with instructions
    template["_instructions"] = {
        "note": "Edit this file to group tubes into cilia",
        "format": {
            "cilium_N": {
                "tubes": "List of original tube IDs for this cilium",
                "doublet_order": "Optional: Doublet numbers 1-9 for each tube (omit for sequential)"
            }
        },
        "example": {
            "cilium_1": {
                "tubes": [1, 5, 9, 13, 17, 21, 25, 29, 33],
                "doublet_order": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "cilium_2": {
                "tubes": [2, 6, 10, 14, 18, 22, 26, 30, 34],
                "doublet_order": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✓ Template exported to: {output_json}")
    print(f"  Found {len(tube_ids)} tubes: {tube_ids}")
    print(f"\nEdit this file to:")
    print(f"  1. Split tubes into multiple cilia (cilium_1, cilium_2, etc.)")
    print(f"  2. Specify doublet order (1-9) for each cilium")
    print(f"  3. Remove the '_instructions' section before using")




def group_and_sort(
    df: pd.DataFrame,
    angpix: float,
    n_cilia=None,
    tilt_psi_threshold: float=10,
    coord_threshold: float=900,
    rot_threshold: float=8,
    enforce_9_doublets: bool=False,
    fit_method: str='ellipse',
    out_png: str=None,
    export_json: str=None  # ADD THIS PARAMETER
) -> pd.DataFrame:
    """
    Main entry point for grouping and sorting.
    
    Parameters:
    -----------
    export_json : str, optional
        Path to export automatic grouping as JSON for record keeping
    """

    # Run group cilia
    grouped_df = group_cilia_and_doublets(
        df=df, 
        n_cilia=n_cilia,
        tilt_psi_threshold=tilt_psi_threshold,
        coord_threshold=coord_threshold,
        rot_threshold=rot_threshold,
        enforce_9_doublets=enforce_9_doublets,
        export_json=export_json  # PASS THE PARAMETER
    )
    
    # Run sort
    sorted_df = sort_doublet_order(
        df=grouped_df,
        pixel_size_A=angpix,
        fit_method=fit_method,
        out_png=out_png
    )

    if out_png:
        print(f"[info] Saved plot → {out_png}")

    return sorted_df