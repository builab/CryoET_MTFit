#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geom/Sort for ReLAX — cross-section extraction, ellipse fit, ordering, and plotting (nm-scaled).

- Works with STAR files that have either _rln* or rln* column names.
- Reads pixel size from _rlnImagePixelSize (Å/px). You can override with --angpix.
- Selects a cross-section via the shortest filament’s midpoint & a plane normal from local vectors.
- Rotates into the Z plane (Psi/Tilt median), enforces consistent orientation (clockwise, right-facing).
- Fits ellipse, assigns rlnAngleRot, optional virtual points for gaps, optional renumbering.
- Plots in **nm** (correct scaling: nm = px * Å/px / 10).

Usage:
  python sort.py \
    --input  example/CCDC147C_004_particles.star \
    --output example/CCDC147C_004_particles_sorted.star \
    --output-png example/cs_ellipse.png \
    --fit-method ellipse \
    --propagate \
    --renumber
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import starfile

# ----------------------------- utils / normalization -----------------------------

def normalize_angle(angle):
    """Normalize angle to range -180..180 (Relion-style)."""
    return (angle + 180) % 360 - 180

def strip_leading_underscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map _rlnX → rlnX (without underscore) for internal consistency; keep originals too."""
    mapping = {c: (c[1:] if c.startswith("_") else c) for c in df.columns}
    df2 = df.copy()
    df2.columns = [mapping[c] for c in df.columns]
    return df2

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def infer_pixel_size_A(df_in: pd.DataFrame, override: float = None) -> float:
    """Å/px from STAR; prefer _rlnImagePixelSize / rlnImagePixelSize; fallback to override."""
    for name in ["_rlnImagePixelSize", "rlnImagePixelSize", "ImagePixelSize", "Angpix"]:
        if name in df_in.columns and pd.notna(df_in[name]).any():
            return float(pd.to_numeric(df_in[name], errors="coerce").dropna().iloc[0])
    if override is not None:
        return float(override)
    raise ValueError("Pixel size not found. Provide --angpix or include _rlnImagePixelSize in STAR.")

# ----------------------------- geometry helpers (your logic) -----------------------------

def cumulative_distance(points):
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0], np.cumsum(d)))

def calculate_perpendicular_distance(point, plane_normal, reference_point):
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)

def find_cross_section_points(data, plane_normal, reference_point):
    """
    For each filament, pick the point closest to the plane; also return the max min-distance.
    Expects columns rlnHelicalTubeID, rlnCoordinateX/Y/Z (IN PIXELS).
    """
    cross_section = []
    max_distance = 0
    grouped = data.groupby('rlnHelicalTubeID', sort=False)
    for filament_id, group in grouped:
        pts = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(float)
        dists = np.array([calculate_perpendicular_distance(p, plane_normal, reference_point) for p in pts])
        min_dist = float(np.min(dists))
        max_distance = max(max_distance, min_dist)
        closest = group.iloc[int(np.argmin(dists))].copy()
        closest['distance_to_plane'] = min_dist
        cross_section.append(closest)
    return pd.DataFrame(cross_section), max_distance

def find_shortest_filament(data):
    shortest_len, shortest_mid, shortest_id = float('inf'), None, None
    for filament_id, g in data.groupby('rlnHelicalTubeID', sort=False):
        pts = g[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        length = float(np.linalg.norm(mx - mn))
        if length < shortest_len:
            shortest_len, shortest_mid, shortest_id = length, (mn + mx) / 2.0, filament_id
    return shortest_id, shortest_mid

def calculate_normal_vector(filament_points, window_size=3):
    """
    Local average of segment vectors around the midpoint (pixels). Ensures z-positive.
    """
    n = filament_points.shape[0]
    mid = n // 2
    s = max(mid - window_size, 0)
    e = min(mid + window_size, n - 1)
    vecs = []
    for i in range(s, e):
        vecs.append(filament_points[i + 1] - filament_points[i])
    vecs = np.asarray(vecs, float)
    avg = np.mean(vecs, axis=0)
    nv = avg / np.linalg.norm(avg)
    if nv[2] < 0:  # enforce pointing +z
        nv = -nv
    return nv

def process_cross_section(data):
    """
    Find cross-section based on the midpoint of the SHORTEST filament.
    """
    shortest_id, midpoint = find_shortest_filament(data)
    filament_pts = data.loc[data['rlnHelicalTubeID'] == shortest_id, ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].to_numpy(float)
    nvec = calculate_normal_vector(filament_pts)
    return find_cross_section_points(data, nvec, midpoint)

def rotate_cross_section(cross_section):
    """
    Rotate by median Psi (about Z) then median Tilt (about Y) to bring the section into the Z plane.
    Mutates a copy of cross_section (pixels).
    """
    rotated = cross_section.copy()
    psi = 90 - float(np.nanmedian(rotated['rlnAnglePsi']))
    tilt = float(np.nanmedian(rotated['rlnAngleTilt']))

    psi_rad = np.radians(psi)
    tilt_rad = np.radians(tilt)
    Rz = np.array([
        [np.cos(-psi_rad), -np.sin(-psi_rad), 0],
        [np.sin(-psi_rad),  np.cos(-psi_rad), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [1, 0, 0],
        [0, np.cos(-tilt_rad), -np.sin(-tilt_rad)],
        [0, np.sin(-tilt_rad),  np.cos(-tilt_rad)]
    ])

    for idx, row in rotated.iterrows():
        v = np.array([row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']], float)
        v = Rz @ v
        v = Ry @ v
        rotated.at[idx, 'rlnCoordinateX'] = v[0]
        rotated.at[idx, 'rlnCoordinateY'] = v[1]
        rotated.at[idx, 'rlnCoordinateZ'] = v[2]

    rotated[['rlnAngleTilt','rlnAnglePsi']] = 0
    return rotated

def polygon_signed_area(points):
    """Signed area (for orientation check). Negative → clockwise."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    return area

def enforce_consistent_cross_section_orientation(df):
    """
    Ensure rotated cross-section is clockwise and right-facing (pixels).
    """
    points = df[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    area = polygon_signed_area(points)
    out = df.copy()
    if area > 0:  # counterclockwise → flip Y
        out['rlnCoordinateY'] = -out['rlnCoordinateY']
    if out['rlnCoordinateX'].mean() < 0:  # mostly on left → flip X
        out['rlnCoordinateX'] = -out['rlnCoordinateX']
    return out

# ----------------------------- ellipse math (your logic) -----------------------------

def fit_ellipse(x, y, axis_handle=None):
    """
    Least-squares ellipse fit (Python port of the approach used in your script).
    Returns dict with {a,b,phi,X0,Y0,X0_in,Y0_in,long_axis,short_axis,status}.
    """
    x = np.asarray(x).ravel().astype(float)
    y = np.asarray(y).ravel().astype(float)

    if len(x) < 5:
        print("WARNING: Not enough points to fit an ellipse!")

    # de-bias for stability
    mx, my = np.mean(x), np.mean(y)
    x -= mx
    y -= my

    X = np.column_stack([x**2, x*y, y**2, x, y])
    a, _, _, _ = lstsq(X, -np.ones_like(x), lapack_driver='gelsy')
    A, B, C, D, E = a

    disc = B**2 - 4*A*C
    if disc >= 0:
        warnings.warn("Invalid ellipse (discriminant >= 0).")
        return {'a':None,'b':None,'phi':None,'X0':None,'Y0':None,'X0_in':None,'Y0_in':None,
                'long_axis':None,'short_axis':None,'status':'Invalid'}

    orientation = 0.5 * np.arctan2(B, (A - C))
    cphi, sphi = np.cos(orientation), np.sin(orientation)

    A_r = A * cphi**2 - B * cphi * sphi + C * sphi**2
    C_r = A * sphi**2 + B * cphi * sphi + C * cphi**2
    D_r = D * cphi - E * sphi
    E_r = D * sphi + E * cphi

    if A_r < 0 or C_r < 0:
        A_r, C_r, D_r, E_r = -A_r, -C_r, -D_r, -E_r

    X0 = (mx - D_r / (2 * A_r))
    Y0 = (my - E_r / (2 * C_r))
    F = 1 + (D_r**2) / (4 * A_r) + (E_r**2) / (4 * C_r)
    a_len = np.sqrt(F / A_r)
    b_len = np.sqrt(F / C_r)

    R = np.array([[cphi, sphi], [-sphi, cphi]])
    X0_in, Y0_in = R @ np.array([X0, Y0])
    long_axis = 2 * max(a_len, b_len)
    short_axis = 2 * min(a_len, b_len)

    return {'a':a_len,'b':b_len,'phi':orientation,'X0':X0,'Y0':Y0,
            'X0_in':X0_in,'Y0_in':Y0_in,'long_axis':long_axis,'short_axis':short_axis,'status':'Success'}

def ellipse_points(center, axes, angle, num_points=200):
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.array([axes[0]*np.cos(t), axes[1]*np.sin(t)])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    e_rot = R @ ellipse
    e_rot[0] += center[0]
    e_rot[1] += center[1]
    return e_rot

def angle_along_ellipse(center, axes, angle, points):
    """Return parameter angle t (radians) along ellipse for given points."""
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    a, b = axes
    out = []
    for (px, py) in points:
        xt = px - center[0]
        yt = py - center[1]
        xr = xt * cos_a - yt * sin_a
        yr = xt * sin_a + yt * cos_a
        t = np.arctan2(yr / b, xr / a)
        out.append(t + angle)
    return np.array(out)

# ----------------------------- analysis / ordering -----------------------------

def detect_multiple_missing_points(rot_angles, gap_threshold=50):
    """
    Find large angular gaps in rot angles → indicate missing filaments.
    Returns list of (gap_start_angle, gap_end_angle) in degrees.
    """
    vals = np.sort(np.asarray(rot_angles, float))
    gaps, pairs = [], []
    for i in range(len(vals)):
        nxt = vals[(i + 1) % len(vals)]
        delta = (nxt - vals[i]) % 360
        gaps.append(delta)
        if delta > gap_threshold:
            pairs.append((vals[i], nxt))
    return pairs

def calculate_ellipse_point(ellipse_params, theta_deg):
    """Point on ellipse at theta (deg) using fitted params (pixels)."""
    theta = np.radians(theta_deg)
    phi = ellipse_params['phi']
    x0, y0 = ellipse_params['X0'], ellipse_params['Y0']
    a,  b  = ellipse_params['a'],  ellipse_params['b']
    x = x0 + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi)
    y = y0 + a * np.cos(theta) * np.sin(phi) + b * np.sin(theta) * np.cos(phi)
    return x, y

def calculate_rot_angles_ellipse(rotated_cross_section):
    """
    Fit ellipse, compute rlnAngleRot for each cross-section point.
    Also create a version with virtual points where big gaps are detected.
    """
    df = rotated_cross_section.copy()
    pts = df[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    x, y = pts[:,0], pts[:,1]

    ellipse = fit_ellipse(x, y)
    if ellipse['a'] is None or ellipse['b'] is None:
        raise RuntimeError("Ellipse fit failed.")

    center = [ellipse['X0'], ellipse['Y0']]
    axes   = [ellipse['a'],  ellipse['b']]
    phi    = ellipse['phi']

    # base rot angles
    ang = angle_along_ellipse(center, axes, phi, pts)
    ang = np.degrees(ang) - 270
    ang = np.vectorize(normalize_angle)(ang)
    df['rlnAngleRot'] = ang

    # Handle missing filaments (gaps)
    df_virtual = df.copy()
    gaps = detect_multiple_missing_points(df['rlnAngleRot'])
    for k, (gap_s, gap_e) in enumerate(gaps):
        # virtual at 40° from start
        virt_theta = normalize_angle(gap_s - 40)
        vx, vy = calculate_ellipse_point(ellipse, virt_theta)
        dummy = df.iloc[0].copy()
        dummy['rlnCoordinateX'] = vx
        dummy['rlnCoordinateY'] = vy
        dummy['rlnCoordinateZ'] = df['rlnCoordinateZ'].mean()
        dummy['rlnHelicalTubeID'] = 999 + k  # ensure unique
        dummy['rlnAngleRot'] = virt_theta
        df_virtual = pd.concat([df_virtual, pd.DataFrame([dummy])], ignore_index=True)

        # (Optional) update neighbors' angles — keeping original behavior light here.

    return df, df_virtual, ellipse

def get_filament_order_from_rot(rotated_cross_section):
    """Order tubes by descending rlnAngleRot."""
    s = rotated_cross_section.sort_values('rlnAngleRot', ascending=False)
    return s['rlnHelicalTubeID'].tolist()

def renumber_filament_ids(df_all, sorted_ids, cs_df):
    """
    Map old HelicalTubeID → new 1..N order based on sorted_ids (list of original IDs).
    """
    mapping = {orig: new_i + 1 for new_i, orig in enumerate(sorted_ids, start=0)}
    df_out = df_all.copy()
    cs_out = cs_df.copy()
    df_out['rlnHelicalTubeID'] = df_out['rlnHelicalTubeID'].map(mapping)
    cs_out['rlnHelicalTubeID'] = cs_out['rlnHelicalTubeID'].map(mapping)
    return df_out, cs_out, mapping

def propagate_rot_to_entire_cilia(cs_with_rot, df_all):
    """Broadcast rlnAngleRot from cross-section to whole dataset by HelicalTubeID."""
    rot_lookup = cs_with_rot.set_index('rlnHelicalTubeID')['rlnAngleRot'].to_dict()
    out = df_all.copy()
    out['rlnAngleRot'] = out['rlnHelicalTubeID'].map(rot_lookup)
    return out

# ----------------------------- plotting (SAME STYLE, nm scaling) -----------------------------

def _px_to_nm(arr_px, pixel_size_A: float):
    return np.asarray(arr_px, float) * (float(pixel_size_A) / 10.0)

def plot_cs(cross_section_px: pd.DataFrame, pixel_size_A: float, output_png: str = None):
    """
    Base cross-section plot (your original style) — now internally converts px → nm.
    """
    cs = cross_section_px.reset_index(drop=True).copy()
    Xnm = _px_to_nm(cs['rlnCoordinateX'], pixel_size_A)
    Ynm = _px_to_nm(cs['rlnCoordinateY'], pixel_size_A)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(Xnm, Ynm, c=cs['rlnHelicalTubeID'], cmap='viridis', s=100, edgecolors='k')

    # circular line  FIX: connect in circular order
    if 'rlnAngleRot' in cs.columns and cs['rlnAngleRot'].notna().all():
        # Match your renumber/order logic: descending rlnAngleRot
        order = np.argsort(cs['rlnAngleRot'].to_numpy())[::-1]
    else:
        # Fallback: polar angle about centroid (only used if rlnAngleRot absent)
        xc, yc = float(np.mean(Xnm)), float(np.mean(Ynm))
        order = np.argsort(np.arctan2(Ynm - yc, Xnm - xc))

    # Xnm/Ynm are NumPy arrays → use plain indexing, not .iloc
    xs = Xnm[order].tolist()
    ys = Ynm[order].tolist()
    xs.append(xs[0]); ys.append(ys[0])
    plt.plot(xs, ys, 'k-', alpha=0.5, zorder=1)



    # annotate
    for i in range(len(cs)):
        plt.annotate(str(cs['rlnHelicalTubeID'][i]), (Xnm[i], Ynm[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.title('Filament Plot: X vs Y with Filament IDs and Circular Connecting Lines')
    plt.colorbar(scatter, label='Filament ID')
    plt.grid(True)

    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plt

def plot_ellipse_cs(cross_section_px: pd.DataFrame, output_png: str, pixel_size_A: float, full_star_data: pd.DataFrame = None):
    """
    Your ellipse-plotting function, unchanged in logic, but plotting in **nm**.
    Ellipse fitting is done on **pixels** (scale-invariant), then drawn after px→nm conversion.
    """
    cs = cross_section_px.copy()
    pts = cs[['rlnCoordinateX','rlnCoordinateY']].to_numpy(float)
    x = pts[:,0]; y = pts[:,1]

    try:
        ellipse_params = fit_ellipse(x, y)
        if ellipse_params['a'] is None or ellipse_params['b'] is None:
            raise ValueError("Ellipse fitting failed: invalid parameters.")

        center = [ellipse_params['X0'], ellipse_params['Y0']]
        axes   = [ellipse_params['a'],  ellipse_params['b']]
        angle  = ellipse_params['phi']
        elliptical_distortion = ellipse_params['a'] / ellipse_params['b']

        fitted = ellipse_points(center, axes, angle)

        # order points (degrees)
        angles = angle_along_ellipse(center, axes, angle, pts)
        angles = angles / np.pi * 180  # for potential diagnostics

        # base plot (nm)
        plt_obj = plot_cs(cs, pixel_size_A=pixel_size_A, output_png=None)

        # draw fitted ellipse in nm
        ex_nm = _px_to_nm(fitted[0], pixel_size_A)
        ey_nm = _px_to_nm(fitted[1], pixel_size_A)
        plt.plot(ex_nm, ey_nm, 'r--', label='Fitted Ellipse')

        # annotate distortion at mean(X,Y)
        mx_nm = float(np.mean(_px_to_nm(x, pixel_size_A)))
        my_nm = float(np.mean(_px_to_nm(y, pixel_size_A)))
        plt.text(mx_nm, my_nm, f"Elliptical distortion: {elliptical_distortion:.2f}",
                 fontsize=9, ha='center', va='center')

        # REMOVED TANGENT VECTORS	
        # optional rotation vectors from full_star_data
        #if full_star_data is not None and 'rlnHelicalTubeID' in cs.columns:
        #    rot_lookup = (full_star_data[['rlnHelicalTubeID','rlnAngleRot']]
        #                  .drop_duplicates('rlnHelicalTubeID')
        #                  .set_index('rlnHelicalTubeID')['rlnAngleRot']
        #                  .to_dict())
        #    cs['rlnAngleRot'] = cs['rlnHelicalTubeID'].map(rot_lookup)
        #    for _, row in cs.iterrows():
        #        if not pd.isna(row.get('rlnAngleRot', np.nan)):
        #            x0_nm = float(_px_to_nm(row['rlnCoordinateX'], pixel_size_A))
        #            y0_nm = float(_px_to_nm(row['rlnCoordinateY'], pixel_size_A))
        #           theta = np.deg2rad(row['rlnAngleRot'])
        #            dx, dy = np.cos(theta) * 10.0, np.sin(theta) * 10.0  # 10 nm arrow
        #            plt.arrow(x0_nm, y0_nm, dx, dy, head_width=3, head_length=5,
        #                      fc='blue', ec='blue', alpha=0.7)

        plt.legend()
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.title("Ellipse Fit of Cross section")
        plt.axis('equal')
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        return elliptical_distortion

    except Exception as e:
        print(f"WARNING: {e}")
        # fallback: just base plot
        plot_cs(cs, pixel_size_A=pixel_size_A, output_png=output_png)
        return -1.0

# ----------------------------- pipeline -----------------------------

def run_pipeline(df_in: pd.DataFrame, pixel_size_A: float,
                 fit_method: str = "ellipse",
                 propagate: bool = True,
                 renumber: bool = False,
                 out_png: str = None):
    """
    Returns (df_out, cross_section_rot, ellipse_params or None).
    df_out has rlnAngleRot propagated (if propagate), and possibly renumbered IDs.
    """
    # Use rln* columns internally
    df = strip_leading_underscore_cols(df_in)

    # Required cols
    require_cols(df, ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ','rlnHelicalTubeID'])
    for c in ['rlnAngleTilt','rlnAnglePsi']:
        if c not in df.columns:
            # fill zeros if missing (rotation still works)
            df[c] = 0.0

    # 1) cross-section (pixels)
    cs, _ = process_cross_section(df)

    # 2) rotate into Z plane + orientation consistency (pixels)
    cs_rot = rotate_cross_section(cs)
    cs_rot = enforce_consistent_cross_section_orientation(cs_rot)

    # 3) ellipse method (recommended)
    ellipse_params = None
    if fit_method.lower() == "ellipse":
        cs_with_rot, cs_with_virtual, ellipse_params = calculate_rot_angles_ellipse(cs_rot)
        cs_for_order = cs_with_rot
    else:
        # simple method (fallback) — reuse cs_rot without ellipse angles
        cs_with_rot = cs_rot.copy()
        cs_for_order = cs_rot.copy()

    # 4) get order & optional renumber
    order_ids = get_filament_order_from_rot(cs_for_order) if 'rlnAngleRot' in cs_for_order.columns else cs_for_order['rlnHelicalTubeID'].tolist()
    df_out = df.copy()
    cs_out = cs_with_rot.copy()

    if renumber and len(order_ids) > 0:
        df_out, cs_out, mapping = renumber_filament_ids(df_out, order_ids, cs_out)
        print(f"[info] Renumbered HelicalTubeID using ellipse order. Mapping: {mapping}")

    # 5) propagate rlnAngleRot to whole dataset (optional)
    if propagate and 'rlnAngleRot' in cs_out.columns:
        df_out = propagate_rot_to_entire_cilia(cs_out, df_out)

    # 6) plot (nm scaling via pixel_size_A)
    if out_png:
        plot_ellipse_cs(cs_out, output_png=out_png, pixel_size_A=pixel_size_A, full_star_data=df_out)

    # Return with original column prefixes preserved when writing
    return df_out, cs_out, ellipse_params

# ----------------------------- I/O / CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ReLAX sort: cross-section ellipse ordering & nm plotting.")
    p.add_argument("--input", required=True, help="Input STAR")
    p.add_argument("--output", required=True, help="Output STAR")
    p.add_argument("--output-png", default=None, help="Output PNG (ellipse plot, nm-scaled)")
    p.add_argument("--fit-method", default="ellipse", choices=["ellipse","simple"], help="Ordering method")
    p.add_argument("--propagate", action="store_true", help="Propagate rlnAngleRot to entire dataset")
    p.add_argument("--renumber", action="store_true", help="Renumber rlnHelicalTubeID by ellipse order")
    p.add_argument("--angpix", type=float, default=None, help="Override Å/pixel if STAR lacks _rlnImagePixelSize")
    return p.parse_args()

def to_original_prefix(df: pd.DataFrame, template_cols: list[str]) -> pd.DataFrame:
    """Write columns with leading underscore if that’s what the input had."""
    # If template has _rln*, we convert rln* back to _rln*
    uses_underscore = any(c.startswith("_rln") for c in template_cols)
    if not uses_underscore:
        return df
    out = df.copy()
    rename = {c: f"_{c}" for c in out.columns if c.startswith("rln")}
    out = out.rename(columns=rename)
    return out

def main():
    args = parse_args()

    tbl = starfile.read(args.input)
    if isinstance(tbl, dict):
        key = next(iter(tbl.keys()))
        df_in = tbl[key].copy()
        input_cols = list(df_in.columns)
    else:
        df_in = tbl.copy()
        input_cols = list(df_in.columns)

    # Pixel size (Å/px)
    pxA = infer_pixel_size_A(df_in, args.angpix)
    print(f"[info] Pixel size: {pxA:.4f} Å/px = {pxA/10.0:.4f} nm/px")

    df_out, cs_out, ellipse_params = run_pipeline(
        df_in=df_in,
        pixel_size_A=pxA,
        fit_method=args.fit_method,
        propagate=args.propagate,
        renumber=args.renumber,
        out_png=args.output_png
    )

    # Preserve input’s _rln* naming style on write
    write_obj = None
    if isinstance(tbl, dict):
        key = next(iter(tbl.keys()))
        out_block = to_original_prefix(df_out, input_cols)
        tbl[key] = out_block
        write_obj = tbl
    else:
        write_obj = to_original_prefix(df_out, input_cols)

    starfile.write(write_obj, args.output, overwrite=True)
    print(f"[info] Wrote STAR → {args.output}")
    if args.output_png:
        print(f"[info] Saved plot → {args.output_png}")

if __name__ == "__main__":
    main()


