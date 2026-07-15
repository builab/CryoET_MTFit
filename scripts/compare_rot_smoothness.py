#!/usr/bin/env python3
# coding: utf-8

"""
Compare angle smoothness between two STAR files (e.g. unsmoothed vs. Rot-smoothed
output from the predict step).

Reports two different things, on purpose kept separate:
  1. Consecutive-particle jump along each tube (the wobble metric — this is what
     should go DOWN if smoothing helped).
  2. Per-particle value shift between the two files (just confirms smoothing
     changed something — NOT a measure of wobble, and naturally grows with a
     higher smoothing factor).

Usage:
  python3 compare_rot_smoothness.py before.star after.star
  python3 compare_rot_smoothness.py before.star after.star --angle rlnAngleTilt
"""

import argparse
import numpy as np
import pandas as pd
import starfile


def load(path):
    df = starfile.read(path)
    if isinstance(df, dict):
        df = df.get('particles', next(iter(df.values())))
    return df


def consecutive_jumps(df, angle_col):
    """Per-tube max/mean absolute jump between consecutive particles along the tube."""
    rows = []
    for tube_id, g in df.groupby('rlnHelicalTubeID'):
        angles = g[angle_col].to_numpy(dtype=float)
        if len(angles) < 2:
            continue
        diffs = np.abs(np.diff(angles))
        diffs = np.minimum(diffs, 360 - diffs)
        rows.append({'tube': tube_id, 'n': len(angles),
                      'max_jump': diffs.max(), 'mean_jump': diffs.mean()})
    return pd.DataFrame(rows).sort_values('tube').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare angle wobble between two STAR files (e.g. before/after smoothing)."
    )
    parser.add_argument('file_a', help='First STAR file (e.g. unsmoothed / before)')
    parser.add_argument('file_b', help='Second STAR file (e.g. smoothed / after)')
    parser.add_argument('--angle', default='rlnAngleRot',
                         help='Angle column to check (default: rlnAngleRot)')
    parser.add_argument('--label_a', default='before', help='Label for file_a in the report')
    parser.add_argument('--label_b', default='after', help='Label for file_b in the report')
    args = parser.parse_args()

    df_a = load(args.file_a)
    df_b = load(args.file_b)

    jumps_a = consecutive_jumps(df_a, args.angle)
    jumps_b = consecutive_jumps(df_b, args.angle)

    print(f"\n=== Consecutive-particle jump in {args.angle} (wobble metric — lower is smoother) ===")
    merged = jumps_a.merge(jumps_b, on='tube', suffixes=(f'_{args.label_a}', f'_{args.label_b}'))
    for _, row in merged.iterrows():
        print(f"  tube {int(row['tube']):3d}: "
              f"{args.label_a} max={row[f'max_jump_{args.label_a}']:5.2f} mean={row[f'mean_jump_{args.label_a}']:5.2f}   "
              f"{args.label_b} max={row[f'max_jump_{args.label_b}']:5.2f} mean={row[f'mean_jump_{args.label_b}']:5.2f}")

    print(f"\n  OVERALL max jump:  {args.label_a}={jumps_a['max_jump'].max():.2f}   "
          f"{args.label_b}={jumps_b['max_jump'].max():.2f}")
    print(f"  OVERALL mean jump: {args.label_a}={jumps_a['mean_jump'].mean():.2f}   "
          f"{args.label_b}={jumps_b['mean_jump'].mean():.2f}")

    if len(df_a) == len(df_b):
        delta = np.abs(df_a[args.angle].to_numpy(float) - df_b[args.angle].to_numpy(float))
        delta = np.minimum(delta, 360 - delta)
        print(f"\n=== Per-particle value shift between the two files "
              f"(confirms smoothing changed something — NOT a wobble metric) ===")
        print(f"  max shift: {delta.max():.3f}   mean shift: {delta.mean():.3f}")
    else:
        print(f"\n[note] Particle counts differ ({len(df_a)} vs {len(df_b)}) "
              f"— skipping per-particle value-shift comparison.")


if __name__ == '__main__':
    main()
