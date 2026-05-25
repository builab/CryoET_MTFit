#!/usr/bin/env python3
"""
Generate a simulated axoneme structure with 9 doublet microtubules and central pair.

Creates two STAR files:
1. doublet.star - 9 doublet microtubules arranged in a circle
2. cp.star - Central pair microtubule along the axis

Basal bodies
cilia_simulation.py --diameter 190 --length 400 --doublet_periodicity 16.3 --cp_periodicity 16.3 --rot_offset -35
"""

import argparse
import numpy as np
import pandas as pd
import starfile


def generate_doublets(
    diameter_nm: float,
    angpix: float,
    length_nm: float,
    periodicity_nm: float,
    rot_offset: float = 0.0,
    tomo_name: str = "TS_001"
) -> pd.DataFrame:
    """
    Generate 9 doublet microtubules arranged in a circle.
    
    Args:
        diameter_nm: Diameter of the cilia in nanometers.
        angpix: Pixel size in Angstroms per pixel.
        length_nm: Length of the cilia in nanometers.
        periodicity_nm: Spacing between particles along each doublet in nanometers.
        rot_offset: Constant rotation offset in degrees added to rlnAngleRot of all doublets.
        tomo_name: Tomogram name for rlnTomoName column.
    
    Returns:
        DataFrame with doublet particles.
    """
    
    # Convert units
    diameter_angstrom = diameter_nm * 10  # nm to Angstrom
    length_angstrom = length_nm * 10
    periodicity_angstrom = periodicity_nm * 10
    
    # Calculate radius for doublet placement
    radius_angstrom = diameter_angstrom / 2.0
    radius_px = radius_angstrom / angpix
    
    # Origin position in pixels (cilia axis parallel to Y)
    origin_x_px = 100.0
    origin_y_px = 0.0
    origin_z_px = 100.0
    
    # Calculate number of particles per doublet
    n_particles_per_doublet = int(np.ceil(length_angstrom / periodicity_angstrom)) + 1
    
    # Generate particles for 9 doublets
    particles = []
    n_doublets = 9
    
    print(f"Generating doublet microtubules:")
    print(f"  Diameter: {diameter_nm} nm ({diameter_angstrom} Å, {diameter_angstrom/angpix:.2f} px)")
    print(f"  Length: {length_nm} nm ({length_angstrom} Å)")
    print(f"  Doublet periodicity: {periodicity_nm} nm ({periodicity_angstrom} Å)")
    print(f"  Rotation offset: {rot_offset}° (constant per doublet)")
    print(f"  Pixel size: {angpix} Å/px")
    print(f"  Origin: ({origin_x_px}, {origin_y_px}, {origin_z_px}) px")
    print(f"  Particles per doublet: {n_particles_per_doublet}")
    print(f"  Number of doublets: {n_doublets}")
    
    for doublet_id in range(1, n_doublets + 1):
        # Calculate angular position for this doublet (40 degrees apart)
        # Add constant rotation offset for all particles in this doublet
        base_angle = (doublet_id - 1) * 40.0
        angle_deg = base_angle + rot_offset
        
        # Convert to RELION convention: -180 to 180
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        
        # Use base angle (without offset) for spatial positioning
        angle_rad = np.radians(base_angle)
        
        # Calculate X, Z offset from center for this doublet (axis parallel to Y)
        offset_x_px = radius_px * np.cos(angle_rad)
        offset_z_px = radius_px * np.sin(angle_rad)
        
        # Generate particles along Y-axis for this doublet
        for i in range(n_particles_per_doublet):
            y_angstrom = i * periodicity_angstrom
            y_px = y_angstrom / angpix
            
            # Skip if beyond length
            if y_angstrom > length_angstrom:
                break
            
            particle = {
                'rlnTomoName': tomo_name,
                'rlnCoordinateX': origin_x_px + offset_x_px,
                'rlnCoordinateY': origin_y_px + y_px,
                'rlnCoordinateZ': origin_z_px + offset_z_px,
                'rlnAngleRot': angle_deg,
                'rlnAngleTilt': 90.0,
                'rlnAnglePsi': 90.0,
                'rlnImagePixelSize': angpix,
                'rlnHelicalTubeID': doublet_id
            }
            particles.append(particle)
    
    df = pd.DataFrame(particles)
    
    print(f"  ✓ Generated {len(df)} particles across {n_doublets} doublets")
    
    # Print summary statistics
    for doublet_id in range(1, n_doublets + 1):
        n_particles = len(df[df['rlnHelicalTubeID'] == doublet_id])
        base_angle = (doublet_id - 1) * 40.0
        if base_angle > 180:
            base_angle -= 360
        print(f"    Doublet {doublet_id}: {n_particles} particles starting at {base_angle}°")
    
    return df


def generate_central_pair(
    angpix: float,
    length_nm: float,
    periodicity_nm: float,
    tomo_name: str = "TS_001"
) -> pd.DataFrame:
    """
    Generate central pair microtubule along the axis.
    
    Args:
        angpix: Pixel size in Angstroms per pixel.
        length_nm: Length of the cilia in nanometers.
        periodicity_nm: Spacing between particles along central pair in nanometers.
        tomo_name: Tomogram name for rlnTomoName column.
    
    Returns:
        DataFrame with central pair particles.
    """
    
    # Convert units
    length_angstrom = length_nm * 10
    periodicity_angstrom = periodicity_nm * 10
    
    # Origin position in pixels (cilia axis parallel to Y)
    origin_x_px = 100.0
    origin_y_px = 0.0
    origin_z_px = 100.0
    
    # Calculate number of particles
    n_particles = int(np.ceil(length_angstrom / periodicity_angstrom)) + 1
    
    # Generate particles along Y-axis
    particles = []
    
    print(f"\nGenerating central pair microtubule:")
    print(f"  Length: {length_nm} nm ({length_angstrom} Å)")
    print(f"  CP periodicity: {periodicity_nm} nm ({periodicity_angstrom} Å)")
    print(f"  Origin: ({origin_x_px}, {origin_y_px}, {origin_z_px}) px")
    print(f"  Particles: {n_particles}")
    
    for i in range(n_particles):
        y_angstrom = i * periodicity_angstrom
        y_px = y_angstrom / angpix
        
        # Skip if beyond length
        if y_angstrom > length_angstrom:
            break
        
        particle = {
            'rlnTomoName': tomo_name,
            'rlnCoordinateX': origin_x_px,
            'rlnCoordinateY': origin_y_px + y_px,
            'rlnCoordinateZ': origin_z_px,
            'rlnAngleRot': 0.0,
            'rlnAngleTilt': 90.0,
            'rlnAnglePsi': 90.0,
            'rlnImagePixelSize': angpix,
            'rlnHelicalTubeID': 10
        }
        particles.append(particle)
    
    df = pd.DataFrame(particles)
    
    print(f"  ✓ Generated {len(df)} particles")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate simulated axoneme structure with doublets and central pair.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--diameter',
        type=float,
        required=True,
        help='Diameter of the cilia in nanometers (e.g., 270)'
    )
    parser.add_argument(
        '--angpix',
        type=float,
        default=10.0,
        help='Pixel size in Angstroms per pixel'
    )
    parser.add_argument(
        '--length',
        type=float,
        required=True,
        help='Length of the cilia in nanometers (e.g., 300)'
    )
    parser.add_argument(
        '--doublet_periodicity',
        type=float,
        required=True,
        help='Periodicity between particles in doublets in nanometers (e.g., 16.3)'
    )
    parser.add_argument(
        '--cp_periodicity',
        type=float,
        required=True,
        help='Periodicity between particles in central pair in nanometers (e.g., 16.3)'
    )
    parser.add_argument(
        '--rot_offset',
        type=float,
        default=0.0,
        help='Constant rotation offset in degrees added to all doublets (e.g., 30 for centrioles)'
    )
    parser.add_argument(
        '--tomo_name',
        type=str,
        default='TS_001',
        help='Tomogram name for rlnTomoName column'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.diameter <= 0:
        parser.error("Diameter must be positive")
    if args.angpix <= 0:
        parser.error("Pixel size must be positive")
    if args.length <= 0:
        parser.error("Length must be positive")
    if args.doublet_periodicity <= 0:
        parser.error("Doublet periodicity must be positive")
    if args.cp_periodicity <= 0:
        parser.error("CP periodicity must be positive")
    
    # Generate doublets
    df_doublets = generate_doublets(
        diameter_nm=args.diameter,
        angpix=args.angpix,
        length_nm=args.length,
        periodicity_nm=args.doublet_periodicity,
        rot_offset=args.rot_offset,
        tomo_name=args.tomo_name
    )
    
    # Generate central pair
    df_cp = generate_central_pair(
        angpix=args.angpix,
        length_nm=args.length,
        periodicity_nm=args.cp_periodicity,
        tomo_name=args.tomo_name
    )
    
    # Write STAR files
    doublet_file = 'doublet.star'
    cp_file = 'cp.star'
    
    starfile.write(df_doublets, doublet_file, overwrite=True)
    starfile.write(df_cp, cp_file, overwrite=True)
    
    print(f"\n✓ STAR files written:")
    print(f"  Doublets: {doublet_file}")
    print(f"  Central pair: {cp_file}")


if __name__ == '__main__':
    main()