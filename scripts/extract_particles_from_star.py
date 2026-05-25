#!/usr/bin/env python3
"""
Extract particles from RELION STAR files based on rlnTomoName and/or rlnHelicalTubeID.

Usage:
    python extract_particles_warpstar.py --i input.star --o output.star --rlnTomoName TS_001.tomostar
    python extract_particles_warpstar.py --i input.star --o output.star --rlnHelicalTubeID 2
    python extract_particles_warpstar.py --i input.star --o output.star --rlnTomoName TS_001.tomostar --rlnHelicalTubeID 2
    
Builab@McGill 2025
"""

import argparse
import sys
try:
    import starfile
except ImportError:
    print("Error: starfile module not found. Install with: pip install starfile")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract particles from STAR file based on rlnTomoName and/or rlnHelicalTubeID',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--i', '--input', dest='input_star', required=True,
                        help='Input STAR file')
    parser.add_argument('--o', '--output', dest='output_star', required=True,
                        help='Output STAR file')
    parser.add_argument('--rlnTomoName', dest='tomo_name', default=None,
                        help='Filter by rlnTomoName (e.g., TS_001.tomostar)')
    parser.add_argument('--rlnHelicalTubeID', dest='tube_id', type=int, default=None,
                        help='Filter by rlnHelicalTubeID (integer value)')
    
    args = parser.parse_args()
    
    # Check that at least one filter is provided
    if args.tomo_name is None and args.tube_id is None:
        parser.error('At least one of --rlnTomoName or --rlnHelicalTubeID must be specified')
    
    return args


def extract_particles(input_file, output_file, tomo_name=None, tube_id=None):
    """
    Extract particles from STAR file based on filtering criteria.
    
    Args:
        input_file: Path to input STAR file
        output_file: Path to output STAR file
        tomo_name: Value to filter by rlnTomoName (optional)
        tube_id: Value to filter by rlnHelicalTubeID (optional)
    """
    print(f"Reading STAR file: {input_file}")
    
    # Read the STAR file
    try:
        star_data = starfile.read(input_file)
    except Exception as e:
        print(f"Error reading STAR file: {e}")
        sys.exit(1)
    
    # Check if it's a multi-block STAR file
    if not isinstance(star_data, dict):
        print("Error: Expected a multi-block STAR file with data_particles block")
        sys.exit(1)
    
    # Find the particles block (case-insensitive)
    particles_key = None
    for key in star_data.keys():
        if 'particles' in key.lower():
            particles_key = key
            break
    
    if particles_key is None:
        print("Error: No data_particles block found in STAR file")
        print(f"Available blocks: {list(star_data.keys())}")
        sys.exit(1)
    
    particles_df = star_data[particles_key]
    print(f"Total particles before filtering: {len(particles_df)}")
    
    # Build filter conditions
    filter_mask = None
    filter_desc = []
    
    if tomo_name is not None:
        if 'rlnTomoName' not in particles_df.columns:
            print("Error: rlnTomoName column not found in particles block")
            print(f"Available columns: {list(particles_df.columns)}")
            sys.exit(1)
        
        tomo_mask = particles_df['rlnTomoName'] == tomo_name
        filter_mask = tomo_mask if filter_mask is None else filter_mask & tomo_mask
        filter_desc.append(f"rlnTomoName == '{tomo_name}'")
    
    if tube_id is not None:
        if 'rlnHelicalTubeID' not in particles_df.columns:
            print("Error: rlnHelicalTubeID column not found in particles block")
            print(f"Available columns: {list(particles_df.columns)}")
            sys.exit(1)
        
        tube_mask = particles_df['rlnHelicalTubeID'] == tube_id
        filter_mask = tube_mask if filter_mask is None else filter_mask & tube_mask
        filter_desc.append(f"rlnHelicalTubeID == {tube_id}")
    
    # Apply filter
    print(f"Filtering with: {' AND '.join(filter_desc)}")
    filtered_particles = particles_df[filter_mask]
    
    print(f"Particles after filtering: {len(filtered_particles)}")
    
    if len(filtered_particles) == 0:
        print("Warning: No particles matched the filter criteria")
    
    # Create output data structure with filtered particles
    output_data = star_data.copy()
    output_data[particles_key] = filtered_particles
    
    # Write output STAR file
    print(f"Writing output to: {output_file}")
    try:
        starfile.write(output_data, output_file)
        print("Done!")
    except Exception as e:
        print(f"Error writing STAR file: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    extract_particles(
        args.input_star,
        args.output_star,
        tomo_name=args.tomo_name,
        tube_id=args.tube_id
    )


if __name__ == '__main__':
    main()