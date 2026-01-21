#!/usr/bin/env python3
"""
CLI wrapper for filament processing pipeline with subcommands:
- fit     : Initial curve fitting and clustering
- clean   : Remove overlapping tubes
- connect : Connect broken tube segments
- predict : Predict angles from template matching
- pipeline: Run full pipeline (fit -> clean -> connect -> predict)

@Builab 2025
"""

__version__ = "1.0.0"
__author__ = "Builab"
__date__ = "2025-01-27"

import sys
import os
import argparse
import pandas as pd

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.fit import fit_curves
from utils.connect import connect_tubes
from utils.predict import predict_angles
from utils.sort import group_and_sort

from utils.clean import (
    clean_tubes,
    filter_short_tubes,
    filter_by_direction
)
    
from utils.io import read_star, write_star, validate_dataframe, load_coordinates

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_LCC_KEEP_PERCENTAGE = 70.0  # Percentage of top LCC particles to keep

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_step_header(step_num: int, step_name: str) -> None:
    """Print formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80)


def print_summary(title: str, items: list) -> None:
    """Print formatted summary section."""
    print(f"\n{title}")
    print("-" * len(title))
    for item in items:
        print(f"  {item}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"[INFO] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"[WARNING] {message}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"[SUCCESS] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"[ERROR] {message}", file=sys.stderr)        
        
def save_output_or_exit(df: pd.DataFrame, output_file: str, success_msg: str = None) -> None:
    """
    Save dataframe to STAR file or exit if empty.
    
    Parameters:
    -----------
    df : DataFrame to save
    output_file : path to output file
    success_msg : optional custom success message (default: "Output saved to: {output_file}")
    """
    if not df.empty:
        write_star(df, output_file, overwrite=True)
        msg = success_msg if success_msg else f"Output saved to: {output_file}"
        print_success(msg)
    else:
        print_warning("No output particles generated")
        sys.exit(1)

# =============================================================================
# ARGUMENT PARSERS
# =============================================================================

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to multiple subcommands."""
    parser.add_argument('input', help='Input STAR file')
    parser.add_argument('-o', '--output', help='Output STAR file (default: auto-generated)')
    parser.add_argument('--angpix', type=float, default=14.00,
                       help='Pixel size in Angstroms/pixel (default: 14.00)')


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add fitting-specific arguments."""
    parser.add_argument('--poly_order', type=int, default=3,
                       help='Polynomial order for fitting (default: 3)')
    parser.add_argument('--min_seed', type=int, default=6,
                       help='Minimum seed points (default: 6)')
    parser.add_argument('--sample_step', type=float, default=83.0,
                       help='Resampling step in Angstroms (default: 83.0)')
    
    # Advanced fit parameters (hidden from help)
    parser.add_argument('--max_dis_to_line_ang', type=float, default=50, help=argparse.SUPPRESS)
    parser.add_argument('--min_dis_neighbor_seed_ang', type=float, default=60, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_neighbor_seed_ang', type=float, default=320, help=argparse.SUPPRESS)
    parser.add_argument('--poly_order_seed', type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument('--max_seed_fitting_error', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--max_angle_change_per_4nm', type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_to_curve_ang', type=float, default=80, help=argparse.SUPPRESS)
    parser.add_argument('--min_dis_neighbor_curve_ang', type=float, default=60, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_neighbor_curve_ang', type=float, default=320, help=argparse.SUPPRESS)
    parser.add_argument('--min_number_growth', type=int, default=0, help=argparse.SUPPRESS)


def add_clean_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cleaning-specific arguments."""
    parser.add_argument('--dist_thres', type=float, default=50,
                       help='Overlap removal threshold in Angstroms (default: 50)')
    parser.add_argument('--margin', type=float, default=500,
                       help='Bounding box margin in Angstroms (default: 500)')
    parser.add_argument('--psi_min', type=float, default=0.0,
                   help='Minimum rlnAnglePsi angle in degrees (default: 0)')
    parser.add_argument('--psi_max', type=float, default=180.0,
                   help='Maximum rlnAnglePsi angle in degrees (default: 180)')
    parser.add_argument('--direction_dev', type=float, default=0.0,
                   help='Maximum range for deviation from median angle in degrees (default: 0 (do nothing))')
    parser.add_argument('--direction_angle', type=str, default='Psi',
                   help='Angle for median filtering (Rot/Tilt/Psi) (default: Psi)')
                   
                   
def add_connect_arguments(parser: argparse.ArgumentParser) -> None:
    """Add connection-specific arguments."""
    parser.add_argument('--dist_extrapolate', type=float, required=True,
                       help='Initial extrapolation distance in Angstroms (REQUIRED)')
    parser.add_argument('--overlap_thres', type=float, required=True,
                       help='Connection overlap threshold in Angstroms (REQUIRED)')
    parser.add_argument('--iterations', type=int, default=2,
                       help='Connection iterations (default: 2)')
    parser.add_argument('--dist_scale', type=float, default=1.5,
                       help='Distance scale factor per iteration (default: 1.5)')
    parser.add_argument('--min_seed', type=int, default=5,
                       help='Minimum seed points (default: 5)')
    parser.add_argument('--poly_order', type=int, default=3,
                       help='Polynomial order for refitting (default: 3)')
    parser.add_argument('--sample_step', type=float, default=82.0,
                       help='Resampling step for refitting in Angstroms (default: 82.0)')
    parser.add_argument('--min_part_per_tube', type=int, default=5,
                       help='Minimum particles per tube (default: 5)')
    parser.add_argument('--poly_order_seed', type=int, default=1, help=argparse.SUPPRESS)


def add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    """Add predict-specific arguments."""
    parser.add_argument('--template', type=str, default=None,
                       help='Template STAR file for angle prediction')
    parser.add_argument('--neighbor_rad', type=float, default=100,
                       help='Radius for finding neighbor particles in Angstroms (default: 100)')
    parser.add_argument('--max_delta_deg', type=float, default=20,
                       help='Max angle deviation within same tube in degrees (default: 20)')
    parser.add_argument('--lcc_keep_percentage', type=float, default=DEFAULT_LCC_KEEP_PERCENTAGE,
                       help=f'Percentage of LCC particles to keep (default: {DEFAULT_LCC_KEEP_PERCENTAGE})')
    parser.add_argument('--direction', type=int, choices=[0, 1], default=0, 
                            help='0: Keep as is, 1: Flip Psi direction')                   
          
                       
def add_sort_arguments(parser: argparse.ArgumentParser) -> None:
    """Add predict-specific arguments."""
    parser.add_argument('--n_cilia', type=int, default=None,
                       help='Number of cilia in the tomogram')
    parser.add_argument('--tilt_psi_threshold', type=float, default=10,
                       help='Tilt/Psi threshold for cilia group (default: 10)')
    parser.add_argument('--rot_threshold', type=float, default=8,
                       help='Max rot angle deviation within same tube for doublet group (default: 8)')
    parser.add_argument('--coord_threshold', type=float, default=900,
                       help='Max spacial distance threshold for cilia classification (default: 900)')
    parser.add_argument('--fit_method', default="ellipse", choices=["ellipse","simple"], help="Ordering method")
    parser.add_argument('--enforce_9_doublets', action='store_true', help="Force exactly 9 doublets per cilium")
    parser.add_argument('--export-json', type=str, metavar='FILE',
                       help='Export automatic grouping to JSON file for record keeping')
    
    # Manual grouping options (mutually exclusive)
    manual_group = parser.add_mutually_exclusive_group()
    manual_group.add_argument('--manual', type=str, metavar='FILE',
                       help='Manual grouping file (JSON format)')

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_fitting(file_path: str, args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """
    Run initial curve fitting and clustering.
    Always resample twice the frequency of the requested sample_step to make denser lines
    
    Parameters
    ----------
    file_path : str
        Path to input STAR file
    args : argparse.Namespace
        Command line arguments
    step_num : int, optional
        Step number for pipeline execution
        
    Returns
    -------
    pd.DataFrame
        Resampled particles with tube IDs
    """
    if step_num is not None:
        print_step_header(step_num, "CURVE FITTING & CLUSTERING")
    else:
        print("="*80)
        print("CURVE FITTING & CLUSTERING")
        print("="*80)
    
    pixel_size = args.angpix
    
    # Load coordinates
    coords, tomo_name, pixel_size = load_coordinates(file_path, pixel_size)
    
    if coords is None:
        raise ValueError("Failed to load coordinates from input file")
    
    print_info(f"Loaded {len(coords)} particles from {tomo_name}")
    
    # Run fitting
    df_resampled, _, cluster_count = fit_curves(
        coords=coords,
        tomo_name=tomo_name,
        angpix=pixel_size,
        poly_order=args.poly_order,
        sample_step=args.sample_step / pixel_size,
        integration_step=1.0 / pixel_size,
        min_seed=args.min_seed,
        max_distance_to_line=args.max_dis_to_line_ang / pixel_size,
        min_distance_in_extension_seed=args.min_dis_neighbor_seed_ang / pixel_size,
        max_distance_in_extension_seed=args.max_dis_neighbor_seed_ang / pixel_size,
        poly_order_seed=args.poly_order_seed,
        seed_evaluation_constant=args.max_seed_fitting_error,
        max_angle_change_per_4nm=args.max_angle_change_per_4nm,
        max_distance_to_curve=args.max_dis_to_curve_ang / pixel_size,
        min_distance_in_extension=args.min_dis_neighbor_curve_ang / pixel_size,
        max_distance_in_extension=args.max_dis_neighbor_curve_ang / pixel_size,
        min_number_growth=args.min_number_growth,
    )
    
    print_summary("Fitting Results", [
        f"Tubes found: {cluster_count}",
        f"Particles resampled: {len(df_resampled)}"
    ])
    
    return df_resampled


def run_cleaning(df_input: pd.DataFrame, args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """
    Remove overlapping tubes.
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame with particles
    args : argparse.Namespace
        Command line arguments
    step_num : int, optional
        Step number for pipeline execution
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with overlapping tubes removed
    """
    if step_num is not None:
        print_step_header(step_num, "CLEANING (OVERLAP REMOVAL)")
    else:
        print("=" * 80)
        print("CLEANING (OVERLAP REMOVAL)")
        print("=" * 80)

    print_info(f"Distance threshold: {args.dist_thres} Å")
    print_info(f"Bounding box margin: {args.margin} Å")
    
    if df_input['rlnHelicalTubeID'].nunique() == 1:
        print('Only 1 tube found. Skip cleaning!')
        return df_input
    
    df_filtered = clean_tubes(df=df_input,
        angpix=args.angpix,
        distance_threshold=args.dist_thres,
        margin=args.margin,
        psi_min=args.psi_min,
        psi_max=args.psi_max,
        direction_angle=args.direction_angle,
        direction_max_dev=args.direction_dev
    )

    return df_filtered


def run_connection(
    df_input: pd.DataFrame,
    args: argparse.Namespace,
    step_num: int = None
) -> pd.DataFrame:
    """
    Connect broken tube segments using iterative trajectory extrapolation.
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame with particles
    args : argparse.Namespace
        Command line arguments containing connection parameters
    step_num : int, optional
        Step number for pipeline execution
        
    Returns
    -------
    pd.DataFrame
        DataFrame with connected and refitted tubes
    """
    if step_num is not None:
        print_step_header(step_num, "CONNECTION (TRAJECTORY EXTRAPOLATION)")
    else:
        print("="*80)
        print("CONNECTION (TRAJECTORY EXTRAPOLATION)")
        print("="*80)
    
    # Run connection pipeline (without filtering)
    df_connected = connect_tubes(
        df=df_input,
        angpix=args.angpix,
        overlap_threshold=args.overlap_thres,
        min_seed=args.min_seed,
        dist_extrapolate=args.dist_extrapolate,
        poly_order_seed=args.poly_order_seed,
        poly_order_final=args.poly_order,
        sample_step=args.sample_step,
        max_iterations=args.iterations,
        dist_scale=args.dist_scale,
        debug=True
    )
    		
    # Filter short tubes if requested
    if args.min_part_per_tube > 0:
        tubes_before = df_connected['rlnHelicalTubeID'].nunique()
        df_final = filter_short_tubes(df_connected, args.min_part_per_tube)
        tubes_after = df_final['rlnHelicalTubeID'].nunique()
        tubes_removed = tubes_before - tubes_after
        
        if tubes_removed > 0:
            print_info(f"Filtered out {tubes_removed} tubes with < {args.min_part_per_tube} particles")
            print_info(f"Final: {tubes_after} tubes, {len(df_final)} particles")
    else:
        df_final = df_connected
    
    return df_final


def run_prediction(df_input: pd.DataFrame, df_template: pd.DataFrame, 
                  args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """
    Predict angles based on template matching.
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame with particles (fitted tubes)
    df_template : pd.DataFrame
        Template DataFrame with reference angles
    args : argparse.Namespace
        Command tube arguments
    step_num : int, optional
        Step number for pipeline execution
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predicted angles
    """
    if step_num is not None:
        print_step_header(step_num, "PREDICT (ANGLE BASED ON TEMPLATE MATCH)")
    else:
        print("="*80)
        print("PREDICT (ANGLE BASED ON TEMPLATE MATCH)")
        print("="*80)

    print_info(f"Neighbor radius: {args.neighbor_rad} Å")
    print_info(f"Template file: {args.template}")
    print_info(f"LCC keep percentage: {args.lcc_keep_percentage}%")
    
    # 2. Load the template (Conditional)
    df_template = None
    if args.template is not None:
        # Only call read_star if a path was actually provided
        print(f"[INFO] Loading template from: {args.template}")
        df_template = read_star(args.template)
    else:
        print("[INFO] No template provided. Operating in template-free mode.")

    df_all = predict_angles(
        df_input=df_input,
        df_template=df_template,
        angpix=args.angpix,
        neighbor_radius=args.neighbor_rad,
        lcc_keep_percent=args.lcc_keep_percentage,
        snap_max_delta=args.max_delta_deg,
        snap_min_points=5,
        direction=args.direction)
    
    print_summary("Prediction Results", [
        f"Particles with predicted angles: {len(df_all[0])}"
    ])
    
    return df_all[0]

def run_sort(df_input: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Predict angles based on template matching.
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame with particles (fitted tubes)
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sorted doublet
    """
    
    print("="*80)
    print("GROUP CILIA AND SORT DOUBLET NUMBER")
    print("="*80)
    
    print("This step is optional and be useful for Chlamydomonas.")
    print("This step is NOT Tested yet for parallel cilia.")
        
    print_info(f"Sorting method: {args.fit_method}")
    print_info(f"Tilt/Psi threshold: {args.tilt_psi_threshold}")
    print_info(f"Distance Threshold: {args.coord_threshold} Å")
    print_info(f"Rot threshold: {args.rot_threshold}")
    if args.enforce_9_doublets:
        print_info(f"Enforce 9 doublets: True")
    
    # ADD: Check if export_json is requested
    export_json = getattr(args, 'export_json', None)
    if export_json:
        print_info(f"Will export grouping to: {export_json}")

    df_sorted = group_and_sort(
        df=df_input,
        angpix=args.angpix,
        n_cilia=args.n_cilia,
        tilt_psi_threshold=args.tilt_psi_threshold,
        rot_threshold=args.rot_threshold,
        enforce_9_doublets=args.enforce_9_doublets,
        fit_method=args.fit_method,
        out_png=args.out_png,
        export_json=export_json  # ADD THIS
    )
    
    return df_sorted
    
# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def cmd_fit(args: argparse.Namespace) -> None:
    """Execute fit subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_fitted.star"
    
    try:
        df_fitted = run_fitting(args.input, args)
        
        save_output_or_exit(df_fitted, output_file)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)


def cmd_clean(args: argparse.Namespace) -> None:
    """Execute clean subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_cleaned.star"
    
    try:
        # Read input
        df_input = read_star(args.input)
        
        df_cleaned = run_cleaning(df_input, args)
                
        save_output_or_exit(df_cleaned, output_file)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)


def cmd_connect(args: argparse.Namespace) -> None:
    """Execute connect subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_connected.star"
    
    try:
        # Read input
        df_input = read_star(args.input)
        
        df_connected = run_connection(df_input, args)
        
        save_output_or_exit(df_connected, output_file)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)


def cmd_predict(args: argparse.Namespace) -> None:
    """Execute predict subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_predicted.star"
    
    try:
        # Read input files
        df_input = read_star(args.input)
        df_template = None
        if args.template is not None:        
           df_template = read_star(args.template) 
        
        df_predicted = run_prediction(df_input, df_template, args)
        
        save_output_or_exit(df_predicted, output_file)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)
        
def cmd_sort(args: argparse.Namespace) -> None:
    """Execute sort subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_sorted.star"
    
    try:
        df_input = read_star(args.input)
        tomo = df_input['rlnTomoName'].iloc[0]
        
        # Generate JSON filename if export requested but no name given
        if hasattr(args, 'export_json') and args.export_json:
            json_file = args.export_json
        else:
            # Auto-generate JSON filename based on input
            json_file = f"{os.path.splitext(args.input)[0]}_grouping.json"
            args.export_json = json_file if not hasattr(args, 'manual') or not args.manual else None
        
        # Check for manual grouping
        if hasattr(args, 'manual') and args.manual:
            print("Using manual grouping from JSON file...")
            from utils.sort import manual_group_and_sort
            df_sorted = manual_group_and_sort(
                df=df_input,
                manual_json=args.manual,
                angpix=args.angpix,
                fit_method=args.fit_method,
                out_png=f"{tomo}.png"
            )
        else:
            # Automatic grouping
            args.out_png = f"{tomo}.png"
            df_sorted = run_sort(df_input, args)
        
        save_output_or_exit(df_sorted, output_file)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Execute full pipeline."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_processed.star"
    
    # Print pipeline header
    print("="*80)
    print("FILAMENT PROCESSING PIPELINE")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Pixel size: {args.angpix} Å/px")
    print(f"Sample step: {args.sample_step} Å")
    print(f"Polynomial order: {args.poly_order}")
    print(f"Minimum number of seed: {args.min_seed}")
    print(f"Distance threshold: {args.dist_thres} Å")
    print(f"Distance extrapolate: {args.dist_extrapolate} Å")
    print(f"Overlap threshold: {args.overlap_thres} Å")
    print(f"Neighbor radius: {args.neighbor_rad} Å")
    print(f"Template star file: {args.template}")
    
    try:
        # Run pipeline steps
        df_fitted = run_fitting(args.input, args, step_num=1)
        df_cleaned = run_cleaning(df_fitted, args, step_num=2)
        df_connected = run_connection(df_cleaned, args, step_num=3)
        
        # Load template for prediction
        df_template = read_star(args.template)
        
        df_final = run_prediction(df_connected, df_template, args, step_num=4)
        
        # Save output
        if not df_final.empty:
            write_star(df_final, output_file, overwrite=True)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            print_summary("Final Output", [
                f"File, Tubes, Particles,Orientation(PsiAngle)",
                f"{output_file},{df_final['rlnHelicalTubeID'].nunique()},{len(df_final)},{df_final['rlnAnglePsi'].median():.2f}"
            ])
        else:
            print_warning("Pipeline produced no output particles")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"{e}")
        sys.exit(1)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point with subcommand parsing."""
    parser = argparse.ArgumentParser(
        description='Filament processing toolkit for cryo-ET data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  fit       Initial curve fitting and clustering
  clean     Remove overlapping tubes
  connect   Connect broken tube segments
  predict   Predict angles using template matching
  sort   	Group cilia and sort doublet order
  pipeline  Run full pipeline (fit -> clean -> connect -> predict)

Examples:
  # Run individual steps
  %(prog)s fit input.star --angpix 14 --sample_step 82
  %(prog)s clean fitted.star --dist_thres 100
  %(prog)s connect cleaned.star --dist_extrapolate 1500 --overlap_thres 80 --min_part_per_tube 5
  %(prog)s predict connected.star --template input.star --neighbor_rad 100 --max_delta_deg 20
  %(prog)s sort processed.star --n_cilia 2 --tilt_psi_threshold 10 --rot_threshold 8

  
  # Run full pipeline (no sort)
  %(prog)s pipeline input.star --angpix 14 --sample_step 82 \\
           --dist_thres 100 --dist_extrapolate 1500 --overlap_thres 80 \\
           --min_part_per_tube 5 --neighbor_rad 100 --template input.star
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # FIT subcommand
    fit_parser = subparsers.add_parser('fit', help='Initial curve fitting and clustering')
    add_common_arguments(fit_parser)
    add_fit_arguments(fit_parser)
    fit_parser.set_defaults(func=cmd_fit)
    
    # CLEAN subcommand
    clean_parser = subparsers.add_parser('clean', help='Remove overlapping tubes')
    add_common_arguments(clean_parser)
    add_clean_arguments(clean_parser)
    clean_parser.set_defaults(func=cmd_clean)
    
    # CONNECT subcommand
    connect_parser = subparsers.add_parser('connect', help='Connect broken tube segments')
    add_common_arguments(connect_parser)
    add_connect_arguments(connect_parser)
    connect_parser.set_defaults(func=cmd_connect)
    
    # PREDICT subcommand
    predict_parser = subparsers.add_parser('predict', help='Predict angles from template')
    add_common_arguments(predict_parser)
    add_predict_arguments(predict_parser)
    predict_parser.set_defaults(func=cmd_predict)
    
    # SORT subcommand
    sort_parser = subparsers.add_parser('sort', help='Group cilia and sort doublet order')
    add_common_arguments(sort_parser)
    add_sort_arguments(sort_parser)
    sort_parser.set_defaults(func=cmd_sort)
    
    # PIPELINE subcommand
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    add_common_arguments(pipeline_parser)
    add_fit_arguments(pipeline_parser)
    add_clean_arguments(pipeline_parser)
    add_predict_arguments(pipeline_parser)
    
    # Connection arguments for pipeline
    pipeline_parser.add_argument('--dist_extrapolate', type=float, required=True,
                                help='Initial extrapolation distance in Angstroms (REQUIRED)')
    pipeline_parser.add_argument('--overlap_thres', type=float, required=True,
                                help='Connection overlap threshold in Angstroms (REQUIRED)')
    pipeline_parser.add_argument('--iterations', type=int, default=2,
                                help='Connection iterations (default: 2)')
    pipeline_parser.add_argument('--dist_scale', type=float, default=1.5,
                                help='Distance scale factor per iteration (default: 1.5)')
    pipeline_parser.add_argument('--min_part_per_tube', type=int, default=5,
                                help='Minimum particles per tube (default: 5)')
    
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Execute subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
