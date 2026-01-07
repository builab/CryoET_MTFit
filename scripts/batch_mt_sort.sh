#!/bin/bash

# Batch processing script for mt_fit.py sort command
# Reads mt_fit_summary.csv and generates commands based on NO_TUBES

set -e  # Exit on error

# Configuration
CSV_FILE="mt_fit_summary.csv"
OUTPUT_DIR="sort"
MT_FIT_SCRIPT="mt_fit.py"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print header
echo "========================================================================"
echo "Batch Processing MT Fit Sort"
echo "========================================================================"
echo "CSV file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR/"
echo ""

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

# Function to calculate n_cilia based on NO_TUBES
calculate_n_cilia() {
    local no_tubes=$1
    echo $(( (no_tubes - 1) / 10 + 1 ))
}

# Read CSV file (skip header) and process each line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r star_file no_tubes no_particles median_psi; do
    # Remove leading/trailing whitespace
    star_file=$(echo "$star_file" | xargs)
    no_tubes=$(echo "$no_tubes" | xargs)
    
    # Skip empty lines
    if [ -z "$star_file" ]; then
        continue
    fi
    
    # Calculate n_cilia
    n_cilia=$(calculate_n_cilia "$no_tubes")
    
    # Generate output filename
    # Extract basename and replace _particles_processed.star with _sort.json
    basename=$(basename "$star_file")
    output_json="${OUTPUT_DIR}/${basename/_particles_processed.star/_sort.json}"
    
    # Print info
    echo "------------------------------------------------------------------------"
    echo "Processing: $basename"
    echo "  Tubes: $no_tubes -> n_cilia: $n_cilia"
    echo "  Output: $output_json"
    
    # Build and execute command
    cmd="python $MT_FIT_SCRIPT sort \"$star_file\" --n_cilia $n_cilia --export-json \"$output_json\""
    echo "  Command: $cmd"
    echo ""
    
    # Execute the command
    eval "$cmd"
    
    # Check if command succeeded
    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
        exit 1
    fi
    echo ""
done

echo "========================================================================"
echo "All processing completed!"
echo "========================================================================"