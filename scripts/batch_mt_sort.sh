#!/bin/bash

# Batch processing script for mt_fit.py sort command
# Reads mt_fit_summary.csv and generates commands based on NO_TUBES
# Outputs a summary report at the end

set +e  # Allow script to continue on errors

# Configuration
CSV_FILE="mt_fit_summary.csv"
OUTPUT_DIR="sort"
MT_FIT_SCRIPT="mt_fit.py"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_REPORT="batch_mt_fit_summary_${TIMESTAMP}.txt"

# Temporary file for tracking results
TEMP_RESULTS=$(mktemp)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to calculate n_cilia based on NO_TUBES
calculate_n_cilia() {
    local no_tubes=$1
    echo $(( (no_tubes - 1) / 10 + 1 ))
}

# Print header
echo "========================================================================"
echo "Batch Processing MT Fit Sort"
echo "========================================================================"
echo "Started at: $(date)"
echo "CSV file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR/"
echo ""

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

# Count total entries
TOTAL=$(tail -n +2 "$CSV_FILE" | grep -v '^[[:space:]]*$' | wc -l | xargs)
echo "Total entries to process: $TOTAL"
echo ""

# Initialize counters
SUCCESS=0
FAILED=0

# Process each line
COUNTER=0
tail -n +2 "$CSV_FILE" | while IFS=',' read -r star_file no_tubes no_particles median_psi; do
    # Remove leading/trailing whitespace
    star_file=$(echo "$star_file" | xargs)
    no_tubes=$(echo "$no_tubes" | xargs)
    
    # Skip empty lines
    if [ -z "$star_file" ]; then
        continue
    fi
    
    COUNTER=$((COUNTER + 1))
    
    # Calculate n_cilia
    n_cilia=$(calculate_n_cilia "$no_tubes")
    
    # Generate output filename
    basename=$(basename "$star_file")
    output_json="${OUTPUT_DIR}/${basename/_particles_processed.star/_sort.json}"
    
    # Print progress
    echo "[$COUNTER/$TOTAL] Processing: $basename (tubes: $no_tubes, n_cilia: $n_cilia)"
    
    # Check if input file exists
    if [ ! -f "$star_file" ]; then
        echo "  ✗ FAILED: Input file not found"
        echo "$basename|FAILED|FILE_NOT_FOUND|$star_file" >> "$TEMP_RESULTS"
        continue
    fi
    
    # Build and execute command (suppress output)
    cmd="$MT_FIT_SCRIPT sort \"$star_file\" --n_cilia $n_cilia --export-json \"$output_json\""
    eval "$cmd" > /dev/null 2>&1
    EXIT_CODE=$?
    
    # Check if command succeeded
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ SUCCESS"
        echo "$basename|SUCCESS|$n_cilia|$output_json" >> "$TEMP_RESULTS"
    else
        echo "  ✗ FAILED (Exit code: $EXIT_CODE)"
        echo "$basename|FAILED|EXIT_CODE_$EXIT_CODE|$n_cilia" >> "$TEMP_RESULTS"
    fi
done

# Count results
SUCCESS=$(grep -c "SUCCESS" "$TEMP_RESULTS" 2>/dev/null | xargs || echo 0)
FAILED=$(grep -c "FAILED" "$TEMP_RESULTS" 2>/dev/null | xargs || echo 0)

# Calculate success rate safely
if [ "$TOTAL" -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($SUCCESS/$TOTAL)*100}")
else
    SUCCESS_RATE="0.0"
fi

# Generate summary report
cat > "$SUMMARY_REPORT" <<EOF
======================================================================
MT Fit Batch Processing Summary Report
======================================================================
Run Timestamp: $TIMESTAMP
Started: $(date)
Completed: $(date)

Configuration:
  CSV file: $CSV_FILE
  Output directory: $OUTPUT_DIR/
  MT Fit script: $MT_FIT_SCRIPT

Results:
  Total entries: $TOTAL
  Successful: $SUCCESS
  Failed: $FAILED
  Success rate: ${SUCCESS_RATE}%

======================================================================
Detailed Results:
======================================================================

SUCCESSFUL PROCESSING:
EOF

# Add successful entries
if [ $SUCCESS -gt 0 ]; then
    grep "SUCCESS" "$TEMP_RESULTS" | while IFS='|' read -r basename status n_cilia output_json; do
        echo "  ✓ $basename (n_cilia: $n_cilia) -> $output_json" >> "$SUMMARY_REPORT"
    done
else
    echo "  (none)" >> "$SUMMARY_REPORT"
fi

cat >> "$SUMMARY_REPORT" <<EOF

FAILED PROCESSING:
EOF

# Add failed entries
if [ $FAILED -gt 0 ]; then
    grep "FAILED" "$TEMP_RESULTS" | while IFS='|' read -r basename status reason details; do
        echo "  ✗ $basename - Reason: $reason" >> "$SUMMARY_REPORT"
    done
else
    echo "  (none)" >> "$SUMMARY_REPORT"
fi

cat >> "$SUMMARY_REPORT" <<EOF

======================================================================
EOF

# Clean up temp file
rm "$TEMP_RESULTS"

# Display summary
echo ""
echo "========================================================================"
echo "Processing Complete!"
echo "========================================================================"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
echo ""
echo "Summary report saved to: $SUMMARY_REPORT"
echo "========================================================================"

# Display the summary report
echo ""
cat "$SUMMARY_REPORT"

# Exit with appropriate code
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi