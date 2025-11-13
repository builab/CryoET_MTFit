#!/bin/bash
# Batch script to run mt_fit.py pipeline
# Use this so we can run a common parameter on everything and generate a summarized csv file
# Builab, McGill 2025

# Define the common parameters for the fitting script. Mind the gap at the start & end of quote
FIT_PARAMS=" --angpix 14 --sample_step 83 --min_seed 6 --poly_order 3 --dist_thres 50 --dist_extrapolate 2000 --overlap_thres 100 --neighbor_rad 100 --min_part_per_tube 10 "

# Optional param, make as empty if needed
OPTIONAL_PARAMS=""
#OPTIONAL_PARAMS="--psi_min 30 --psi_max 150"

# Define the output log file to track fitting results
LOG_FILE="mt_fit_summary.csv"

# Create/clear the log file with headers
echo "STAR_FILE,NO_TUBES,NO_PARTICLES,MEDIAN_PSI" > "$LOG_FILE"

# Find all .star files in the current directory and loop through them
for STAR_FILE in *.star; do
    
    # Check if any star files were found
    if [ ! -e "$STAR_FILE" ]; then
        echo "No .star files found. Exiting."
        exit 1
    fi
    
    echo "--- Processing $STAR_FILE ---"
            
    # Execute the fitting script and capture *only* the specific result value
    # We assume mt_fit.py prints the key result (e.g., a goodness-of-fit score)
    # as the LAST LINE of its output.
    
    # Store the full output for potential debugging
    FIT_LOG="${STAR_FILE%.star}_fit.log"
    
    # Run the script, sending standard output and standard error to a log file
    echo "mt_fit.py pipeline $STAR_FILE $FIT_PARAMS --template $STAR_FILE $OPTIONAL_PARAMS" > $FIT_LOG
    mt_fit.py pipeline "$STAR_FILE" $FIT_PARAMS --template "$STAR_FILE" $OPTIONAL_PARAMS >> "$FIT_LOG" 2>&1
    
    # Get the exit status of the previous command (mt_fit.py)
    EXIT_CODE=$?
    
    # Extract the key result value from the log file (e.g., the last line)
    # You will need to adjust this command based on how mt_fit.py prints the result.
    FIT_RESULT_VALUE=$(tail -n 1 "$FIT_LOG")
    
    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS_STATUS="YES"
        echo "  -> Fitting Succeeded. Result: File,Number_of_Tubes,Number_of_particles,MedianPsi"
        echo "                              $FIT_RESULT_VALUE"

        # Log the result to the summary file
        echo "$FIT_RESULT_VALUE" >> "$LOG_FILE"
        
    else
        SUCCESS_STATUS="NO"
        echo "-> Fitting FAILED (Exit Code $EXIT_CODE). Check $FIT_LOG"
        
        # Log the failure
        echo "$STAR_FILE,$SUCCESS_STATUS,N/A" >> "$LOG_FILE"
    fi
    
done

echo "--- Batch Processing Complete ---"
echo "Check $LOG_FILE for summary of mt_fit.py results."