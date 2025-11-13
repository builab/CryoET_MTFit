#!/bin/bash
# Usage: ./run_tomo_extract_refine.sh input.star

STAR_FILE="$1"

if [ -z "$STAR_FILE" ]; then
    echo "Usage: $0 input.star"
    exit 1
fi

# Get unique list of .tomostar files
TOMOSTARS=$(grep '\.tomostar' "$STAR_FILE" | \
    awk '{for(i=1;i<=NF;i++) if ($i ~ /\.tomostar$/) print $i}' | \
    sort -u)

# Loop through each tomostar
for TOMO in $TOMOSTARS; do
    echo "Processing $TOMO ..."

    # Define temporary output names
    TMP_STAR="tmp_${TOMO%.tomostar}.star"
    OUT_MRC="out_${TOMO%.tomostar}.mrc"

    # Run extract script
    extract_tomo_star.py --input "$STAR_FILE" --tomo "$TOMO" --output "$TMP_STAR"

    # Run RELION refine
    relion_refine.py --i "$TMP_STAR" --o "$OUT_MRC"
done

echo "✅ All tomostars processed successfully."
