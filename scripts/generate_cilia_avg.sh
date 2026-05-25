#!/bin/bash
# Usage: ./run_tomo_extract_refine.sh input.star

STAR_FILE="$1"
OUT_DIR="avg"
TOMO_STAR="combine_10Apx_tomograms.star"
REF="templates/doublet_8nm_10.00Apx.mrc"

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

    # Run extract script for each tomogram
    extract_particles_from_star.py --i "$STAR_FILE" --rlnTomoName "$TOMO" --o "$TMP_STAR"

    # Run RELION refine to reconstruct the average map from each tomogram
    `which relion_refine` --o "$OUT_DIR"/tmp/run --i "$TMP_STAR" --tomograms $TOMO_STAR --ref $REF --trust_ref_size --ini_high 30 --pad 1  --ctf --iter 1 --tau2_fudge 1 --K 1 --skip_align --norm --scale  --j 10
    `which relion_refine` --o avg/avg --i Select/job011/particles.star --tomograms combine_10Apx_tomograms.star --ref Refine3D/job008/run_class001.mrc --trust_ref_size --ini_high 35 --dont_combine_weights_via_disc --pool 3 --pad 2  --ctf --iter 1 --tau2_fudge 1 --particle_diameter 900 --K 1 --flatten_solvent --zero_mask --skip_align  --sym C1 --norm --scale  --j 10
done

echo "✅ All tomostars processed successfully."
