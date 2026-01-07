- Test with data
- Restrict polynomial 2 or 3 to relatively straight line (Low priority)
- Fit with spline line for more accurate? (Future version) (Low priority)
- What about fit on existing data with rlnHelicalTubeID for segmentation & also correction
- Generate json or specific output line so we can use a batch function for processing.
- For the tip, perhaps a directional filter (the majority) will work better

- Average: 
python to write star file of each cilia
classification of star 

Less priorty
- Do a basic rlnAnglePsi and rlnAngleRot for microtubule prediction (no template).
- Later on, need to check for polarity (does the 14PF template get right 13PF polarity)
- For microtubule, perhaps a local smoothing (every 20 particles) is better

v0.9.2
 - Fix bugs to longer tubes
 - Fix fit.py, predict.py to output right rlnAngleTilt and rlnAnglePsi
 - Implement filter by psi in clean.py to remove horizontal tubes.
 
v0.9.3
 - Make visualize_star_angles.py to see the smoothness of angles to improve predict.py using outlier detection.
 - Implement smooth_angles for predict.py and work so well
 
v0.9.4
 - Finish combine_mtstar.py to combine the output file to a single star file used for Warp export

v0.9.6
 - Fix a lots of serious bug regarding resampling and imagePixelSize
 - update visualize_star_angles.py
 - Better name sanitization.
 
v0.9.7
 - rename & refactor code

v0.9.8
 - improve accuracy of resample using better integration_step
 - fix the error "x & y not the same length"
 - implement direction_dev, which is quite good.
 
v0.9.9
 - implement doublet sorting
 - add_id_to_warpstar.py for better processing
 - combine_mtstarfiles.py for combine everything for warp extraction
 
v.0.9.10
 - Implement json export
 - Implement manual sort
 - Implement batch_mt_fit.sh

v.0.9.11
 - Add extract_particles_from_star.py to extract rlnTomoName or rlnHelicalTubeID
 - Generate average of each cilia (Not tested)
 - Add file name to view_star.py, rename to view_single_star.py
 - Fix bug when only 1 tube in cleaning
 - Fix bug when rlnLCCmax not present (none pytom_match_pick output)
 - Improve the line connecting
 - Overlay and scoring calculating for view_single_star.py (utils/scoring.py)
 
v.0.9.12
 - Fix sort with skilearn version
