CryoET_MTFit/
├── install.sh			# Installing script
├── source-env.sh		# Source script
├── requirements.txt	# Requirements
├── TODO.md			    # TO DO
├── README.md			
├── structure.md		# Source script
├── examples/			# Test example
│   └── CCDC146C_001_particles.star     # Template matching file
├── utils/
│   ├── __init__.py     # Module exports
│   ├── fit.py          # Initial Curve fitting logic
│   ├── clean.py        # Overlap detection & filtering logic
│   ├── connect.py      # Line connecting logic
│   ├── view.py      	# Visualize star file
│   ├── predict.py      # Generate proper angles from template
│   ├── sort.py         # Sort cilia order
│   └── io.py           # I/O utilities
└── scripts/
    ├── mt_fit.py       # CLI wrapper to fit, clean & connect
    ├── view_star_.py   # CLI wrapper to visualize star file
    ├── visualize_star_angles.py   # Visualize angles from star file
    ├── combine_mtstar2relionwarp.py   # Combine star files to a single file
    ├── add_id_to_warpstar.py   # Add rlnHelicalTubeID, rlnCiliaGroup, rlnRandomSubset to warpstar
    ├── batch_mt_fit.sh   # Batch script for mt_fit.py pipeline
    ├── batch_sort.sh   # Batch script for mt_fit.py sorting
    └── mtfitchimerax.py     # ChimeraX interface of mt_fit.py pipeline only