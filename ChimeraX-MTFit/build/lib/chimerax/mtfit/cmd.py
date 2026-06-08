import os
import subprocess
import sys
import sysconfig

from chimerax.core.commands import (
    CmdDesc, register,
    FloatArg, IntArg, EnumOf,
    ModelArg,
)

TEMPDIR = "/tmp"

# Steps that can be run individually, in pipeline order
STEPS = ("pipeline", "fit", "clean", "connect", "predict")

_OUTPUT_SUFFIX = {
    "fit":      "_fitted",
    "clean":    "_cleaned",
    "connect":  "_connected",
    "predict":  "_predicted",
    "pipeline": "_processed",
}

# Directory where this file (and bundled scripts/utils) lives
_BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))


def _bundled_script() -> str:
    return os.path.join(_BUNDLE_DIR, "mt_fit.py")


def _chimerax_python() -> str:
    """Return the actual Python interpreter — sys.executable in ChimeraX is the app launcher."""
    bin_dir = sysconfig.get_config_var("BINDIR")
    versioned = os.path.join(bin_dir, f"python{sys.version_info.major}.{sys.version_info.minor}")
    if os.path.exists(versioned):
        return versioned
    plain = os.path.join(bin_dir, "python3")
    if os.path.exists(plain):
        return plain
    return sys.executable


# ---------------------------------------------------------------------------
# mtfit — run the full pipeline
# ---------------------------------------------------------------------------

def mtfit(session,
          model,
          step="pipeline",
          voxel_size=14.0,
          sample_step=82.0,
          min_seed=6,
          poly=3,
          clean_dist_thres=50.0,
          dist_extrapolate=2000.0,
          overlap_thres=100.0,
          min_part_per_tube=5,
          neighbor_rad=100.0):
    """Run the full MT fitting pipeline, or just a single step of it, on a particle list model."""

    mt_fit_script = _bundled_script()
    if not os.path.exists(mt_fit_script):
        session.logger.error(f"Bundled mt_fit.py not found: {mt_fit_script}")
        return

    # --- 1. Save particle list to temp file ---
    input_star_file = model.name
    if not input_star_file:
        session.logger.error("Could not resolve star file name from model.")
        return

    os.makedirs(TEMPDIR, exist_ok=True)
    tmp_star = os.path.join(TEMPDIR, os.path.basename(input_star_file))
    session.logger.info(f"Saving particle list to {tmp_star}")

    from chimerax.core.commands import run
    run(session, f'save "{tmp_star}" partlist #{".".join(str(i) for i in model.id)}')

    # --- 2. Build the subcommand for the requested step (or full pipeline) ---
    cmd = [_chimerax_python(), mt_fit_script, step, tmp_star, "--angpix", str(voxel_size)]

    if step in ("fit", "pipeline"):
        cmd += ["--sample_step", str(sample_step),
                "--min_seed", str(min_seed),
                "--poly_order", str(poly)]
    if step in ("clean", "pipeline"):
        cmd += ["--dist_thres", str(clean_dist_thres)]
    if step in ("connect", "pipeline"):
        cmd += ["--dist_extrapolate", str(dist_extrapolate),
                "--overlap_thres", str(overlap_thres),
                "--min_part_per_tube", str(min_part_per_tube)]
    if step in ("predict", "pipeline"):
        cmd += ["--template", tmp_star,
                "--neighbor_rad", str(neighbor_rad)]

    # utils/ is bundled alongside this file — add to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = _BUNDLE_DIR + os.pathsep + env.get("PYTHONPATH", "")

    session.logger.info(f"Running '{step}'...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        session.logger.error(f"'{step}' failed:\n{result.stderr}")
        _cleanup([tmp_star])
        return

    # --- 3. Load result ---
    base = os.path.splitext(os.path.basename(tmp_star))[0]
    output_star = os.path.join(TEMPDIR, f"{base}{_OUTPUT_SUFFIX[step]}.star")

    if os.path.exists(output_star):
        session.logger.info(f"Loading result: {output_star}")
        run(session, f'open "{output_star}"')
    else:
        session.logger.warning(f"Output file not found: {output_star}")

    _cleanup([tmp_star, output_star])


def _cleanup(files):
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

def register_commands(logger):
    register("mtfit", CmdDesc(
        required=[("model", ModelArg)],
        keyword=[
            ("step",              EnumOf(STEPS)),
            ("voxel_size",        FloatArg),
            ("sample_step",       FloatArg),
            ("min_seed",          IntArg),
            ("poly",              IntArg),
            ("clean_dist_thres",  FloatArg),
            ("dist_extrapolate",  FloatArg),
            ("overlap_thres",     FloatArg),
            ("min_part_per_tube", IntArg),
            ("neighbor_rad",      FloatArg),
        ],
        synopsis="Run the full MT fitting pipeline, or a single step (fit/clean/connect/predict), on a particle list model",
    ), mtfit)
