import os
import subprocess

from chimerax.core.commands import (
    CmdDesc, register,
    FloatArg, IntArg,
    ModelArg, StringArg,
)
from chimerax.core.settings import Settings

TEMPDIR = "/tmp"


# ---------------------------------------------------------------------------
# Persistent settings — stored in ChimeraX preferences automatically
# ---------------------------------------------------------------------------

class _MTFitSettings(Settings):
    EXPLICIT_SAVE = {
        "project_root": "",
    }


_settings = None


def _get_settings(session):
    global _settings
    if _settings is None:
        _settings = _MTFitSettings(session, "MTFit")
    return _settings


def _get_project_root(session):
    root = _get_settings(session).project_root
    if not root:
        raise ValueError(
            "MTFit project path is not set. Run this once to configure it:\n"
            "  mtfit setpath /path/to/CryoET_MTFit"
        )
    return root


# ---------------------------------------------------------------------------
# mtfit setpath — configure project root (run once per machine)
# ---------------------------------------------------------------------------

def mtfit_setpath(session, project_root):
    """Store the CryoET_MTFit project directory in ChimeraX preferences."""
    project_root = os.path.expanduser(project_root)
    if not os.path.isdir(project_root):
        session.logger.error(f"Directory not found: {project_root}")
        return

    python = os.path.join(project_root, ".venv", "bin", "python3")
    script = os.path.join(project_root, "scripts", "mt_fit.py")
    if not os.path.exists(python):
        session.logger.warning(f"venv not found at expected path: {python}")
    if not os.path.exists(script):
        session.logger.warning(f"mt_fit.py not found at expected path: {script}")

    s = _get_settings(session)
    s.project_root = project_root
    s.save()
    session.logger.info(f"MTFit project path saved: {project_root}")


# ---------------------------------------------------------------------------
# mtfit — run the full pipeline
# ---------------------------------------------------------------------------

def mtfit(session,
          model,
          voxel_size=14.0,
          sample_step=82.0,
          min_seed=6,
          poly=3,
          clean_dist_thres=50.0,
          dist_extrapolate=2000.0,
          overlap_thres=100.0,
          min_part_per_tube=5,
          neighbor_rad=100.0):
    """Run the full MT fitting pipeline on a particle list model."""

    try:
        project_root = _get_project_root(session)
    except ValueError as e:
        session.logger.error(str(e))
        return

    python_exec = os.path.join(project_root, ".venv", "bin", "python3")
    mt_fit_script = os.path.join(project_root, "scripts", "mt_fit.py")

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

    # --- 2. Run pipeline subprocess ---
    cmd = [
        python_exec, mt_fit_script, "pipeline", tmp_star,
        "--angpix",            str(voxel_size),
        "--sample_step",       str(sample_step),
        "--min_seed",          str(min_seed),
        "--poly_order",        str(poly),
        "--dist_thres",        str(clean_dist_thres),
        "--dist_extrapolate",  str(dist_extrapolate),
        "--overlap_thres",     str(overlap_thres),
        "--min_part_per_tube", str(min_part_per_tube),
        "--neighbor_rad",      str(neighbor_rad),
        "--template",          tmp_star,
    ]

    session.logger.info(f"Running pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        session.logger.error(f"Pipeline failed:\n{result.stderr}")
        _cleanup([tmp_star])
        return

    # --- 3. Load result ---
    base = os.path.splitext(os.path.basename(tmp_star))[0]
    output_star = os.path.join(TEMPDIR, f"{base}_processed.star")

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
    # mtfit <model> [options]
    register("mtfit", CmdDesc(
        required=[("model", ModelArg)],
        keyword=[
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
        synopsis="Run full MT fitting pipeline on a particle list model",
    ), mtfit)

    # mtfit setpath <project_root>
    register("mtfit setpath", CmdDesc(
        required=[("project_root", StringArg)],
        synopsis="Set the CryoET_MTFit project directory (run once per machine)",
    ), mtfit_setpath)
