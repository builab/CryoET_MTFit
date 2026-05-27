import os
import shlex
import subprocess

from chimerax.core.commands import (
    CmdDesc, register,
    FloatArg, IntArg,
    ModelArg,
)
from chimerax.core.settings import Settings

TEMPDIR = "/tmp"


class _MTFitSettings(Settings):
    EXPLICIT_SAVE = {"python_exec": ""}

_settings = None

def _get_settings(session):
    global _settings
    if _settings is None:
        _settings = _MTFitSettings(session, "MTFit")
    return _settings

# Directory where this file (and bundled scripts/utils) lives
_BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))


def _bundled_script() -> str:
    return os.path.join(_BUNDLE_DIR, "mt_fit.py")


def _login_shell_run(cmd_list, **kwargs):
    """Run sourcing the user's shell rc so conda/venv is active."""
    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd_list)
    home = os.path.expanduser("~")
    # Source shell rc files in order — conda init typically lives in .zshrc or .bashrc
    source_cmds = (
        f'[ -f {home}/.zshrc ] && source {home}/.zshrc 2>/dev/null;'
        f'[ -f {home}/.bash_profile ] && source {home}/.bash_profile 2>/dev/null;'
        f'[ -f {home}/.bashrc ] && source {home}/.bashrc 2>/dev/null;'
    )
    shell = os.environ.get("SHELL", "/bin/zsh")
    return subprocess.run([shell, "-c", source_cmds + cmd_str], **kwargs)


def _find_python_with_deps() -> str:
    """
    Find a Python that has all required CryoET packages.
    Checks: login shell python3 → conda envs → venvs in home.
    Returns the path, or None if not found.
    """
    import glob
    required = ["pandas", "numpy", "starfile", "scipy", "copick", "sklearn", "plotly"]
    check = f"import {', '.join(required)}"

    def has_deps(python_path):
        if not os.path.exists(python_path):
            return False
        r = subprocess.run([python_path, "-c", check], capture_output=True, timeout=10)
        return r.returncode == 0

    # 1. Login shell python3 (picks up conda base or active env)
    r = _login_shell_run(["which", "python3"], capture_output=True, text=True)
    shell_python = r.stdout.strip()
    if shell_python and has_deps(shell_python):
        return shell_python

    # 2. Scan all conda environments (user home + system-level for HPC/Linux)
    home = os.path.expanduser("~")
    conda_search_roots = (
        [os.path.join(home, b) for b in
         ["miniconda3", "miniconda", "anaconda3", "anaconda",
          "miniforge3", "miniforge", "mambaforge",
          "opt/miniconda3", "opt/miniforge3", "opt/anaconda3"]]
        + ["/opt/conda", "/opt/miniconda3", "/opt/anaconda3",
           "/opt/miniforge3", "/usr/local/miniconda3", "/usr/local/anaconda3"]
    )
    for root in conda_search_roots:
        # Check base env
        base_py = os.path.join(root, "bin", "python3")
        if has_deps(base_py):
            return base_py
        # Check named envs
        for py in glob.glob(os.path.join(root, "envs", "*/bin/python3")):
            if has_deps(py):
                return py

    # 3. Scan .venv directories up to 4 levels deep under home
    for pattern in [
        os.path.join(home, ".venv", "bin", "python3"),
        os.path.join(home, "*", ".venv", "bin", "python3"),
        os.path.join(home, "*", "*", ".venv", "bin", "python3"),
        os.path.join(home, "*", "*", "*", ".venv", "bin", "python3"),
        os.path.join(home, "*", "*", "*", "*", ".venv", "bin", "python3"),
    ]:
        for py in glob.glob(pattern):
            if has_deps(py):
                return py

    return None


def _check_dependencies(session):
    """Return cached Python path, or scan for one with all deps (and cache it)."""
    s = _get_settings(session)

    # Use cached path if it still works
    if s.python_exec and os.path.exists(s.python_exec):
        required = ["pandas", "numpy", "starfile", "scipy", "copick", "sklearn", "plotly"]
        r = subprocess.run([s.python_exec, "-c", f"import {', '.join(required)}"],
                           capture_output=True, timeout=10)
        if r.returncode == 0:
            return s.python_exec
        # Cached path no longer valid — rescan
        s.python_exec = ""

    session.logger.info("MTFit: scanning for Python environment with CryoET packages...")
    python = _find_python_with_deps()
    if python is None:
        session.logger.error(
            "MTFit: could not find a Python environment with all required packages.\n"
            "Install them into your active environment with:\n"
            "  pip install pandas numpy starfile scipy copick scikit-learn plotly\n"
            "Then restart ChimeraX."
        )
        return None

    session.logger.info(f"MTFit: found Python at {python} (cached for future runs)")
    s.python_exec = python
    s.save()
    return python


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

    mt_fit_script = _bundled_script()

    if not os.path.exists(mt_fit_script):
        session.logger.error(f"Bundled mt_fit.py not found: {mt_fit_script}")
        return

    python_exec = _check_dependencies(session)
    if python_exec is None:
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

    # --- 2. Run pipeline via login shell (picks up user's conda/venv) ---
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

    # utils/ is bundled alongside this file — add to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = _BUNDLE_DIR + os.pathsep + env.get("PYTHONPATH", "")

    session.logger.info(f"Running pipeline with {python_exec}...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

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
