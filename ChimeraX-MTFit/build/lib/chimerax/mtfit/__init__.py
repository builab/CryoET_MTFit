from chimerax.core.toolshed import BundleAPI


def _ensure_dependencies():
    """Install any missing PyPI dependencies into ChimeraX's Python (runs once)."""
    import subprocess, sys, sysconfig, os

    needed = {
        "pandas":      "pandas>=1.5,<3",
        "starfile":    "starfile>=0.5",
        "copick":      "copick>=1.0",
        "sklearn":     "scikit-learn>=1.0",
        "plotly":      "plotly>=5.0",
        "scipy":       "scipy>=1.7",
        "matplotlib":  "matplotlib>=3.5",
    }

    # Find ChimeraX's actual Python interpreter (sys.executable is the app launcher).
    # sysconfig BINDIR is baked at build time and may not exist on Linux system installs
    # (e.g. /home/runner/work/... CI path) — fall back to launcher's own directory.
    bin_dir = sysconfig.get_config_var("BINDIR")
    if not os.path.isdir(bin_dir):
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    python = os.path.join(bin_dir, f"python{sys.version_info.major}.{sys.version_info.minor}")
    if not os.path.exists(python):
        python = os.path.join(bin_dir, "python3")

    # Check importability using the subprocess Python — not find_spec(), which checks
    # the current ChimeraX process and can be fooled by an active conda environment
    # that the subprocess Python cannot access.
    missing = []
    for import_name, pip_spec in needed.items():
        result = subprocess.run(
            [python, "-c", f"import {import_name}"],
            capture_output=True
        )
        if result.returncode != 0:
            missing.append(pip_spec)

    if not missing:
        return

    # Install one at a time -- a package that fails to build (e.g. copick pulling in
    # cryptography, which needs a Rust/OpenSSL toolchain this system may not have)
    # must not block the others from installing via one combined pip command.
    for pip_spec in missing:
        print(f"MTFit: installing missing package: {pip_spec}")
        try:
            subprocess.check_call([python, "-m", "pip", "install", pip_spec])
        except subprocess.CalledProcessError as e:
            print(f"MTFit: could not install {pip_spec} ({e}); "
                  f"features needing it will be unavailable.")


_ensure_dependencies()


class _MTFitBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        if ci.name == "mtfit":
            cmd.register_commands(logger)

    @staticmethod
    def start_tool(session, bi, ti):
        from . import tool
        return tool.MTFitTool(session, ti.name)

    @staticmethod
    def get_class(class_name):
        if class_name == "MTFitTool":
            from . import tool
            return tool.MTFitTool
        return None


bundle_api = _MTFitBundle()
