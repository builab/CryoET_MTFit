from chimerax.core.toolshed import BundleAPI


def _ensure_dependencies():
    """Install any missing PyPI dependencies into ChimeraX's Python (runs once)."""
    import importlib.util, subprocess, sys, sysconfig, os

    needed = {
        "pandas":      "pandas>=1.5,<3",
        "starfile":    "starfile>=0.5",
        "copick":      "copick>=1.0",
        "sklearn":     "scikit-learn>=1.0",
        "plotly":      "plotly>=5.0",
        "scipy":       "scipy>=1.7",
        "matplotlib":  "matplotlib>=3.5",
    }

    missing = [pip_spec for import_name, pip_spec in needed.items()
               if importlib.util.find_spec(import_name) is None]
    if not missing:
        return

    # Find ChimeraX's actual Python interpreter (sys.executable is the app launcher).
    # sysconfig BINDIR is baked at build time and may not exist on Linux system installs
    # (e.g. /home/runner/work/... CI path) — fall back to launcher's own directory.
    bin_dir = sysconfig.get_config_var("BINDIR")
    if not os.path.isdir(bin_dir):
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    python = os.path.join(bin_dir, f"python{sys.version_info.major}.{sys.version_info.minor}")
    if not os.path.exists(python):
        python = os.path.join(bin_dir, "python3")

    print(f"MTFit: installing missing packages: {missing}")
    subprocess.check_call([python, "-m", "pip", "install"] + missing)


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
