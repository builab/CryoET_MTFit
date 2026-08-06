import json
import os
import queue
import subprocess
import sys
import sysconfig
import threading

import numpy as np

from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
from chimerax.core.selection import SELECTION_CHANGED

_BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

_OUTPUT_SUFFIX = {
    "fit":         "_fitted",
    "clean":       "_cleaned",
    "connect":     "_connected",
    "predict":     "_predicted",
    "pipeline":    "_processed",
    "pipeline_mt": "_processed",
    "twist":       "_twisted",
}

_STEPS = ("pipeline", "fit", "clean", "connect", "predict")


def _chimerax_python():
    bin_dir = sysconfig.get_config_var("BINDIR")
    if not os.path.isdir(bin_dir):
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    versioned = os.path.join(bin_dir, f"python{sys.version_info.major}.{sys.version_info.minor}")
    if os.path.exists(versioned):
        return versioned
    plain = os.path.join(bin_dir, "python3")
    if os.path.exists(plain):
        return plain
    return sys.executable


def _run_one(filepath, params, output_path):
    """Run mt_fit.py on a single file. Returns (returncode, stderr)."""
    step = params["step"]
    mt_fit = os.path.join(_BUNDLE_DIR, "mt_fit.py")
    cmd = [_chimerax_python(), mt_fit, step, filepath,
           "--angpix", str(params["voxel_size"]),
           "-o", output_path]
    if step in ("fit", "pipeline", "pipeline_mt"):
        cmd += ["--sample_step", str(params["sample_step"]),
                "--min_seed",    str(params["min_seed"]),
                "--poly_order",  str(params["poly"])]
    if step in ("clean", "pipeline", "pipeline_mt"):
        cmd += ["--dist_thres",    str(params["clean_dist_thres"]),
                "--max_curvature", str(params["max_curvature"])]
    if step in ("connect", "pipeline", "pipeline_mt"):
        cmd += ["--dist_extrapolate",  str(params["dist_extrapolate"]),
                "--overlap_thres",     str(params["overlap_thres"]),
                "--min_part_per_tube", str(params["min_part_per_tube"])]
    if step in ("predict", "pipeline"):
        cmd += ["--template",     filepath,
                "--neighbor_rad", str(params["neighbor_rad"]),
                "--rot_smooth_factor", str(params["rot_smooth_factor"]),
                "--rot_smooth_window", str(params["rot_smooth_window"])]
    if step == "twist":
        cmd += ["--twist_angle", str(params["twist_angle"])]
        if params.get("twist_template"):
            cmd += ["--template", params["twist_template"],
                    "--neighbor_rad", str(params["neighbor_rad"])]
    if step == "pipeline_mt":
        # Starts from the raw picks, so the input file is already its own
        # polarity reference -- no separate template field needed here.
        cmd += ["--twist_angle", str(params["twist_angle"]),
                "--template", filepath,
                "--neighbor_rad", str(params["neighbor_rad"])]
    env = os.environ.copy()
    env["PYTHONPATH"] = _BUNDLE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stderr


def _extract_stats(star_path):
    """Read a star file and return a dict of summary statistics."""
    stats = {"particles": 0, "tubes": 0, "parts_per_tube": "", "mean_psi": "", "mean_tilt": ""}
    try:
        import starfile
        df = starfile.read(star_path)
        if isinstance(df, dict):
            df = df.get("particles", next(iter(df.values())))
        stats["particles"] = len(df)
        if "rlnHelicalTubeID" in df.columns:
            stats["tubes"] = int(df["rlnHelicalTubeID"].nunique())
            if stats["tubes"] > 0:
                ppt = len(df) / stats["tubes"]
                stats["parts_per_tube"] = f"{ppt:.1f}"
        if "rlnAnglePsi" in df.columns:
            stats["mean_psi"] = f"{df['rlnAnglePsi'].mean():.1f}"
        if "rlnAngleTilt" in df.columns:
            stats["mean_tilt"] = f"{df['rlnAngleTilt'].mean():.1f}"
    except Exception:
        pass
    return stats


class MTFitTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = "help:user/tools/MTFit.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self._build_ui()
        self.tool_window.manage(placement=None)

        self._add_handler    = session.triggers.add_handler(ADD_MODELS,    self._on_models_changed)
        self._remove_handler = session.triggers.add_handler(REMOVE_MODELS, self._on_models_changed)

        self._result_queue = queue.Queue()
        self._running = False
        self._results = []   # list of result dicts for CSV export
        self._extra_model_ids = []  # models opened outside the results table (e.g. Combine)

    def delete(self):
        self.session.triggers.remove_handler(self._add_handler)
        self.session.triggers.remove_handler(self._remove_handler)
        self.session.triggers.remove_handler(self._selection_handler)
        super().delete()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from Qt.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
            QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
            QComboBox, QFormLayout, QSizePolicy, QGroupBox,
            QLineEdit, QTableWidget, QTableWidgetItem,
            QAbstractItemView, QHeaderView, QFileDialog,
        )
        from Qt.QtCore import Qt, QTimer

        parent = self.tool_window.ui_area
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ---- Model selector ----
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Particle list:"))
        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._model_combo.currentIndexChanged.connect(self._refresh_join_tubes)
        model_row.addWidget(self._model_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh model list")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(refresh_btn)
        main_layout.addLayout(model_row)

        # ---- Step selector ----
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Run:"))
        self._step_combo = QComboBox()
        self._step_combo.addItem("Full pipeline, cilia (Fit → Clean → Connect → Predict)", userData="pipeline")
        self._step_combo.addItem("Full pipeline, MT (Fit → Clean → Connect → Twist)", userData="pipeline_mt")
        self._step_combo.addItem("1. Fit only",                userData="fit")
        self._step_combo.addItem("2. Clean only",              userData="clean")
        self._step_combo.addItem("3. Connect only",            userData="connect")
        self._step_combo.addItem("4. Predict only (cilia)",    userData="predict")
        self._step_combo.addItem("5. Twist only (MT)",         userData="twist")
        self._step_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._step_combo.currentIndexChanged.connect(self._on_step_changed)
        step_row.addWidget(self._step_combo)
        main_layout.addLayout(step_row)

        # ---- Parameter tabs ----
        tabs = QTabWidget()

        basic_tab = QWidget()
        basic_form = QFormLayout(basic_tab)
        basic_form.setContentsMargins(8, 8, 8, 8)
        self._voxel_size  = self._float_spin(1.0,   100.0,  14.0, 1)
        self._sample_step = self._float_spin(1.0,   500.0,  82.0, 1)
        self._min_seed    = self._int_spin(2, 50, 6)
        self._poly        = self._int_spin(1, 5,  3)
        basic_form.addRow("Voxel size (Å/px):", self._voxel_size)
        basic_form.addRow("Sample step (Å):",   self._sample_step)
        basic_form.addRow("Min seed:",           self._min_seed)
        basic_form.addRow("Polynomial order:",   self._poly)
        tabs.addTab(basic_tab, "Basic")

        cc_tab = QWidget()
        cc_form = QFormLayout(cc_tab)
        cc_form.setContentsMargins(8, 8, 8, 8)
        self._clean_dist_thres  = self._float_spin(1.0,   500.0,   50.0,  1)
        self._max_curvature     = self._float_spin(0.0,   180.0,    0.0,  1)
        self._max_curvature.setToolTip(
            "Max bend angle between consecutive segments (degrees).\n"
            "Tubes exceeding this are removed. 0 = disabled."
        )
        self._dist_extrapolate  = self._float_spin(1.0, 10000.0, 2000.0, 10)
        self._overlap_thres     = self._float_spin(1.0,   500.0,  100.0,  1)
        self._min_part_per_tube = self._int_spin(1, 50, 5)
        cc_form.addRow("Clean dist threshold (Å):", self._clean_dist_thres)
        cc_form.addRow("Max curvature (°, 0=off):", self._max_curvature)
        cc_form.addRow("Extrapolate distance (Å):", self._dist_extrapolate)
        cc_form.addRow("Overlap threshold (Å):",    self._overlap_thres)
        cc_form.addRow("Min particles per tube:",   self._min_part_per_tube)
        tabs.addTab(cc_tab, "Clean / Connect")

        pred_tab = QWidget()
        pred_form = QFormLayout(pred_tab)
        pred_form.setContentsMargins(8, 8, 8, 8)
        self._neighbor_rad  = self._float_spin(1.0,  1000.0, 100.0, 1)
        self._twist_angle   = self._float_spin(-180.0, 180.0,  0.0, 0.1)
        self._rot_smooth_factor = self._float_spin(0.0, 1.0, 0.0, 0.1, "")
        self._rot_smooth_factor.setToolTip(
            "Blend weight for local Rot angle smoothing (0 = off, 1 = fully\n"
            "replaced by the local average). Reduces small residual wobble,\n"
            "e.g. right at a Connect join, without needing an outlier to trigger."
        )
        self._rot_smooth_window = self._int_spin(1, 51, 5)
        self._rot_smooth_window.setToolTip("Number of neighboring particles to average over for Rot smoothing.")
        pred_form.addRow("Neighbor radius (Å):", self._neighbor_rad)
        pred_form.addRow("Twist angle/particle (°):", self._twist_angle)

        twist_template_row = QHBoxLayout()
        self._twist_template_edit = QLineEdit()
        self._twist_template_edit.setPlaceholderText("Optional: original raw picks, for polarity")
        self._twist_template_browse_btn = QPushButton("Browse…")
        self._twist_template_browse_btn.setFixedWidth(65)
        self._twist_template_browse_btn.clicked.connect(self._browse_twist_template)
        twist_template_row.addWidget(self._twist_template_edit)
        twist_template_row.addWidget(self._twist_template_browse_btn)
        self._twist_template_label = QLabel("Polarity reference (optional):")
        self._twist_template_label.setToolTip(
            "Original raw (pre-Fit) picks. Used only to determine each tube's real\n"
            "polarity so the twist increment is applied with consistent handedness\n"
            "across tubes -- not saved or kept as output. If left empty, twist falls\n"
            "back to row order only, which is self-consistent per tube but not\n"
            "guaranteed consistent between different tubes."
        )
        pred_form.addRow(self._twist_template_label, twist_template_row)

        pred_form.addRow("Rot smoothing factor (0-1):", self._rot_smooth_factor)
        pred_form.addRow("Rot smoothing window:", self._rot_smooth_window)
        tabs.addTab(pred_tab, "Predict / Twist")

        main_layout.addWidget(tabs)

        # ---- Batch & Parameters JSON section ----
        batch_group = QGroupBox("Batch & Parameters")
        batch_form = QFormLayout(batch_group)
        batch_form.setContentsMargins(8, 8, 8, 8)

        # Output folder
        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setPlaceholderText("Where to save processed files (batch)")
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.setFixedWidth(65)
        browse_out_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self._output_folder_edit)
        out_row.addWidget(browse_out_btn)
        batch_form.addRow("Output folder:", out_row)

        # Batch input folder
        folder_row = QHBoxLayout()
        self._batch_folder_edit = QLineEdit()
        self._batch_folder_edit.setPlaceholderText("Folder of .star files for batch run")
        browse_folder_btn = QPushButton("Browse…")
        browse_folder_btn.setFixedWidth(65)
        browse_folder_btn.clicked.connect(self._browse_batch_folder)
        folder_row.addWidget(self._batch_folder_edit)
        folder_row.addWidget(browse_folder_btn)
        batch_form.addRow("Batch folder:", folder_row)

        json_row = QHBoxLayout()
        self._json_edit = QLineEdit()
        self._json_edit.setPlaceholderText("Auto-saved on each run")
        load_json_btn = QPushButton("Load…")
        load_json_btn.setFixedWidth(50)
        load_json_btn.clicked.connect(self._load_json)
        save_json_btn = QPushButton("Save…")
        save_json_btn.setFixedWidth(50)
        save_json_btn.clicked.connect(self._save_json_as)
        json_row.addWidget(self._json_edit)
        json_row.addWidget(load_json_btn)
        json_row.addWidget(save_json_btn)
        batch_form.addRow("Params JSON:", json_row)

        main_layout.addWidget(batch_group)

        # ---- Two run buttons ----
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.setToolTip("Run on the selected particle list model")
        self._run_btn.clicked.connect(self._run_single)
        self._run_batch_btn = QPushButton("Run Batch")
        self._run_batch_btn.setMinimumHeight(32)
        self._run_batch_btn.setToolTip("Run on all .star files in the Batch folder")
        self._run_batch_btn.clicked.connect(self._run_batch_clicked)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._run_batch_btn)
        main_layout.addLayout(run_row)

        # ---- Status label ----
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self._status)

        # ---- Manual Tube Join ----
        join_group = QGroupBox("Manual Tube Join")
        join_layout = QVBoxLayout(join_group)

        join_row = QHBoxLayout()
        join_row.addWidget(QLabel("Tube A:"))
        self._join_tube_a = QComboBox()
        self._join_tube_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._join_tube_a.currentIndexChanged.connect(self._highlight_join_selection)
        join_row.addWidget(self._join_tube_a)
        join_row.addWidget(QLabel("Tube B:"))
        self._join_tube_b = QComboBox()
        self._join_tube_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._join_tube_b.currentIndexChanged.connect(self._highlight_join_selection)
        join_row.addWidget(self._join_tube_b)
        refresh_join_btn = QPushButton("↻")
        refresh_join_btn.setFixedWidth(28)
        refresh_join_btn.setToolTip("Refresh tube list from the selected particle list")
        refresh_join_btn.clicked.connect(self._refresh_join_tubes)
        join_row.addWidget(refresh_join_btn)
        join_layout.addLayout(join_row)

        join_hint = QLabel(
            "Pick two tubes above, or select particles in the 3D view — selecting "
            "any particle of a tube auto-selects the whole tube and fills Tube A/B."
        )
        join_hint.setWordWrap(True)
        join_layout.addWidget(join_hint)

        self._join_btn = QPushButton("Join Selected Tubes")
        self._join_btn.clicked.connect(self._join_selected_tubes)
        join_layout.addWidget(self._join_btn)

        main_layout.addWidget(join_group)

        self._join_model = None
        self._join_pid_to_tube = {}
        self._join_tube_to_pids = {}
        self._join_syncing = False
        self._join_next_slot = 'a'
        self._selection_handler = self.session.triggers.add_handler(
            SELECTION_CHANGED, self._on_chimerax_selection_changed)

        # ---- Results table ----
        self._results_group = QGroupBox(
            "Results  (⚠ Problematic = 0 tubes found OR <30% particles retained)"
        )
        results_layout = QVBoxLayout(self._results_group)

        csv_row = QHBoxLayout()
        self._auto_csv_lbl = QLabel("")
        csv_row.addWidget(self._auto_csv_lbl)
        csv_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear all rows and close any models MTFit opened for them")
        clear_btn.clicked.connect(self._clear_results)
        csv_row.addWidget(clear_btn)
        combine_btn = QPushButton("Combine Selected…")
        combine_btn.setToolTip("Combine the checked output files into one STAR file")
        combine_btn.clicked.connect(self._combine_selected)
        csv_row.addWidget(combine_btn)
        save_csv_btn = QPushButton("Save CSV…")
        save_csv_btn.setToolTip("Save an additional copy of the results table")
        save_csv_btn.clicked.connect(self._save_csv)
        csv_row.addWidget(save_csv_btn)
        results_layout.addLayout(csv_row)

        # col 0 = checkbox, 1 = File, 2 = Status, 3-9 = stats / action
        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            ["", "File", "Status", "In ✦", "Out ✦", "Tubes", "Pts/Tube", "Mean Psi°", "Open / Save / Remove"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (2, 3, 4, 5, 6, 7, 8):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(150)
        results_layout.addWidget(self._table)

        self._results_group.setVisible(False)
        main_layout.addWidget(self._results_group)

        # Poll timer for batch background results
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_results)

        self._refresh_models()
        self._refresh_join_tubes()
        self._on_step_changed()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _float_spin(self, lo, hi, default, step, suffix=""):
        from Qt.QtWidgets import QDoubleSpinBox
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(f" {suffix}")
        return w

    def _int_spin(self, lo, hi, default):
        from Qt.QtWidgets import QSpinBox
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        return w

    def _browse_output_folder(self):
        from Qt.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(None, "Select output folder")
        if folder:
            self._output_folder_edit.setText(folder)

    def _browse_twist_template(self):
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, "Select original raw picks (polarity reference)", "", "STAR files (*.star)"
        )
        if path:
            self._twist_template_edit.setText(path)

    def _on_models_changed(self, trigger_name, changes):
        self._refresh_models()

    def _on_step_changed(self):
        """Twist angle only ever reaches the CLI when Step is 'Twist only' or
        'Full pipeline, MT' -- the cilia pipeline and other steps never run a
        twist stage. Disable the field otherwise so it can't look like it's
        being applied when it silently isn't."""
        step = self._step_combo.itemData(self._step_combo.currentIndex())
        uses_twist = step in ("twist", "pipeline_mt")
        self._twist_angle.setEnabled(uses_twist)

        # The polarity-reference field is only relevant for standalone Twist --
        # the MT pipeline starts from the raw picks itself, so it automatically
        # uses its own input as the polarity reference, no field needed.
        is_standalone_twist = step == "twist"
        self._twist_template_edit.setEnabled(is_standalone_twist)
        self._twist_template_label.setEnabled(is_standalone_twist)
        self._twist_template_browse_btn.setEnabled(is_standalone_twist)

        if uses_twist:
            self._twist_angle.setToolTip(
                "Degrees added to rlnAngleRot per particle along each tube."
            )
        else:
            self._twist_angle.setToolTip(
                "Only applied when Step is 'Twist only (MT)' or 'Full pipeline, "
                "MT' -- other steps never run a twist stage, so this value is "
                "ignored while a different step is selected."
            )

    def _first_model(self, opened):
        """Normalize the return value of an 'open' command run() call to a single model."""
        if isinstance(opened, (list, tuple)):
            opened = opened[0] if opened else None
        return opened if opened is not None and hasattr(opened, 'id') else None

    def _close_model(self, model_id):
        if not model_id:
            return
        try:
            run(self.session, f'close {model_id}')
        except Exception as e:
            self.session.logger.warning(f"MTFit: could not close model {model_id}: {e}")

    def _refresh_models(self):
        self._model_combo.clear()
        for m in self.session.models:
            if hasattr(m, 'name') and m.name.endswith('.star'):
                id_str = '#' + '.'.join(str(i) for i in m.id)
                self._model_combo.addItem(f"{id_str} — {m.name}", userData=id_str)
        if self._model_combo.count() == 0:
            self._model_combo.addItem("No particle lists found")

    def _get_model_by_id(self, model_id):
        return next((m for m in self.session.models
                      if '#' + '.'.join(str(i) for i in m.id) == model_id), None)

    def _get_selected_model(self):
        idx = self._model_combo.currentIndex()
        model_id = self._model_combo.itemData(idx)
        if not model_id:
            return None
        return self._get_model_by_id(model_id)

    # ------------------------------------------------------------------
    # Manual Tube Join
    # ------------------------------------------------------------------

    def _refresh_join_tubes(self):
        """Rebuild the Tube A/B dropdowns from the currently selected particle list."""
        self._join_tube_a.blockSignals(True)
        self._join_tube_b.blockSignals(True)
        self._join_tube_a.clear()
        self._join_tube_b.clear()

        model = self._get_selected_model()
        self._join_model = model
        self._join_pid_to_tube = {}
        self._join_tube_to_pids = {}
        self._join_next_slot = 'a'

        if model is None:
            self.session.logger.info("MTFit: Manual Join — no particle list selected.")
        elif not (hasattr(model, 'get_particle') and hasattr(model, 'particle_ids')):
            self.session.logger.warning(
                f"MTFit: Manual Join — selected model ({type(model).__name__}) "
                "isn't an ArtiaX particle list.")
        else:
            tube_to_pids = {}
            pid_to_tube = {}
            n_errors = 0
            last_error = None
            for pid in model.particle_ids:
                try:
                    raw_tube_id = model.get_particle(pid)['rlnHelicalTubeID']
                except Exception as e:
                    n_errors += 1
                    last_error = e
                    continue
                tube_id = int(round(raw_tube_id))
                pid_to_tube[pid] = tube_id
                tube_to_pids.setdefault(tube_id, []).append(pid)
            self._join_pid_to_tube = pid_to_tube
            self._join_tube_to_pids = tube_to_pids

            if n_errors and not tube_to_pids:
                self.session.logger.warning(
                    f"MTFit: Manual Join — could not read rlnHelicalTubeID for any of "
                    f"{n_errors} particles ({last_error!r}). Available attributes: "
                    f"{model.get_particle(model.particle_ids[0]).attributes() if model.size else 'n/a'}")
            elif n_errors:
                self.session.logger.info(
                    f"MTFit: Manual Join — {n_errors} particle(s) missing rlnHelicalTubeID, skipped.")

            for tube_id in sorted(tube_to_pids.keys()):
                label = f"Tube {tube_id} ({len(tube_to_pids[tube_id])} pts)"
                self._join_tube_a.addItem(label, userData=tube_id)
                self._join_tube_b.addItem(label, userData=tube_id)

        if self._join_tube_a.count() == 0:
            self._join_tube_a.addItem("—", userData=None)
            self._join_tube_b.addItem("—", userData=None)

        self._join_tube_a.blockSignals(False)
        self._join_tube_b.blockSignals(False)

    def _highlight_join_selection(self):
        """Select both chosen tubes' particles in the 3D view."""
        if self._join_model is None or self._join_syncing:
            return
        tube_a = self._join_tube_a.currentData()
        tube_b = self._join_tube_b.currentData()
        wanted_pids = set()
        if tube_a is not None:
            wanted_pids.update(self._join_tube_to_pids.get(tube_a, []))
        if tube_b is not None:
            wanted_pids.update(self._join_tube_to_pids.get(tube_b, []))

        self._join_syncing = True
        try:
            mask = np.array([pid in wanted_pids for pid in self._join_model.particle_ids])
            self._join_model.selected_particles = mask
        except Exception as e:
            self.session.logger.warning(f"MTFit: could not highlight tube selection: {e}")
        finally:
            self._join_syncing = False

    def _on_chimerax_selection_changed(self, trigger_name, data):
        """When the user selects particles in the 3D view, auto-expand the
        selection to the whole tube and fill Tube A/B in rotation: 1st distinct
        tube picked goes to A, 2nd to B, 3rd back to A (overwriting), and so on
        -- so redoing a wrong pick is just "select again", no extra buttons."""
        if self._join_syncing or self._join_model is None:
            return
        model = self._join_model
        try:
            if model.deleted:
                return
            mask = model.selected_particles
        except Exception:
            return
        if mask is None:
            return

        selected_pids = [pid for pid, sel in zip(model.particle_ids, mask) if sel]
        if not selected_pids:
            return

        tube_ids = {self._join_pid_to_tube[pid] for pid in selected_pids
                    if pid in self._join_pid_to_tube}
        if len(tube_ids) != 1:
            return  # ambiguous selection spanning multiple tubes -- don't guess
        tube_id = next(iter(tube_ids))

        # Expand the selection to the whole tube, if it wasn't already
        full_pids = set(self._join_tube_to_pids.get(tube_id, []))
        if full_pids != set(selected_pids):
            self._join_syncing = True
            try:
                mask_full = np.array([pid in full_pids for pid in model.particle_ids])
                model.selected_particles = mask_full
            except Exception:
                pass
            finally:
                self._join_syncing = False

        # Don't re-fill a slot that already holds this exact tube
        current_a = self._join_tube_a.currentData()
        current_b = self._join_tube_b.currentData()
        if tube_id in (current_a, current_b):
            return

        target_combo = self._join_tube_a if self._join_next_slot == 'a' else self._join_tube_b
        idx = target_combo.findData(tube_id)
        if idx >= 0:
            target_combo.blockSignals(True)
            target_combo.setCurrentIndex(idx)
            target_combo.blockSignals(False)
            self._join_next_slot = 'b' if self._join_next_slot == 'a' else 'a'
            self._highlight_join_selection()

    def _join_selected_tubes(self):
        tube_a = self._join_tube_a.currentData()
        tube_b = self._join_tube_b.currentData()
        if tube_a is None or tube_b is None or tube_a == tube_b:
            self.session.logger.error("MTFit: pick two different tubes to join.")
            return
        model = self._join_model
        if model is None:
            self.session.logger.error("MTFit: no particle list selected.")
            return

        model_id = '#' + '.'.join(str(i) for i in model.id)
        import tempfile
        tmp_dir = tempfile.gettempdir()
        tmp_name = os.path.basename(model.name)
        if not tmp_name.lower().endswith('.star'):
            tmp_name += '.star'
        tmp_star = os.path.join(tmp_dir, tmp_name)

        out_dir = self._output_folder_edit.text().strip() or tmp_dir
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(tmp_name)[0]
        output_path = os.path.join(out_dir, f"{base}_joined.star")

        in_count = len(model.particle_ids)

        try:
            run(self.session, f'save "{tmp_star}" partlist {model_id}')

            mt_fit = os.path.join(_BUNDLE_DIR, "mt_fit.py")
            cmd = [_chimerax_python(), mt_fit, "join", tmp_star,
                   "--tube_a", str(int(tube_a)), "--tube_b", str(int(tube_b)),
                   "--angpix", str(self._voxel_size.value()),
                   "--poly_order", str(self._poly.value()),
                   "--sample_step", str(self._sample_step.value()),
                   "-o", output_path]
            env = os.environ.copy()
            env["PYTHONPATH"] = _BUNDLE_DIR + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if proc.returncode != 0:
                lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
                note = lines[-1] if lines else "Unknown error"
                self.session.logger.error(f"MTFit: join failed: {note}")
                return

            opened_model_id = None
            try:
                opened = run(self.session, f'open "{output_path}" format star')
                opened_model = self._first_model(opened)
                if opened_model is not None:
                    opened_model_id = '#' + '.'.join(str(i) for i in opened_model.id)
            except Exception as e:
                self.session.logger.warning(f"MTFit: could not auto-open joined result: {e}")

            stats = _extract_stats(output_path)
            result = dict(status="good", path=tmp_star, output_path=output_path,
                          in_particles=in_count,
                          note=f"Joined tubes {int(tube_a)}+{int(tube_b)}",
                          model_id=opened_model_id, **stats)
            self._show_results()
            self._results.append(result)
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._fill_row(row, result)
            self.session.logger.info(
                f"MTFit: joined tubes {int(tube_a)} and {int(tube_b)} -> {output_path}")
        except Exception as e:
            self.session.logger.error(f"MTFit: join failed: {e}")
        finally:
            try:
                os.remove(tmp_star)
            except Exception:
                pass
            self._refresh_join_tubes()

    # ------------------------------------------------------------------
    # Batch folder + JSON
    # ------------------------------------------------------------------

    def _browse_batch_folder(self):
        from Qt.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            None, "Select folder containing .star files for batch"
        )
        if folder:
            self._batch_folder_edit.setText(folder)

    def _get_params(self):
        return dict(
            step             = self._step_combo.itemData(self._step_combo.currentIndex()),
            voxel_size       = self._voxel_size.value(),
            sample_step      = self._sample_step.value(),
            min_seed         = self._min_seed.value(),
            poly             = self._poly.value(),
            clean_dist_thres = self._clean_dist_thres.value(),
            max_curvature    = self._max_curvature.value(),
            dist_extrapolate = self._dist_extrapolate.value(),
            overlap_thres    = self._overlap_thres.value(),
            min_part_per_tube= self._min_part_per_tube.value(),
            neighbor_rad     = self._neighbor_rad.value(),
            twist_angle      = self._twist_angle.value(),
            twist_template   = self._twist_template_edit.text().strip(),
            rot_smooth_factor= self._rot_smooth_factor.value(),
            rot_smooth_window= self._rot_smooth_window.value(),
        )

    def _apply_params(self, p):
        step = p.get("step", "pipeline")
        idx = next((i for i in range(self._step_combo.count())
                    if self._step_combo.itemData(i) == step), 0)
        self._step_combo.setCurrentIndex(idx)
        self._voxel_size.setValue(p.get("voxel_size", 14.0))
        self._sample_step.setValue(p.get("sample_step", 82.0))
        self._min_seed.setValue(p.get("min_seed", 6))
        self._poly.setValue(p.get("poly", 3))
        self._clean_dist_thres.setValue(p.get("clean_dist_thres", 50.0))
        self._max_curvature.setValue(p.get("max_curvature", 0.0))
        self._dist_extrapolate.setValue(p.get("dist_extrapolate", 2000.0))
        self._overlap_thres.setValue(p.get("overlap_thres", 100.0))
        self._min_part_per_tube.setValue(p.get("min_part_per_tube", 5))
        self._neighbor_rad.setValue(p.get("neighbor_rad", 100.0))
        self._twist_angle.setValue(p.get("twist_angle", 0.0))
        self._twist_template_edit.setText(p.get("twist_template", ""))
        self._rot_smooth_factor.setValue(p.get("rot_smooth_factor", 0.0))
        self._rot_smooth_window.setValue(p.get("rot_smooth_window", 5))

    def _auto_save_json(self):
        """Save params to the path in the JSON field, or default path if empty."""
        path = self._json_edit.text().strip()
        if not path:
            path = os.path.join(os.path.expanduser("~"), "mtfit_params.json")
            self._json_edit.setText(path)
        try:
            with open(path, "w") as f:
                json.dump(self._get_params(), f, indent=2)
        except Exception as e:
            self.session.logger.warning(f"MTFit: could not save params JSON: {e}")

    def _load_json(self):
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, "Load parameters JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                self._apply_params(json.load(f))
            self._json_edit.setText(path)
        except Exception as e:
            self.session.logger.error(f"MTFit: could not load params JSON: {e}")

    def _save_json_as(self):
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            None, "Save parameters JSON", "mtfit_params.json", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._get_params(), f, indent=2)
            self._json_edit.setText(path)
        except Exception as e:
            self.session.logger.error(f"MTFit: could not save params JSON: {e}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run_batch_clicked(self):
        if self._running:
            return
        folder = self._batch_folder_edit.text().strip()
        if not folder:
            self.session.logger.error("MTFit: set a Batch folder first.")
            return
        self._auto_save_json()
        self._start_batch(folder)

    def _run_single(self):
        if self._running:
            return
        idx = self._model_combo.currentIndex()
        model_id = self._model_combo.itemData(idx)
        if not model_id:
            self.session.logger.error("No particle list selected.")
            return

        self._auto_save_json()
        self._status.setText("Running…")
        self._run_btn.setEnabled(False)
        self._run_batch_btn.setEnabled(False)

        params = self._get_params()
        step   = params["step"]

        model = next((m for m in self.session.models
                      if '#' + '.'.join(str(i) for i in m.id) == model_id), None)
        if model is None:
            self._status.setText("Model not found.")
            self._run_btn.setEnabled(True)
            self._run_batch_btn.setEnabled(True)
            return

        import tempfile
        tmp_dir = tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_name = os.path.basename(model.name)
        if not tmp_name.lower().endswith('.star'):
            tmp_name += '.star'
        tmp_star = os.path.join(tmp_dir, tmp_name)

        result = None
        try:
            run(self.session, f'save "{tmp_star}" partlist {model_id}')

            suffix   = _OUTPUT_SUFFIX[step]
            base     = os.path.splitext(tmp_name)[0]
            out_dir  = self._output_folder_edit.text().strip() or tmp_dir
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"{base}{suffix}.star")

            returncode, stderr = _run_one(tmp_star, params, output_path)

            if returncode != 0:
                lines = [l for l in stderr.strip().splitlines() if l.strip()]
                note  = lines[-1] if lines else "Unknown error"
                self._status.setText("Failed — see Log.")
                self.session.logger.error(f"MTFit '{step}' failed:\n{stderr}")
                result = dict(status="failed", path=tmp_star, output_path=None,
                              in_particles="?", note=note,
                              particles=0, tubes=0, parts_per_tube="", mean_psi="", mean_tilt="")
            else:
                try:
                    import starfile as _sf
                    df_in = _sf.read(tmp_star)
                    if isinstance(df_in, dict):
                        df_in = df_in.get("particles", next(iter(df_in.values())))
                    in_count = len(df_in)
                except Exception:
                    in_count = "?"

                stats = _extract_stats(output_path)
                opened_model_id = None
                try:
                    opened = run(self.session, f'open "{output_path}" format star')
                    model = self._first_model(opened)
                    if model is not None:
                        opened_model_id = '#' + '.'.join(str(i) for i in model.id)
                except Exception as e:
                    self.session.logger.warning(f"MTFit: could not auto-open result: {e}")
                self._status.setText("Done.")
                result = dict(status="good", path=tmp_star, output_path=output_path,
                              in_particles=in_count, note="", model_id=opened_model_id, **stats)
        except Exception as e:
            self._status.setText("Error — see Log.")
            self.session.logger.error(f"MTFit: unexpected error in '{step}': {e}")
            result = dict(status="failed", path=tmp_star, output_path=None,
                          in_particles="?", note=str(e),
                          particles=0, tubes=0, parts_per_tube="", mean_psi="", mean_tilt="")
        finally:
            try:
                os.remove(tmp_star)
            except Exception:
                pass

        self._run_btn.setEnabled(True)
        self._run_batch_btn.setEnabled(True)
        self._show_results()
        self._results.append(result)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._fill_row(row, result)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def _start_batch(self, folder):
        star_files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".star")
        )
        if not star_files:
            self.session.logger.error(f"No .star files found in: {folder}")
            return

        self._running = True
        self._run_btn.setEnabled(False)
        self._run_batch_btn.setEnabled(False)
        self._results = []
        self._table.setRowCount(0)
        self._show_results()

        from Qt.QtWidgets import QTableWidgetItem
        for i, path in enumerate(star_files):
            self._table.insertRow(i)
            self._table.setItem(i, 0, QTableWidgetItem(""))
            self._table.setItem(i, 1, QTableWidgetItem(os.path.basename(path)))
            self._table.setItem(i, 2, QTableWidgetItem("⏳ Pending"))
            for col in range(3, 9):
                self._table.setItem(i, col, QTableWidgetItem(""))
            self._results.append({"status": "pending", "path": path})

        self._status.setText(f"0 / {len(star_files)}")
        self._poll_timer.start(200)

        params   = self._get_params()
        out_dir  = self._output_folder_edit.text().strip() or folder
        os.makedirs(out_dir, exist_ok=True)
        threading.Thread(
            target=self._batch_worker,
            args=(star_files, params, out_dir),
            daemon=True,
        ).start()

    def _batch_worker(self, files, params, out_dir):
        for i, filepath in enumerate(files):
            step   = params["step"]
            suffix = _OUTPUT_SUFFIX[step]
            base   = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(out_dir, f"{base}{suffix}.star")

            try:
                import starfile
                df_in = starfile.read(filepath)
                if isinstance(df_in, dict):
                    df_in = df_in.get("particles", next(iter(df_in.values())))
                in_count = len(df_in)
            except Exception:
                in_count = "?"

            returncode, stderr = _run_one(filepath, params, output_path)

            if returncode != 0:
                lines = [l for l in stderr.strip().splitlines() if l.strip()]
                note  = lines[-1] if lines else "Unknown error"
                result = dict(index=i, status="failed", path=filepath, output_path=None,
                              in_particles=in_count, note=note,
                              particles=0, tubes=0, parts_per_tube="", mean_psi="", mean_tilt="")
            else:
                stats = _extract_stats(output_path)
                out_p = stats["particles"]
                low   = (isinstance(in_count, int) and in_count > 0
                         and out_p / in_count < 0.3)
                prob  = (stats["tubes"] == 0 or low)
                if stats["tubes"] == 0:
                    note = "No tubes found"
                elif low:
                    pct  = int(100 * out_p / in_count)
                    note = f"Low yield ({pct}% kept)"
                else:
                    note = ""
                result = dict(index=i, status="problematic" if prob else "good",
                              path=filepath, output_path=output_path,
                              in_particles=in_count, note=note, **stats)

            self._result_queue.put(result)

        self._result_queue.put(None)  # sentinel

    def _poll_results(self):
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if result is None:
                self._poll_timer.stop()
                self._running = False
                self._run_btn.setEnabled(True)
                self._run_batch_btn.setEnabled(True)
                good = sum(1 for r in self._results if r.get("status") == "good")
                prob = sum(1 for r in self._results
                           if r.get("status") in ("problematic", "failed"))
                self._status.setText(f"Done — {good} good, {prob} problematic")
                self._auto_save_csv()
                break

            i = result["index"]
            self._results[i] = result
            self._fill_row(i, result)
            done = sum(1 for r in self._results if r.get("status") != "pending")
            self._status.setText(f"{done} / {len(self._results)}")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------

    def _show_results(self):
        self._results_group.setVisible(True)

    def _fill_row(self, i, result):
        from Qt.QtWidgets import QTableWidgetItem, QPushButton, QCheckBox, QWidget, QHBoxLayout
        from Qt.QtGui import QColor
        from Qt.QtCore import Qt

        status = result["status"]
        if status == "good":
            label = "✓ Good"
            color = QColor(200, 240, 200)
        elif status == "problematic":
            label = "⚠ Problematic"
            color = QColor(255, 220, 150)
        elif status == "failed":
            label = "✗ Failed"
            color = QColor(255, 180, 180)
        else:
            label = "⏳ Running"
            color = None

        # col 0: checkbox (only for rows with output)
        if status in ("good", "problematic") and result.get("output_path"):
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.setContentsMargins(4, 0, 4, 0)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb = QCheckBox()
            cb_layout.addWidget(cb)
            self._table.setCellWidget(i, 0, cb_widget)
        else:
            self._table.setCellWidget(i, 0, None)
            self._table.setItem(i, 0, QTableWidgetItem(""))

        # cols 1-7: text data
        display_name = (os.path.basename(result["output_path"])
                        if result.get("output_path") else os.path.basename(result["path"]))
        vals = [
            display_name,
            label,
            str(result.get("in_particles", "")),
            str(result.get("particles", "")),
            str(result.get("tubes", "")),
            str(result.get("parts_per_tube", "")),
            str(result.get("mean_psi", "")),
        ]
        for offset, val in enumerate(vals):
            col = offset + 1
            item = self._table.item(i, col) or QTableWidgetItem()
            item.setText(val)
            self._table.setItem(i, col, item)
            if color:
                item.setBackground(color)

        # col 8: Open / Save buttons
        if status in ("good", "problematic") and result.get("output_path"):
            path = result["output_path"]
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(3)

            open_btn = QPushButton("Open")
            open_btn.setFixedWidth(48)
            open_btn.clicked.connect(lambda _c, p=path, row=i: self._open_row(row, p))

            save_btn = QPushButton("Save")
            save_btn.setFixedWidth(44)
            save_btn.setToolTip(
                "Save the currently-open ChimeraX model back to this file "
                "(use after manual edits). If no model is open, copies the "
                "output file to the Output folder.")
            save_btn.clicked.connect(lambda _c, p=path, row=i: self._save_row(row, p))

            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(24)
            remove_btn.setToolTip(
                "Close this row's model in ChimeraX (does not delete the output file)")
            remove_btn.clicked.connect(lambda _c, row=i: self._remove_row(row))

            cell_layout.addWidget(open_btn)
            cell_layout.addWidget(save_btn)
            cell_layout.addWidget(remove_btn)
            self._table.setCellWidget(i, 8, cell)

    def _open_row(self, row, path):
        try:
            opened = run(self.session, f'open "{path}" format star')
            model = self._first_model(opened)
            if model is not None:
                self._results[row]["model_id"] = '#' + '.'.join(str(i) for i in model.id)
        except Exception as e:
            self.session.logger.warning(f"MTFit: could not open {path}: {e}")

    def _remove_row(self, row):
        model_id = self._results[row].get("model_id")
        if not model_id:
            self.session.logger.info("MTFit: nothing open for this row.")
            return
        self._close_model(model_id)
        self._results[row]["model_id"] = None

    def _save_row(self, row, output_path):
        """Save this row's own tracked ChimeraX model to a user-chosen location."""
        from Qt.QtWidgets import QFileDialog
        basename = os.path.basename(output_path)

        default_dir = self._output_folder_edit.text().strip() or os.path.expanduser("~")
        save_path, _ = QFileDialog.getSaveFileName(
            None, "Save particle list as", os.path.join(default_dir, basename),
            "STAR files (*.star)"
        )
        if not save_path:
            return

        # Use the model this specific row opened — not a name-based search, since
        # MTFit reuses the same output filename across re-runs of the same input,
        # so multiple rows can have same-named models open simultaneously and a
        # name search would silently grab the wrong (e.g. stale) one.
        model_id = self._results[row].get("model_id")
        model = next((m for m in self.session.models
                      if '#' + '.'.join(str(i) for i in m.id) == model_id), None) if model_id else None
        if model is not None:
            run(self.session, f'save "{save_path}" partlist {model_id}')
        elif os.path.exists(output_path):
            import shutil
            shutil.copy2(output_path, save_path)
        else:
            self.session.logger.warning("MTFit: output file not found — open it first.")
            return
        self.session.logger.info(f"MTFit: saved to {save_path}")

    def _clear_results(self):
        for r in self._results:
            self._close_model(r.get("model_id"))
        for model_id in self._extra_model_ids:
            self._close_model(model_id)
        self._extra_model_ids = []
        self._results = []
        self._table.setRowCount(0)
        self._auto_csv_lbl.setText("")

    def _combine_selected(self):
        """Combine the checked output STAR files into one file."""
        from Qt.QtWidgets import QCheckBox, QFileDialog

        checked_paths = []
        for row in range(self._table.rowCount()):
            cb_widget = self._table.cellWidget(row, 0)
            if cb_widget is None:
                continue
            cb = cb_widget.findChild(QCheckBox)
            if cb is not None and cb.isChecked():
                result = self._results[row]
                path = result.get("output_path")
                if path and os.path.exists(path):
                    checked_paths.append(path)

        if not checked_paths:
            self.session.logger.warning("MTFit: no rows checked for combining.")
            return

        out_dir = self._output_folder_edit.text().strip() or os.path.expanduser("~")
        default = os.path.join(out_dir, "combined.star")
        out_path, _ = QFileDialog.getSaveFileName(
            None, "Save combined STAR file as", default, "STAR files (*.star)"
        )
        if not out_path:
            return

        try:
            from .utils.io import combine_star_files
            combine_star_files(checked_paths, out_path)
            self.session.logger.info(
                f"MTFit: combined {len(checked_paths)} files → {out_path}"
            )
            opened = run(self.session, f'open "{out_path}" format star')
            model = self._first_model(opened)
            if model is not None:
                self._extra_model_ids.append('#' + '.'.join(str(i) for i in model.id))
        except Exception as e:
            self.session.logger.error(f"MTFit: combine failed: {e}")

    def _auto_save_csv(self):
        """Auto-save CSV to output folder after batch completes."""
        out_dir = self._output_folder_edit.text().strip()
        if not out_dir:
            return
        path = os.path.join(out_dir, "mtfit_results.csv")
        self._write_csv(path)
        self._auto_csv_lbl.setText(f"Auto-saved: {os.path.basename(path)}")
        self.session.logger.info(f"MTFit: results auto-saved to {path}")

    def _save_csv(self):
        from Qt.QtWidgets import QFileDialog
        default = os.path.join(
            self._output_folder_edit.text().strip() or os.path.expanduser("~"),
            "mtfit_results.csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            None, "Save results as CSV", default, "CSV files (*.csv)"
        )
        if path:
            self._write_csv(path)

    def _write_csv(self, path):
        import csv
        headers = ["File", "Status", "Input particles", "Output particles",
                   "Tubes", "Particles/tube", "Mean Psi (deg)", "Note", "Output path"]
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)
                for r in self._results:
                    w.writerow([
                        os.path.basename(r.get("path", "")),
                        r.get("status", ""),
                        r.get("in_particles", ""),
                        r.get("particles", ""),
                        r.get("tubes", ""),
                        r.get("parts_per_tube", ""),
                        r.get("mean_psi", ""),
                        r.get("note", ""),
                        r.get("output_path", ""),
                    ])
            self.session.logger.info(f"MTFit: results saved to {path}")
        except Exception as e:
            self.session.logger.error(f"MTFit: could not save CSV: {e}")
