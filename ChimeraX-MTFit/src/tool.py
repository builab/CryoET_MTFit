import json
import os
import queue
import subprocess
import sys
import sysconfig
import threading

from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS

_BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

_OUTPUT_SUFFIX = {
    "fit":      "_fitted",
    "clean":    "_cleaned",
    "connect":  "_connected",
    "predict":  "_predicted",
    "pipeline": "_processed",
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
    if step in ("fit", "pipeline"):
        cmd += ["--sample_step", str(params["sample_step"]),
                "--min_seed",    str(params["min_seed"]),
                "--poly_order",  str(params["poly"])]
    if step in ("clean", "pipeline"):
        cmd += ["--dist_thres",  str(params["clean_dist_thres"])]
    if step in ("connect", "pipeline"):
        cmd += ["--dist_extrapolate",  str(params["dist_extrapolate"]),
                "--overlap_thres",     str(params["overlap_thres"]),
                "--min_part_per_tube", str(params["min_part_per_tube"])]
    if step in ("predict", "pipeline"):
        cmd += ["--template",     filepath,
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

    def delete(self):
        self.session.triggers.remove_handler(self._add_handler)
        self.session.triggers.remove_handler(self._remove_handler)
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
        self._step_combo.addItem("Full pipeline (Fit → Clean → Connect → Predict)", userData="pipeline")
        self._step_combo.addItem("1. Fit only",     userData="fit")
        self._step_combo.addItem("2. Clean only",   userData="clean")
        self._step_combo.addItem("3. Connect only", userData="connect")
        self._step_combo.addItem("4. Predict only", userData="predict")
        self._step_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        step_row.addWidget(self._step_combo)
        main_layout.addLayout(step_row)

        # ---- Parameter tabs ----
        tabs = QTabWidget()

        basic_tab = QWidget()
        basic_form = QFormLayout(basic_tab)
        basic_form.setContentsMargins(8, 8, 8, 8)
        self._voxel_size  = self._float_spin(1.0,   100.0,  14.0, 1, "Å/px")
        self._sample_step = self._float_spin(1.0,   500.0,  82.0, 1, "Å")
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
        self._clean_dist_thres  = self._float_spin(1.0,   500.0,   50.0,  1,  "Å")
        self._dist_extrapolate  = self._float_spin(1.0, 10000.0, 2000.0, 10,  "Å")
        self._overlap_thres     = self._float_spin(1.0,   500.0,  100.0,  1,  "Å")
        self._min_part_per_tube = self._int_spin(1, 50, 5)
        cc_form.addRow("Clean dist threshold (Å):", self._clean_dist_thres)
        cc_form.addRow("Extrapolate distance (Å):", self._dist_extrapolate)
        cc_form.addRow("Overlap threshold (Å):",    self._overlap_thres)
        cc_form.addRow("Min particles per tube:",   self._min_part_per_tube)
        tabs.addTab(cc_tab, "Clean / Connect")

        pred_tab = QWidget()
        pred_form = QFormLayout(pred_tab)
        pred_form.setContentsMargins(8, 8, 8, 8)
        self._neighbor_rad = self._float_spin(1.0, 1000.0, 100.0, 1, "Å")
        pred_form.addRow("Neighbor radius (Å):", self._neighbor_rad)
        tabs.addTab(pred_tab, "Predict")

        main_layout.addWidget(tabs)

        # ---- Batch & Parameters JSON section ----
        batch_group = QGroupBox("Batch & Parameters")
        batch_form = QFormLayout(batch_group)
        batch_form.setContentsMargins(8, 8, 8, 8)

        # Output folder
        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setPlaceholderText("Where to save processed files")
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

        # ---- Results table ----
        self._results_group = QGroupBox(
            "Results  (⚠ Problematic = 0 tubes found OR <30% particles retained)"
        )
        results_layout = QVBoxLayout(self._results_group)

        csv_row = QHBoxLayout()
        self._auto_csv_lbl = QLabel("")
        csv_row.addWidget(self._auto_csv_lbl)
        csv_row.addStretch()
        save_csv_btn = QPushButton("Save CSV…")
        save_csv_btn.setToolTip("Save an additional copy of the results table")
        save_csv_btn.clicked.connect(self._save_csv)
        csv_row.addWidget(save_csv_btn)
        results_layout.addLayout(csv_row)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["File", "Status", "In ✦", "Out ✦", "Tubes", "Pts/Tube", "Mean Psi°", "Open / Save"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4, 5, 6, 7):
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

    def _on_models_changed(self, trigger_name, changes):
        self._refresh_models()

    def _refresh_models(self):
        self._model_combo.clear()
        for m in self.session.models:
            if hasattr(m, 'name') and m.name.endswith('.star'):
                id_str = '#' + '.'.join(str(i) for i in m.id)
                self._model_combo.addItem(f"{id_str} — {m.name}", userData=id_str)
        if self._model_combo.count() == 0:
            self._model_combo.addItem("No particle lists found")

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
            dist_extrapolate = self._dist_extrapolate.value(),
            overlap_thres    = self._overlap_thres.value(),
            min_part_per_tube= self._min_part_per_tube.value(),
            neighbor_rad     = self._neighbor_rad.value(),
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
        self._dist_extrapolate.setValue(p.get("dist_extrapolate", 2000.0))
        self._overlap_thres.setValue(p.get("overlap_thres", 100.0))
        self._min_part_per_tube.setValue(p.get("min_part_per_tube", 5))
        self._neighbor_rad.setValue(p.get("neighbor_rad", 100.0))

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
        tmp_star = os.path.join(tmp_dir, os.path.basename(model.name))
        run(self.session, f'save "{tmp_star}" partlist {model_id}')

        suffix   = _OUTPUT_SUFFIX[step]
        base     = os.path.splitext(os.path.basename(tmp_star))[0]
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
            # Count input particles
            try:
                import starfile
                df_in = starfile.read(tmp_star)
                if isinstance(df_in, dict):
                    df_in = df_in.get("particles", next(iter(df_in.values())))
                in_count = len(df_in)
            except Exception:
                in_count = "?"

            stats = _extract_stats(output_path)
            run(self.session, f'open "{output_path}" format star')
            self._status.setText("Done.")
            result = dict(status="good", path=tmp_star, output_path=output_path,
                          in_particles=in_count, note="", **stats)

        # Clean up input temp (not output — user may want to re-open)
        try:
            os.remove(tmp_star)
        except Exception:
            pass

        self._run_btn.setEnabled(True)
        self._run_batch_btn.setEnabled(True)
        self._show_results()
        self._results = [result]
        self._table.setRowCount(0)
        self._table.insertRow(0)
        self._fill_row(0, result)

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
            self._table.setItem(i, 0, QTableWidgetItem(os.path.basename(path)))
            self._table.setItem(i, 1, QTableWidgetItem("⏳ Pending"))
            for col in range(2, 8):
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
        from Qt.QtWidgets import QTableWidgetItem, QPushButton
        from Qt.QtGui import QColor

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

        vals = [
            os.path.basename(result["path"]),
            label,
            str(result.get("in_particles", "")),
            str(result.get("particles", "")),
            str(result.get("tubes", "")),
            str(result.get("parts_per_tube", "")),
            str(result.get("mean_psi", "")),
        ]
        for col, val in enumerate(vals):
            item = self._table.item(i, col) or __import__("Qt.QtWidgets", fromlist=["QTableWidgetItem"]).QTableWidgetItem()
            item.setText(val)
            self._table.setItem(i, col, item)
            if color:
                item.setBackground(color)

        if status in ("good", "problematic") and result.get("output_path"):
            from Qt.QtWidgets import QWidget, QHBoxLayout
            path = result["output_path"]
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(3)

            open_btn = QPushButton("Open")
            open_btn.setFixedWidth(48)
            open_btn.clicked.connect(
                lambda _c, p=path: run(self.session, f'open "{p}" format star'))

            save_btn = QPushButton("Save")
            save_btn.setFixedWidth(44)
            save_btn.setToolTip(
                "Save the currently-open ChimeraX model back to this file "
                "(use after manual edits). If no model is open, copies the "
                "output file to the Output folder.")
            save_btn.clicked.connect(lambda _c, p=path: self._save_row(p))

            cell_layout.addWidget(open_btn)
            cell_layout.addWidget(save_btn)
            self._table.setCellWidget(i, 7, cell)

    def _save_row(self, output_path):
        """Save the matching open ChimeraX model back to disk (after manual edits)."""
        basename = os.path.basename(output_path)
        # Find a loaded model whose name matches
        model = next((m for m in self.session.models
                      if hasattr(m, 'name') and m.name == basename), None)
        if model is None:
            # Fall back: copy output file to output folder if set
            out_dir = self._output_folder_edit.text().strip()
            if out_dir and os.path.exists(output_path):
                import shutil
                dest = os.path.join(out_dir, basename)
                shutil.copy2(output_path, dest)
                self.session.logger.info(f"MTFit: copied to {dest}")
            else:
                self.session.logger.warning(
                    "MTFit: open the file in ChimeraX first to save edits.")
            return
        model_id = '#' + '.'.join(str(i) for i in model.id)
        out_dir  = self._output_folder_edit.text().strip()
        save_path = os.path.join(out_dir, basename) if out_dir else output_path
        run(self.session, f'save "{save_path}" partlist {model_id}')
        self.session.logger.info(f"MTFit: saved to {save_path}")

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
