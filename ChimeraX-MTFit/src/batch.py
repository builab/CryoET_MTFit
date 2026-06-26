import os
import queue
import subprocess
import sys
import sysconfig
import threading

from chimerax.core.commands import run as chimerax_run

_BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

_OUTPUT_SUFFIX = {
    "fit":      "_fitted",
    "clean":    "_cleaned",
    "connect":  "_connected",
    "predict":  "_predicted",
    "pipeline": "_processed",
}


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


class MTFitBatchDialog:
    """Floating batch-processing window launched from the MTFit panel."""

    def __init__(self, session, default_params=None):
        self.session = session
        self._params = default_params or {}
        self._files = []
        self._row_data = []
        self._result_queue = queue.Queue()
        self._running = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from Qt.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
            QPushButton, QLabel, QListWidget, QTableWidget,
            QTableWidgetItem, QFileDialog, QComboBox, QGroupBox,
            QDoubleSpinBox, QSpinBox, QLineEdit, QAbstractItemView,
            QHeaderView, QSizePolicy,
        )
        from Qt.QtCore import Qt, QTimer

        self._dialog = QDialog()
        self._dialog.setWindowTitle("MTFit — Batch Processing")
        self._dialog.resize(950, 750)
        self._dialog.setAttribute(Qt.WA_DeleteOnClose)

        main = QVBoxLayout(self._dialog)
        main.setSpacing(8)
        main.setContentsMargins(10, 10, 10, 10)

        # ---- File list ----
        files_group = QGroupBox("Input files")
        fl = QVBoxLayout(files_group)

        btn_row = QHBoxLayout()
        add_folder_btn = QPushButton("Add folder…")
        add_files_btn  = QPushButton("Add files…")
        remove_btn     = QPushButton("Remove selected")
        clear_btn      = QPushButton("Clear all")
        add_folder_btn.clicked.connect(self._add_folder)
        add_files_btn.clicked.connect(self._add_files)
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn.clicked.connect(self._clear_files)
        self._file_count_lbl = QLabel("0 files")
        for w in (add_folder_btn, add_files_btn, remove_btn, clear_btn):
            btn_row.addWidget(w)
        btn_row.addStretch()
        btn_row.addWidget(self._file_count_lbl)
        fl.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setMaximumHeight(110)
        fl.addWidget(self._file_list)
        main.addWidget(files_group)

        # ---- Options ----
        opts_group = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_group)

        # Step + output dir row
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Step:"))
        self._step_combo = QComboBox()
        self._step_combo.addItem("Full pipeline (Fit → Clean → Connect → Predict)", userData="pipeline")
        self._step_combo.addItem("1. Fit only",     userData="fit")
        self._step_combo.addItem("2. Clean only",   userData="clean")
        self._step_combo.addItem("3. Connect only", userData="connect")
        self._step_combo.addItem("4. Predict only", userData="predict")
        self._step_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_row.addWidget(self._step_combo)
        top_row.addSpacing(16)
        top_row.addWidget(QLabel("Output dir:"))
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("Same folder as each input file")
        self._out_dir_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_row.addWidget(self._out_dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_output_dir)
        top_row.addWidget(browse_btn)
        opts_layout.addLayout(top_row)

        # Parameters — two columns
        p = self._params
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)

        def fs(lo, hi, val, step): return self._float_spin(lo, hi, val, step)
        def is_(lo, hi, val):       return self._int_spin(lo, hi, val)

        self._voxel_size        = fs(1, 100,    p.get("voxel_size", 14.0),    1)
        self._sample_step       = fs(1, 500,    p.get("sample_step", 82.0),   1)
        self._min_seed          = is_(2, 50,    p.get("min_seed", 6))
        self._poly              = is_(1, 5,     p.get("poly", 3))
        self._clean_dist_thres  = fs(1, 500,    p.get("clean_dist_thres", 50.0),  1)
        self._dist_extrapolate  = fs(1, 10000,  p.get("dist_extrapolate", 2000.0), 10)
        self._overlap_thres     = fs(1, 500,    p.get("overlap_thres", 100.0), 1)
        self._min_part_per_tube = is_(1, 50,    p.get("min_part_per_tube", 5))
        self._neighbor_rad      = fs(1, 1000,   p.get("neighbor_rad", 100.0),  1)

        rows = [
            ("Voxel size (Å/px):", self._voxel_size,        "Sample step (Å):",  self._sample_step),
            ("Min seed:",           self._min_seed,           "Polynomial order:", self._poly),
            ("Clean dist (Å):",     self._clean_dist_thres,   "Extrapolate (Å):",  self._dist_extrapolate),
            ("Overlap thres (Å):",  self._overlap_thres,      "Min parts/tube:",   self._min_part_per_tube),
            ("Neighbor rad (Å):",   self._neighbor_rad,       None,                None),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            grid.addWidget(QLabel(l1), r, 0)
            grid.addWidget(w1,         r, 1)
            if l2:
                grid.addWidget(QLabel(l2), r, 2)
                grid.addWidget(w2,         r, 3)
        opts_layout.addLayout(grid)
        main.addWidget(opts_group)

        # ---- Run row ----
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.clicked.connect(self._start_batch)
        self._progress_lbl = QLabel("")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._progress_lbl)
        run_row.addStretch()
        main.addLayout(run_row)

        # ---- Results table ----
        results_group = QGroupBox("Results")
        rl = QVBoxLayout(results_group)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show:"))
        self._btn_all  = QPushButton("All")
        self._btn_good = QPushButton("Good")
        self._btn_prob = QPushButton("Problematic")
        for btn in (self._btn_all, self._btn_good, self._btn_prob):
            btn.setCheckable(True)
            btn.setFixedWidth(100)
        self._btn_all.setChecked(True)
        self._btn_all.clicked.connect(lambda: self._apply_filter("all"))
        self._btn_good.clicked.connect(lambda: self._apply_filter("good"))
        self._btn_prob.clicked.connect(lambda: self._apply_filter("problematic"))
        filter_row.addWidget(self._btn_all)
        filter_row.addWidget(self._btn_good)
        filter_row.addWidget(self._btn_prob)
        filter_row.addStretch()
        rl.addLayout(filter_row)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["File", "Status", "In ✦", "Out ✦", "Tubes", "Note", "Action"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        rl.addWidget(self._table)
        main.addWidget(results_group)

        # Poll timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_results)

        self._dialog.show()

    def _float_spin(self, lo, hi, val, step):
        from Qt.QtWidgets import QDoubleSpinBox
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setSingleStep(step)
        return w

    def _int_spin(self, lo, hi, val):
        from Qt.QtWidgets import QSpinBox
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _add_folder(self):
        from Qt.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self._dialog, "Select folder containing .star files"
        )
        if not folder:
            return
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".star"):
                path = os.path.join(folder, fname)
                if path not in self._files:
                    self._files.append(path)
                    self._file_list.addItem(path)
        self._update_count()

    def _add_files(self):
        from Qt.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self._dialog, "Select STAR files", "", "STAR files (*.star)"
        )
        for path in paths:
            if path not in self._files:
                self._files.append(path)
                self._file_list.addItem(path)
        self._update_count()

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            path = item.text()
            if path in self._files:
                self._files.remove(path)
            self._file_list.takeItem(self._file_list.row(item))
        self._update_count()

    def _clear_files(self):
        self._files.clear()
        self._file_list.clear()
        self._update_count()

    def _update_count(self):
        n = len(self._files)
        self._file_count_lbl.setText(f"{n} file{'s' if n != 1 else ''}")

    def _browse_output_dir(self):
        from Qt.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self._dialog, "Select output directory")
        if folder:
            self._out_dir_edit.setText(folder)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def _start_batch(self):
        if self._running or not self._files:
            return

        self._running = True
        self._run_btn.setEnabled(False)
        self._table.setRowCount(0)
        self._row_data = []

        from Qt.QtWidgets import QTableWidgetItem
        for path in self._files:
            i = self._table.rowCount()
            self._table.insertRow(i)
            self._table.setItem(i, 0, QTableWidgetItem(os.path.basename(path)))
            self._table.setItem(i, 1, QTableWidgetItem("⏳ Pending"))
            for col in range(2, 7):
                self._table.setItem(i, col, QTableWidgetItem(""))
            self._row_data.append({"status": "pending", "path": path})

        self._progress_lbl.setText(f"0 / {len(self._files)}")
        self._timer.start(200)

        step    = self._step_combo.itemData(self._step_combo.currentIndex())
        out_dir = self._out_dir_edit.text().strip() or None
        params  = dict(
            step             = step,
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

        threading.Thread(
            target=self._batch_worker,
            args=(list(self._files), params, out_dir),
            daemon=True,
        ).start()

    def _batch_worker(self, files, params, out_dir):
        for i, filepath in enumerate(files):
            result = self._process_one(i, filepath, params, out_dir)
            self._result_queue.put(result)
        self._result_queue.put(None)  # sentinel

    def _process_one(self, index, filepath, params, out_dir):
        step   = params["step"]
        suffix = _OUTPUT_SUFFIX[step]
        base   = os.path.splitext(os.path.basename(filepath))[0]
        dest   = out_dir or os.path.dirname(filepath)
        output_path = os.path.join(dest, f"{base}{suffix}.star")

        # Count input particles
        try:
            import starfile as sf
            df_in = sf.read(filepath)
            if isinstance(df_in, dict):
                df_in = df_in.get("particles", next(iter(df_in.values())))
            in_particles = len(df_in)
        except Exception:
            in_particles = "?"

        # Build subprocess command
        mt_fit = os.path.join(_BUNDLE_DIR, "mt_fit.py")
        cmd = [_chimerax_python(), mt_fit, step, filepath,
               "--angpix", str(params["voxel_size"]), "-o", output_path]

        if step in ("fit", "pipeline"):
            cmd += ["--sample_step",  str(params["sample_step"]),
                    "--min_seed",     str(params["min_seed"]),
                    "--poly_order",   str(params["poly"])]
        if step in ("clean", "pipeline"):
            cmd += ["--dist_thres",   str(params["clean_dist_thres"])]
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

        if proc.returncode != 0:
            lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
            note  = lines[-1] if lines else "Unknown error"
            return dict(index=index, status="failed", path=filepath,
                        output_path=None, in_particles=in_particles,
                        out_particles=0, tubes=0, note=note)

        # Parse output stats
        out_particles = 0
        tubes = 0
        try:
            import starfile as sf
            df_out = sf.read(output_path)
            if isinstance(df_out, dict):
                df_out = df_out.get("particles", next(iter(df_out.values())))
            out_particles = len(df_out)
            if "rlnHelicalTubeID" in df_out.columns:
                tubes = df_out["rlnHelicalTubeID"].nunique()
        except Exception:
            pass

        # Classify as problematic?
        low_yield = (isinstance(in_particles, int) and in_particles > 0
                     and out_particles / in_particles < 0.3)
        problematic = (tubes == 0 or low_yield)

        if tubes == 0:
            note = "No tubes found"
        elif low_yield:
            pct  = int(100 * out_particles / in_particles)
            note = f"Low yield ({pct}% kept)"
        else:
            note = ""

        return dict(index=index,
                    status="problematic" if problematic else "good",
                    path=filepath, output_path=output_path,
                    in_particles=in_particles, out_particles=out_particles,
                    tubes=tubes, note=note)

    # ------------------------------------------------------------------
    # Results polling + table updates (runs on Qt main thread via timer)
    # ------------------------------------------------------------------

    def _poll_results(self):
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if result is None:
                self._timer.stop()
                self._running = False
                self._run_btn.setEnabled(True)
                good = sum(1 for r in self._row_data if r.get("status") == "good")
                prob = sum(1 for r in self._row_data
                           if r.get("status") in ("problematic", "failed"))
                self._progress_lbl.setText(
                    f"Done — {good} good, {prob} problematic"
                )
                break

            i = result["index"]
            self._row_data[i] = result
            self._update_row(i, result)

            done = sum(1 for r in self._row_data if r.get("status") != "pending")
            self._progress_lbl.setText(f"{done} / {len(self._files)}")

    def _update_row(self, i, result):
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

        self._table.item(i, 0).setText(os.path.basename(result["path"]))
        self._table.setItem(i, 1, QTableWidgetItem(label))
        self._table.setItem(i, 2, QTableWidgetItem(str(result.get("in_particles",  ""))))
        self._table.setItem(i, 3, QTableWidgetItem(str(result.get("out_particles", ""))))
        self._table.setItem(i, 4, QTableWidgetItem(str(result.get("tubes", ""))))
        self._table.setItem(i, 5, QTableWidgetItem(result.get("note", "")))

        if color:
            for col in range(6):
                item = self._table.item(i, col)
                if item:
                    item.setBackground(color)

        if status in ("good", "problematic") and result.get("output_path"):
            btn = QPushButton("Open")
            btn.setFixedWidth(60)
            path = result["output_path"]
            btn.clicked.connect(lambda _checked, p=path: self._open_in_chimerax(p))
            self._table.setCellWidget(i, 6, btn)

        self._apply_current_filter(i)

    def _open_in_chimerax(self, path):
        chimerax_run(self.session, f'open "{path}" format star')

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filter(self, mode):
        self._btn_all.setChecked(mode == "all")
        self._btn_good.setChecked(mode == "good")
        self._btn_prob.setChecked(mode == "problematic")
        self._current_filter = mode
        for i in range(self._table.rowCount()):
            self._apply_current_filter(i)

    def _apply_current_filter(self, i):
        mode   = getattr(self, "_current_filter", "all")
        status = self._row_data[i].get("status", "pending") if i < len(self._row_data) else "pending"
        if mode == "all":
            hide = False
        elif mode == "good":
            hide = status != "good"
        else:
            hide = status not in ("problematic", "failed")
        self._table.setRowHidden(i, hide)
