from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS


class MTFitTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = "help:user/tools/MTFit.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self._build_ui()
        self.tool_window.manage(placement=None)  # floating window

        # Auto-refresh the model dropdown whenever models are added or removed
        self._add_handler    = session.triggers.add_handler(ADD_MODELS,    self._on_models_changed)
        self._remove_handler = session.triggers.add_handler(REMOVE_MODELS, self._on_models_changed)

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
            QComboBox, QFormLayout, QSizePolicy,
        )
        from Qt.QtCore import Qt

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
        self._step_combo.addItem("1. Fit only", userData="fit")
        self._step_combo.addItem("2. Clean only", userData="clean")
        self._step_combo.addItem("3. Connect only", userData="connect")
        self._step_combo.addItem("4. Predict only", userData="predict")
        self._step_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._step_combo.setToolTip(
            "Run the full pipeline, or just one step on the selected particle list.\n"
            "Running a single step expects the input to already be at the right stage\n"
            "(e.g. run 'Connect' on the output of 'Clean')."
        )
        step_row.addWidget(self._step_combo)
        main_layout.addLayout(step_row)

        # ---- Tabs ----
        tabs = QTabWidget()

        # Tab 1: Basic
        basic_tab = QWidget()
        basic_form = QFormLayout(basic_tab)
        basic_form.setContentsMargins(8, 8, 8, 8)

        self._voxel_size   = self._float_spin(1.0, 100.0, 14.0, 1, "Å/px")
        self._sample_step  = self._float_spin(1.0, 500.0, 82.0, 1, "Å")
        self._min_seed     = self._int_spin(2, 50, 6)
        self._poly         = self._int_spin(1, 5, 3)

        basic_form.addRow("Voxel size (Å/px):", self._voxel_size)
        basic_form.addRow("Sample step (Å):",   self._sample_step)
        basic_form.addRow("Min seed:",           self._min_seed)
        basic_form.addRow("Polynomial order:",   self._poly)
        tabs.addTab(basic_tab, "Basic")

        # Tab 2: Clean / Connect
        cc_tab = QWidget()
        cc_form = QFormLayout(cc_tab)
        cc_form.setContentsMargins(8, 8, 8, 8)

        self._clean_dist_thres  = self._float_spin(1.0, 500.0, 50.0,   1, "Å")
        self._dist_extrapolate  = self._float_spin(1.0, 10000.0, 2000.0, 10, "Å")
        self._overlap_thres     = self._float_spin(1.0, 500.0, 100.0,  1, "Å")
        self._min_part_per_tube = self._int_spin(1, 50, 5)

        cc_form.addRow("Clean dist threshold (Å):", self._clean_dist_thres)
        cc_form.addRow("Extrapolate distance (Å):", self._dist_extrapolate)
        cc_form.addRow("Overlap threshold (Å):",    self._overlap_thres)
        cc_form.addRow("Min particles per tube:",   self._min_part_per_tube)
        tabs.addTab(cc_tab, "Clean / Connect")

        # Tab 3: Predict
        pred_tab = QWidget()
        pred_form = QFormLayout(pred_tab)
        pred_form.setContentsMargins(8, 8, 8, 8)

        self._neighbor_rad = self._float_spin(1.0, 1000.0, 100.0, 1, "Å")
        pred_form.addRow("Neighbor radius (Å):", self._neighbor_rad)
        tabs.addTab(pred_tab, "Predict")

        main_layout.addWidget(tabs)

        # ---- Run button ----
        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.clicked.connect(self._run)
        main_layout.addWidget(self._run_btn)
        self._step_combo.currentIndexChanged.connect(self._update_run_button_label)
        self._update_run_button_label()

        # ---- Status label ----
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self._status)

        self._refresh_models()

    # ------------------------------------------------------------------
    # Helpers
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

    def _update_run_button_label(self):
        idx = self._step_combo.currentIndex()
        step = self._step_combo.itemData(idx)
        label = "Run Full Pipeline" if step == "pipeline" else f"Run Step: {self._step_combo.currentText()}"
        self._run_btn.setText(label)

    def _on_models_changed(self, trigger_name, changes):
        self._refresh_models()

    def _refresh_models(self):
        self._model_combo.clear()
        for m in self.session.models:
            # ArtiaX particle lists have a 'name' ending in .star
            if hasattr(m, 'name') and m.name.endswith('.star'):
                id_str = '#' + '.'.join(str(i) for i in m.id)
                self._model_combo.addItem(f"{id_str} — {m.name}", userData=id_str)

        if self._model_combo.count() == 0:
            self._model_combo.addItem("No particle lists found")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        idx = self._model_combo.currentIndex()
        model_id = self._model_combo.itemData(idx)
        if not model_id:
            self.session.logger.error("No particle list selected.")
            return

        step = self._step_combo.itemData(self._step_combo.currentIndex())

        self._status.setText("Running…")
        self._run_btn.setEnabled(False)

        cmd = (
            f"mtfit {model_id}"
            f" step {step}"
            f" voxel_size {self._voxel_size.value()}"
            f" sample_step {self._sample_step.value()}"
            f" min_seed {self._min_seed.value()}"
            f" poly {self._poly.value()}"
            f" clean_dist_thres {self._clean_dist_thres.value()}"
            f" dist_extrapolate {self._dist_extrapolate.value()}"
            f" overlap_thres {self._overlap_thres.value()}"
            f" min_part_per_tube {self._min_part_per_tube.value()}"
            f" neighbor_rad {self._neighbor_rad.value()}"
        )

        try:
            run(self.session, cmd)
            self._status.setText("Done.")
        except Exception as e:
            self._status.setText("Failed — see Log.")
            self.session.logger.error(str(e))
        finally:
            self._run_btn.setEnabled(True)
