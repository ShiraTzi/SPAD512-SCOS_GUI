import sys
from PySide6 import QtWidgets, QtCore
from camera_interface import CameraInterface
from widgets.live_view import LiveViewWidget
from widgets.trace_view import TraceViewWidget
from widgets.controls_panel import ControlsPanel
from widgets.scos_view import SCOSViewWidget
from utils.data_handlers import load_csv_data
from utils.SCOS_calculation import find_mask
from widgets.sacs_view import SACsViewWidget
import os
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPAD512S Control Center")
        # self.resize(1200, 800)
        self.mask = None  # Store the current ROI mask for use across pages
        
        # 1. Logic layer
        self.logic = CameraInterface(self)

        # 2. UI widgets
        self.controls = ControlsPanel()
        self.stack = QtWidgets.QStackedWidget()

        # Pass logic to pages that need it
        self.live_page = LiveViewWidget(self.logic)
        self.tpsf_page = TraceViewWidget("TPSF Trace")
        self.irf_page = TraceViewWidget("IRF Trace", is_tpsf=False)
        self.scos_page = SCOSViewWidget(self.logic)  # <-- SCOSViewWidget will now own the calculator
        self.sacs_page = SACsViewWidget(self.logic)  # <-- SACsViewWidget will now own the calculator

        self.stack.addWidget(self.live_page)  # index 0
        self.stack.addWidget(self.tpsf_page)  # index 1
        self.stack.addWidget(self.irf_page)   # index 2
        self.stack.addWidget(self.scos_page)  # index 3
        self.stack.addWidget(self.sacs_page)  # index 4

        # 3. Layout
        main = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(main)

        # --- Sidebar Layout ---
        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)
        sidebar_layout.addWidget(self.controls)
        sidebar_layout.addStretch()

        layout.addWidget(sidebar_widget)   # Sidebar on the left
        layout.addWidget(self.stack)       # Main view on the right

        self.setCentralWidget(main)

        # 4. Signal wiring
        self._connect_signals()

        # View switching logic
        self.stack.currentChanged.connect(self.on_view_changed)

        # Initialize UI state
        self.stack.setCurrentIndex(0)
        self.on_view_changed(0) 

        # 5. Status Timers
        self.spec_timer = QtCore.QTimer(self)
        self.spec_timer.setInterval(2000) 
        self.spec_timer.timeout.connect(self.logic.fetch_camera_specs)
        self.spec_timer.start()

    def _connect_signals(self):
        self.logic.specs_updated.connect(self.controls.update_specs)
        self.controls.view_changed.connect(self.stack.setCurrentIndex)

        # Page Signals
        self.tpsf_page.find_requested.connect(lambda: self._run_task(self.logic.find_TPSF, 1))
        self.tpsf_page.measure_requested.connect(lambda: self._run_task(self.logic.measure_TPSF, 1))
        self.tpsf_page.import_requested.connect(lambda: self._import_file(self.tpsf_page))

        self.irf_page.find_irf_requested.connect(lambda: self._run_task(self.logic.find_IRF, 2))
        self.irf_page.measure_requested.connect(lambda: self._run_task(self.logic.measure_IRF, 2))
        self.irf_page.import_requested.connect(lambda: self._import_file(self.irf_page))

        # SCOS page button signals
        self.scos_page.start_measurement.connect(self.logic.Measure_SCOS_Dispatch)
        self.scos_page.stop_live_requested.connect(self.logic.stop_live_scos)
        self.scos_page.noise_requested.connect(self.logic.measure_and_process_dark_noise_background)
        self.scos_page.roi_requested.connect(self._handle_auto_roi)
        
        # SACS page button signals
        self.sacs_page.start_measurement.connect(self.logic.Measure_SACS_Dispatch)
        self.sacs_page.stop_live_requested.connect(self.logic.stop_live_sacs)
        self.sacs_page.roi_requested.connect(self._handle_auto_roi)
        
        # Logic signals
        self.logic.measurement_started.connect(lambda _: self.controls.set_all_enabled(False))
        self.logic.measurement_finished.connect(self.on_measurement_finished)

        # Calibration signals
        self.controls.calibrate_requested.connect(self.logic.calibrate_noises)
        self.controls.calibrate_ms_requested.connect(self.logic.calibrate_master_slave)
        self.controls.calibrate_breakdown.connect(self.logic.calibrate_breakdown)
        self.controls.check_master_only.stateChanged.connect(self.on_master_only_changed)

        # Measurement Options signals
        self.controls.check_pileup.stateChanged.connect(self.on_pileup_changed)
        self.controls.spin_bit_depth.valueChanged.connect(self.on_bit_depth_changed)

        # Data Path & Profile Loading
        self.controls.save_path_changed.connect(lambda path: setattr(self.logic, 'save_path', path))
        self.controls.btn_load_gate_profile.clicked.connect(self._handle_load_gate)

        # Update the measurement button to use the new workflow
        self.tpsf_page.measure_requested.disconnect() # Remove old simple connection
        self.tpsf_page.measure_requested.connect(self.logic.measure_TPSF)

        self.logic.status_msg.connect(self.statusBar().showMessage)
        self.logic.progress_val.connect(self._update_progress)
        self.logic.tpsf_deconv_ready.connect(
            lambda t, raw, decon, iteration_vec, metrics: self.tpsf_page.plot_trace_with_metrics(t, raw, decon, iteration_vec, metrics)
        )
        self.logic.irf_deconv_ready.connect(
            lambda t, raw, decon, iteration_vec, metrics: self.irf_page.plot_trace_with_metrics(t, raw, decon, iteration_vec, metrics)
        )
        self.logic.tpsf_peak_found.connect(self.scos_page.handle_tpsf_peak)
        self.logic.irf_peak_found.connect(self.scos_page.handle_irf_peak)
        self.logic.roi_found.connect(self.scos_page.update_roi_preview)

        # --- Move integration time and FPS calculator logic to SCOSViewWidget ---
        # Remove integration_time_changed and estimates_updated connections from here

        # self.controls.measure_dark_noise_requested.connect(self.logic.measure_and_process_dark_noise_background)
        self.logic.scos_data_ready.connect(self.scos_page.plot_scos_data)
        self.logic.sacs_data_ready.connect(self.sacs_page.plot_sacs_data)
        # self.logic.clean_scos_plot.connect(self.scos_page.clean_plot)
        
    def _update_progress(self, value):
        if not hasattr(self, 'progress_bar'):
            self.progress_bar = QtWidgets.QProgressDialog("Hardware Operation in Progress...", "Cancel", 0, 100, self)
            self.progress_bar.setWindowModality(QtCore.Qt.WindowModal)
            self.progress_bar.setMinimumDuration(0) # Show immediately
        
        self.progress_bar.setValue(value)
        if value >= 100:
            self.progress_bar.close()

    def _handle_load_gate(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Gate Profile")
        if path:
            t_vec, profile = load_csv_data(path)
            if profile is not None:
                self.logic.gate_profile = profile
                self.logic.gate_timevec = t_vec
                self.logic._resample_gate_profile()
                QtWidgets.QMessageBox.information(self, "Success", "Gate Profile Loaded.")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load gate profile from the selected file.")  

    def on_view_changed(self, index: int):
        """Manages visibility of SCOS labels and starts/stops live view."""
        # Stop all live timers first
        if hasattr(self, '_live_scos_timer') and self._live_scos_timer.isActive():
            self._live_scos_timer.stop()
        self.live_page.stop_live()

        if index == 0:  # Live View
            if hasattr(self.controls, 'set_mode'): self.controls.set_mode("LIVE")
            if self.mask is not None:
                self.live_page.display_roi_mask(self.mask)
            self.live_page.start_live()
        elif index == 1: # TPSF
            if hasattr(self.controls, 'set_mode'): self.controls.set_mode("TPSF")
            if self.mask is not None:
                frame = self.logic.get_live_intensity()
                if frame is not None:
                    self.tpsf_page.update_preview(frame, self.mask)
        elif index == 2: # IRF
            if hasattr(self.controls, 'set_mode'): self.controls.set_mode("IRF")
            if self.mask is not None:
                frame = self.logic.get_live_intensity()
                if frame is not None:
                    self.irf_page.update_preview(frame, self.mask)
        elif index == 3: # SCOS
            if hasattr(self.controls, 'set_mode'): self.controls.set_mode("SCOS")
            if self.mask is not None:
                frame = self.logic.get_live_intensity()
                if frame is not None:
                    self.scos_page.update_preview(frame, self.mask)
            elif hasattr(self.scos_page, 'on_activated'):
                self.scos_page.on_activated()
        elif index == 4: # SACS
            if hasattr(self.controls, 'set_mode'): self.controls.set_mode("SACS")
            if self.mask is not None:
                frame = self.logic.get_live_intensity()
                if frame is not None:
                    self.sacs_page.update_preview(frame, self.mask)
            elif hasattr(self.sacs_page, 'on_activated'):
                self.sacs_page.on_activated()

    def _handle_auto_roi(self):
        frame = self.logic.get_live_intensity()
        if frame is not None:
            mask = find_mask(frame)
            self.mask = mask  # Store it for future use
            self.logic.camera.set_roi_mask(mask)
            self.live_page.display_roi_mask(mask)
            self.scos_page.update_preview(frame, mask)
            self.tpsf_page.update_preview(frame, mask)
            self.irf_page.update_preview(frame, mask)
            
            # SACS uses same mask but displays it as square
            self.sacs_page.update_preview(frame, mask)
            

    # def _handle_scos_analysis(self, mode):
    #     if mode == "noise":
    #         print("Starting Dark Noise Measurement...")
    #         self.logic.measure_and_process_dark_noise_background()
    #     elif mode == "gated":
    #         print("Starting Gated SCOS...")
    #         self.scos_page.clear_plots()  # <-- Clear before sweep
    #         self.logic.Measure_Gated_SCOS()
    #     elif mode == "intensity":
    #         print("Starting Intensity SCOS...")
    #         self.scos_page.clear_plots()  # <-- Clear before sweep
    #         self.logic.Measure_PaLS_iSCOS()
    #     else:
    #         self.statusBar().showMessage("Unknown SCOS mode.")

    def _force_view(self, index):
        self.stack.setCurrentIndex(index)
        self.controls.view_dropdown.setCurrentIndex(index)

    def _run_task(self, task_func, page_index: int):
        self._force_view(page_index)
        task_func()

    def _import_file(self, target_page):

        # Ask user to select a folder
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Measurement Folder")
        if not folder:
            return

        # Load required arrays
        t_path = os.path.join(folder, "t_axis.npy")
        raw_path = os.path.join(folder, "raw_counts.npy")
        decon_path = os.path.join(folder, "deconvolved.npy")
        iter_path = os.path.join(folder, "iteration_vec.npy")
        metrics_path = os.path.join(folder, "metrics.npy")

        if not (os.path.exists(t_path) and os.path.exists(raw_path)):
            QtWidgets.QMessageBox.warning(self, "Error", "Missing t_axis.npy or raw_counts.npy in folder.")
            return

        t = np.load(t_path)
        raw = np.load(raw_path)
        decon = np.load(decon_path) if os.path.exists(decon_path) else None
        iteration_vec = np.load(iter_path) if os.path.exists(iter_path) else None
        metrics = np.load(metrics_path, allow_pickle=True).item() if os.path.exists(metrics_path) else None

        # Plot
        self.stack.setCurrentWidget(target_page)
        if metrics is not None and iteration_vec is not None and decon is not None:
            target_page.plot_trace_with_metrics(t, raw, decon, iteration_vec, metrics)
        else:
            if decon is not None:
                target_page.plot_trace_with_deconv(t, raw, decon)
            else:
                target_page.plot_trace(t, raw)

        # Update peak in SCOS page
        peak_val = t[np.argmax(decon if decon is not None else raw)]
        if target_page is self.tpsf_page:
            self.scos_page.handle_tpsf_peak(peak_val)   # UI label update
            self.logic.tpsf_peak_time = peak_val        # logic storage in ps
            self.logic.tpsf_peak_ns = peak_val / 1000.0 # logic storage in ns for SCOS dispatch
        elif target_page is self.irf_page:
            self.scos_page.handle_irf_peak(peak_val)
            self.logic.irf_peak_time = peak_val

    @QtCore.Slot()
    def on_measurement_finished(self):
        self.controls.set_all_enabled(True)
        if self.stack.currentIndex() == 0:
            self.live_page.start_live()

    def closeEvent(self, event):
        self.logic.close()
        event.accept()

    def on_master_only_changed(self, state):
        self.logic.master_only = bool(state)

    def on_pileup_changed(self, state):
        self.logic.pileup_correction = 1 if state else 0

    def on_bit_depth_changed(self, value):
        self.logic.bit_depth = value

    # def _handle_scos_measurement(self, params):
    #     mode = params.get("mode", "acquire")
    #     measurement_type = params.get("measurement_type", "")
    #     if mode == "acquire":
    #         if measurement_type == "Gated SCOS":
    #             self.logic.Measure_Gated_SCOS()
    #         elif measurement_type == "PaLS-iSCOS":
    #             self.logic.Measure_PaLS_iSCOS()
    #         else:
    #             self.statusBar().showMessage("Unknown measurement type.")
    #     elif mode == "live":
    #         self._start_live_scos(measurement_type)
    #     else:
    #         self.statusBar().showMessage("Unknown SCOS mode.")

    # def _start_live_scos(self, measurement_type):
    #     # Example: start a QTimer that calls a function to acquire and plot SCOS data
    #     if hasattr(self, '_live_scos_timer') and self._live_scos_timer.isActive():
    #         self._live_scos_timer.stop()
    #     self._live_scos_timer = QtCore.QTimer(self)
    #     self._live_scos_timer.timeout.connect(lambda: self._update_live_scos(measurement_type))
    #     self._live_scos_timer.start(500)  # update every 500 ms

    # def _update_live_scos(self, measurement_type):
    #     if measurement_type == "Gated SCOS":
    #         # Acquire and plot live SCOS (do not save)
    #         results = self.logic.get_live_scos_data()  # Implement this in your logic
    #         self.scos_page.plot_scos_data(results['delays'], results['K2'], results['intensity'])
    #     elif measurement_type == "PaLS-iSCOS":
    #          # Acquire and plot live PaLS-iSCOS (do not save)
    #         # Similar for PaLS-iSCOS
    #         results = self.logic.get_live_pals_iscos_data()
    #         self.scos_page.plot_scos_data(results['delays'], results['K2'], results['intensity'])
    #     # Add stop condition if needed

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # --- Dark Mode QSS ---
    dark_qss = """
    QWidget { background-color: #232629; color: #f0f0f0; }
    QMainWindow { background-color: #232629; }
    QPushButton { background-color: #32363a; color: #f0f0f0; border: 1px solid #444; padding: 5px; }
    QPushButton:hover { background-color: #3c4044; }
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox, QDoubleSpinBox {
        background-color: #2b2f33; color: #f0f0f0; border: 1px solid #444;
    }
    QMenuBar, QMenu, QToolBar { background-color: #232629; color: #f0f0f0; }
    QStatusBar { background-color: #232629; color: #f0f0f0; }
    QTabWidget::pane { border: 1px solid #444; }
    QTabBar::tab { background: #32363a; color: #f0f0f0; padding: 6px; }
    QTabBar::tab:selected { background: #44484c; }
    QScrollBar:vertical, QScrollBar:horizontal {
        background: #232629; width: 12px; margin: 0px;
    }
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
        background: #44484c; border-radius: 5px;
    }
    QScrollBar::add-line, QScrollBar::sub-line {
        background: none;
    }
    QCheckBox, QRadioButton { color: #f0f0f0; }
    QGroupBox { border: 1px solid #444; margin-top: 6px; }
    QGroupBox:title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
    QProgressBar {
        background-color: #2b2f33; color: #f0f0f0; border: 1px solid #444;
        border-radius: 5px; text-align: center;
    }
    QProgressBar::chunk { background-color: #0078d7; }
    """
    app.setStyleSheet(dark_qss)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())