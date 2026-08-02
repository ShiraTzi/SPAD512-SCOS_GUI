from PySide6 import QtWidgets, QtCore
from widgets.canvas import MplCanvas
import numpy as np
from utils.SCOS_calculation import find_fft_peak
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from utils.SCOS_calculation import find_mask
from utils.SACS_Calculations import square_roi_from_mask

MIN_STEP_SIZE_PS = 18.6
MIN_GATE_WIDTH_PS = 6000
CAMERA_READOUT_TIME_MS = 6.5

class SACsViewWidget(QtWidgets.QWidget):
    # -------- Signals --------
    roi_requested = QtCore.Signal()
    start_measurement = QtCore.Signal(dict)
    stop_live_requested = QtCore.Signal()

    def __init__(self, camera_interface, parent=None):
        super().__init__(parent)
        self.interface = camera_interface
        self.tpsf_peak_time = 0.0
        self.irf_peak_time = 0.0
        self.current_mask = None
        self.is_live_running = False  # ← Add this flag

        # Main Layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # --- Controls Panel ---
        self.controls_group = QtWidgets.QGroupBox("SACS Configuration")
        self.controls_group.setFixedWidth(320)
        ctrl_layout = QtWidgets.QVBoxLayout(self.controls_group)
        ctrl_layout.setSpacing(10)

        # 1. Mode Selection
        ctrl_layout.addWidget(self._make_label("1. MEASUREMENT MODE", True))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["SACS Scan", "SACS Single Gate"])
        self.mode_combo.currentTextChanged.connect(self._update_visibility)
        ctrl_layout.addWidget(self.mode_combo)

        # 2. ROI
        ctrl_layout.addWidget(self._make_label("2. ROI ALIGNMENT", True))
        self.roi_preview = MplCanvas(width=3, height=3)
        self.roi_preview.ax.axis('off')
        self.roi_preview.setMinimumHeight(150)
        ctrl_layout.addWidget(self.roi_preview)

        btn_roi = QtWidgets.QPushButton("Find ROI (Auto-Detect)")
        btn_roi.clicked.connect(self._handle_roi_request)
        ctrl_layout.addWidget(btn_roi)

        # 3. Timing & Frames
        ctrl_layout.addWidget(self._make_label("3. TIMING & FRAMES", True))
        form_layout = QtWidgets.QFormLayout()

        self.integration_time_spin = QtWidgets.QDoubleSpinBox()
        self.integration_time_spin.setRange(0.1, 1000.0)
        self.integration_time_spin.setValue(10.0)
        self.integration_time_spin.setSuffix(" ms")
        form_layout.addRow("Int. Time:", self.integration_time_spin)

        self.frames_per_step_spin = QtWidgets.QSpinBox()
        self.frames_per_step_spin.setRange(10, 10000)
        self.frames_per_step_spin.setValue(100)
        self.frames_per_step_spin.setToolTip("Number of frames to acquire at each gate position")
        form_layout.addRow("Frames/Step:", self.frames_per_step_spin)

        ctrl_layout.addLayout(form_layout)

        # Speckle Size (for SACS coherence calculation)
        self.speckle_size_spin = QtWidgets.QSpinBox()
        self.speckle_size_spin.setRange(1, 50)
        self.speckle_size_spin.setValue(1)
        self.speckle_size_spin.setToolTip("Speckle size for autocorrelation calculation")
        form_layout.addRow("Speckle Size (px):", self.speckle_size_spin)
        
        # Estimates
        self.lbl_est_exposure = QtWidgets.QLabel("Est. Exposure: -- us")
        self.lbl_est_fps = QtWidgets.QLabel("Est. Frame Rate: -- fps")
        ctrl_layout.addWidget(self.lbl_est_exposure)
        ctrl_layout.addWidget(self.lbl_est_fps)

        # 4. Gating Parameters
        ctrl_layout.addWidget(self._make_label("4. GATING PARAMETERS", True))
        gate_layout = QtWidgets.QGridLayout()

        # Start Offset (Relative to Peak)
        self.lbl_start = QtWidgets.QLabel("Start Offset (ps):")
        self.gate_start_spin = QtWidgets.QDoubleSpinBox()
        self.gate_start_spin.setRange(-20000, 70000)
        self.gate_start_spin.setValue(-500)
        gate_layout.addWidget(self.lbl_start, 0, 0)
        gate_layout.addWidget(self.gate_start_spin, 0, 1)

        # Step Size (Scan only)
        self.lbl_step = QtWidgets.QLabel("Step Size (ps):")
        self.gate_step_spin = QtWidgets.QDoubleSpinBox()
        self.gate_step_spin.setRange(MIN_STEP_SIZE_PS, 10000)
        self.gate_step_spin.setValue(500)
        gate_layout.addWidget(self.lbl_step, 1, 0)
        gate_layout.addWidget(self.gate_step_spin, 1, 1)

        # Number of Steps
        self.lbl_num_steps = QtWidgets.QLabel("Num Steps:")
        self.num_steps_spin = QtWidgets.QSpinBox()
        self.num_steps_spin.setRange(1, 500)
        self.num_steps_spin.setValue(10)
        gate_layout.addWidget(self.lbl_num_steps, 2, 0)
        gate_layout.addWidget(self.num_steps_spin, 2, 1)

        self.gate_container = QtWidgets.QWidget()
        self.gate_container.setLayout(gate_layout)
        ctrl_layout.addWidget(self.gate_container)

        # Peaks and pathlength difference estimates
        self.lbl_tpsf_peak = QtWidgets.QLabel("TPSF Peak: -- ns")
        self.lbl_irf_peak = QtWidgets.QLabel("IRF Peak: -- ns")
        ctrl_layout.addWidget(self.lbl_tpsf_peak)
        ctrl_layout.addWidget(self.lbl_irf_peak)

        # Action Buttons
        ctrl_layout.addStretch()

        self.btn_clear = QtWidgets.QPushButton("CLEAR PLOTS")
        self.btn_clear.setStyleSheet("height: 30px; background-color: #95a5a6; color: white;")
        self.btn_clear.clicked.connect(self.clear_plots)
        ctrl_layout.addWidget(self.btn_clear)

        self.btn_run = QtWidgets.QPushButton("START ACQUISITION")
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; height: 35px;")
        self.btn_run.clicked.connect(lambda: self._emit_measurement("acquire"))
        ctrl_layout.addWidget(self.btn_run)

        self.btn_live = QtWidgets.QPushButton("START LIVE SACS")
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; height: 35px;")
        self.btn_live.clicked.connect(self._handle_live_toggle)
        ctrl_layout.addWidget(self.btn_live)

        # --- Display Area ---
        self.display_tabs = QtWidgets.QTabWidget()
        self.canvas_size = MplCanvas()
        self.canvas_peak = MplCanvas()
        self.canvas_fft_size = MplCanvas()
        self.canvas_fft_peak = MplCanvas()
        
        self.display_tabs.addTab(self.canvas_size, "Speckle Size")
        self.display_tabs.addTab(self.canvas_peak, "Peak Value")
        self.display_tabs.addTab(self.canvas_fft_size, "FFT (Size)")
        self.display_tabs.addTab(self.canvas_fft_peak, "FFT (Peak)")

        layout.addWidget(self.controls_group)
        layout.addWidget(self.display_tabs)

        # Init State
        self._update_visibility(self.mode_combo.currentText())
        self.integration_time_spin.valueChanged.connect(self._update_estimates)
        self._update_estimates()

    def _make_label(self, text, bold=False):
        lbl = QtWidgets.QLabel(text)
        if bold:
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
        return lbl

    def _update_visibility(self, text):
        is_scan = "Scan" in text
        self.lbl_step.setVisible(is_scan)
        self.gate_step_spin.setVisible(is_scan)
        self.lbl_num_steps.setVisible(is_scan)
        self.num_steps_spin.setVisible(is_scan)

    def _update_estimates(self):
        int_time_ms = self.integration_time_spin.value()
        exposure_us = int_time_ms * 1000.0
        fps = 1000.0 / max(int_time_ms, CAMERA_READOUT_TIME_MS)
        self.lbl_est_exposure.setText(f"Est. Exposure: {exposure_us:.1f} us")
        self.lbl_est_fps.setText(f"Est. Frame Rate: {fps:.3f} fps")

    def _handle_live_toggle(self, checked):
        """Toggle live SACS on/off, updating button text."""
        if checked:
            self.is_live_running = True
            self.btn_live.setText("STOP LIVE SACS")
            self._emit_measurement("live")
        else:
            self.is_live_running = False
            self.btn_live.setText("START LIVE SACS")
            self.stop_live_requested.emit()

    def _handle_roi_request(self):
        """Request ROI independently - won't affect SCOS page."""
        self.roi_requested.emit()

    def _emit_measurement(self, mode):
        params = {
            "mode": mode,
            "type": "SACS Scan" if "Scan" in self.mode_combo.currentText() else "SACS Single Gate",
            "int_time": self.integration_time_spin.value(),
            "n_frames": self.frames_per_step_spin.value(),
            "gate_start": self.gate_start_spin.value(),
            "gate_step": self.gate_step_spin.value(),
            "num_steps": self.num_steps_spin.value(),
            "roi": self.current_mask,
            "peak_ns": self.tpsf_peak_time,
            "speckle_size": self.speckle_size_spin.value()  
        }
        self.start_measurement.emit(params)

    def update_preview(self, frame, mask=None):
        """Updates the small preview with the SPAD frame and square ROI overlay (SACS-specific)."""
        self.roi_preview.ax.clear()
        
        if frame is not None:
            self.roi_preview.ax.imshow(frame, cmap='gray')
            
        if mask is not None:
            self.current_mask = mask
            try:
                # Get square ROI bounds from mask
                y0, y1, x0, x1 = square_roi_from_mask(mask)
                
                # Draw square rectangle
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, 
                            fill=False, edgecolor='red', linewidth=2)
                self.roi_preview.ax.add_patch(rect)
            except Exception as e:
                print(f"Error drawing square ROI: {e}")
        
        self.roi_preview.ax.set_title("SACS ROI Preview (Square)")
        self.roi_preview.ax.axis('off')
        self.roi_preview.draw()

    def plot_sacs_data(self, x_data, size_data, peak_data, label="Gate Delay (ps)", clean_flag=False):
        """Plot SACS metrics (Size and Peak Value) - similar to SCOS with FFT peaks."""
        # Clear for LIVE mode, keep for scan mode
        if "LIVE" in label or "Single" in label:
            self.canvas_size.ax.clear()
            self.canvas_peak.ax.clear()
            self.canvas_fft_size.ax.clear()
            self.canvas_fft_peak.ax.clear()
        
        if clean_flag:
            self.canvas_size.ax.clear()
            self.canvas_peak.ax.clear()
            self.canvas_fft_size.ax.clear()
            self.canvas_fft_peak.ax.clear()

        # Plot Size Values
        self.canvas_size.ax.plot(x_data, size_data, '-', label=label, linewidth=1.5, alpha=0.7)
        self.canvas_size.ax.set_ylabel("Speckle Size (pixels)")
        self.canvas_size.ax.set_xlabel("Time (frame)")
        self.canvas_size.ax.set_title("SACS: Speckle Size")
        self.canvas_size.ax.legend(fontsize=8, loc='best')
        self.canvas_size.ax.grid(True, alpha=0.3)
        self.canvas_size.draw_idle()

        # Plot Peak Values
        self.canvas_peak.ax.plot(x_data, peak_data, '-', label=label, linewidth=1.5, alpha=0.7)
        self.canvas_peak.ax.set_ylabel("Peak Value")
        self.canvas_peak.ax.set_xlabel("Time (frame)")
        self.canvas_peak.ax.set_title("SACS: Peak Value")
        self.canvas_peak.ax.legend(fontsize=8, loc='best')
        self.canvas_peak.ax.grid(True, alpha=0.3)
        self.canvas_peak.draw_idle()

        # FFT of Size with peak detection and SNR
        if len(size_data) > 1:
            peak_freq_size, SNR_size, positive_magnitude_size, positive_freqs_size, _ = find_fft_peak(x_data, size_data)
            self.canvas_fft_size.ax.clear()
            self.canvas_fft_size.ax = self.canvas_fft_size.ax.twinx()
            self.canvas_fft_size.ax.plot(positive_freqs_size, positive_magnitude_size, 'b--', 
                                          label=f"FFT Size (Peak: {peak_freq_size:.2f} Hz, SNR: {SNR_size:.2f})", 
                                          linewidth=1, alpha=0.7)
            self.canvas_fft_size.ax.set_ylabel("FFT(Size)", color='b')
            self.canvas_fft_size.ax.tick_params(axis='y', labelcolor='b')
            self.canvas_fft_size.ax.set_xlabel("Frequency (Hz)")
            self.canvas_fft_size.ax.set_title("FFT: Speckle Size")
            self.canvas_fft_size.ax.legend(fontsize=8, loc='upper right')
            self.canvas_fft_size.ax.grid(True, alpha=0.3)
            self.canvas_fft_size.draw_idle()

        # FFT of Peak with peak detection and SNR
        if len(peak_data) > 1:
            peak_freq_peak, SNR_peak, positive_magnitude_peak, positive_freqs_peak, _ = find_fft_peak(x_data, peak_data)
            self.canvas_fft_peak.ax.clear()
            self.canvas_fft_peak.ax = self.canvas_fft_peak.ax.twinx()
            self.canvas_fft_peak.ax.plot(positive_freqs_peak, positive_magnitude_peak, 'g--', 
                                          label=f"FFT Peak (Peak: {peak_freq_peak:.2f} Hz, SNR: {SNR_peak:.2f})", 
                                          linewidth=1, alpha=0.7)
            self.canvas_fft_peak.ax.set_ylabel("FFT(Peak)", color='g')
            self.canvas_fft_peak.ax.tick_params(axis='y', labelcolor='g')
            self.canvas_fft_peak.ax.set_xlabel("Frequency (Hz)")
            self.canvas_fft_peak.ax.set_title("FFT: Peak Value")
            self.canvas_fft_peak.ax.legend(fontsize=8, loc='upper right')
            self.canvas_fft_peak.ax.grid(True, alpha=0.3)
            self.canvas_fft_peak.draw_idle()

    def handle_tpsf_peak(self, val):
        """Update TPSF peak label."""
        self.tpsf_peak_time = val
        self.lbl_tpsf_peak.setText(f"TPSF Peak: {val:.2f} ns")

    def handle_irf_peak(self, val):
        """Update IRF peak label."""
        self.irf_peak_time = val
        self.lbl_irf_peak.setText(f"IRF Peak: {val:.2f} ns")

    def clear_plots(self):
        """Clear all plot areas."""
        for canvas in [self.canvas_size, self.canvas_peak, self.canvas_fft_size, self.canvas_fft_peak]:
            canvas.ax.clear()
            canvas.ax.set_title("")
            canvas.draw()

    def on_activated(self):
        """Called when page is shown."""
        pass
    
    def update_preview(self, frame, mask=None):
        """Updates the small preview with the SPAD frame and square ROI overlay (SACS-specific)."""
        self.roi_preview.ax.clear()
        
        if frame is not None:
            self.roi_preview.ax.imshow(frame, cmap='gray')
            
        if mask is not None:
            self.current_mask = mask
            # Get square ROI bounds from mask
            y0, y1, x0, x1 = square_roi_from_mask(mask)
            
            # Draw square
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, 
                                fill=False, edgecolor='red', linewidth=2)
            self.roi_preview.ax.add_patch(rect)
        
        self.roi_preview.ax.set_title("SACS ROI Preview (Square)")
        self.roi_preview.draw()
    def _emit_measurement(self, mode):
        params = {
            "mode": mode,
            "type": "SACS Scan" if "Scan" in self.mode_combo.currentText() else "SACS Single Gate",
            "int_time": self.integration_time_spin.value(),
            "n_frames": self.frames_per_step_spin.value(),
            "gate_start": self.gate_start_spin.value(),
            "gate_step": self.gate_step_spin.value(),
            "num_steps": self.num_steps_spin.value(),
            "roi": self.current_mask,  # ← Square ROI mask
            "peak_ns": self.tpsf_peak_time,
            "speckle_size": self.speckle_size_spin.value()
        }
        self.start_measurement.emit(params)
        
    def _handle_live_toggle(self, checked):
        """Toggle live SACS on/off, updating button text."""
        if checked:
            self.is_live_running = True
            self.interface.is_live_sacs_running = True  # ← Set camera flag
            self.btn_live.setText("STOP LIVE SACS")
            self._emit_measurement("live")
        else:
            self.is_live_running = False
            self.interface.is_live_sacs_running = False  # ← Clear camera flag
            self.btn_live.setText("START LIVE SACS")
            self.stop_live_requested.emit()