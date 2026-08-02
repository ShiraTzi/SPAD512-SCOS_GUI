from PySide6 import QtWidgets, QtCore
from widgets.canvas import MplCanvas
import numpy as np
from utils.SCOS_calculation import find_fft_peak


MIN_STEP_SIZE_PS = 18.6
MIN_GATE_WIDTH_PS = 6000
CAMERA_READOUT_TIME_MS = 6.5

class SCOSViewWidget(QtWidgets.QWidget):
    # -------- Signals --------
    roi_requested = QtCore.Signal()
    start_measurement = QtCore.Signal(dict)  # Emits params dict
    stop_live_requested = QtCore.Signal()
    noise_requested = QtCore.Signal(float, int, int)

    def __init__(self, camera_interface, parent=None):
        super().__init__(parent)
        self.interface = camera_interface
        self.tpsf_peak_time = 0.0
        self.irf_peak_time = 0.0
        self.current_mask = None

        # Main Layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # --- Controls Panel ---
        self.controls_group = QtWidgets.QGroupBox("SCOS Configuration")
        self.controls_group.setFixedWidth(320)
        ctrl_layout = QtWidgets.QVBoxLayout(self.controls_group)
        ctrl_layout.setSpacing(10)

        # 1. Mode Selection
        ctrl_layout.addWidget(self._make_label("1. MEASUREMENT MODE", True))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Gated SCOS (Scan)", "PaLS-iSCOS (Single Gate)"])
        self.mode_combo.currentTextChanged.connect(self._update_visibility)
        ctrl_layout.addWidget(self.mode_combo)

        # 2. ROI
        ctrl_layout.addWidget(self._make_label("2. ROI ALIGNMENT", True))
        self.roi_preview = MplCanvas(width=3, height=3)
        self.roi_preview.ax.axis('off')
        self.roi_preview.setMinimumHeight(150)
        ctrl_layout.addWidget(self.roi_preview)

        btn_roi = QtWidgets.QPushButton("Find ROI (Auto-Detect)")
        btn_roi.clicked.connect(self.roi_requested.emit)
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

        # Estimates
        self.lbl_est_exposure = QtWidgets.QLabel("Est. Exposure: -- us")
        self.lbl_est_fps = QtWidgets.QLabel("Est. Frame Rate: -- fps")
        ctrl_layout.addWidget(self.lbl_est_exposure)
        ctrl_layout.addWidget(self.lbl_est_fps)

        #  Gating Parameters
        # ctrl_layout.addWidget(self._make_label("4. GATING PARAMETERS", True))
        gate_layout = QtWidgets.QGridLayout()

        # Start Offset (Relative to Peak)
        self.lbl_start = QtWidgets.QLabel("Gate Start Offset (ps):")
        self.gate_start_spin = QtWidgets.QDoubleSpinBox()
        self.gate_start_spin.setRange(-20000, 70000)
        self.gate_start_spin.setValue(-500)
        gate_layout.addWidget(self.lbl_start, 0, 0)
        gate_layout.addWidget(self.gate_start_spin, 0, 1)

        # Step Size (Scan only)
        self.lbl_step = QtWidgets.QLabel("Gate Step Size (ps):")
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
        form_layout.addRow(self.lbl_num_steps, self.num_steps_spin)

        self.gate_container = QtWidgets.QWidget()
        self.gate_container.setLayout(gate_layout)
        ctrl_layout.addWidget(self.gate_container)
        
        
        # 4. Calibration (Added back)
        ctrl_layout.addWidget(self._make_label("4. CALIBRATION", True))
        self.btn_clear = QtWidgets.QPushButton("CLEAR PLOTS")
        self.btn_clear.setStyleSheet("height: 30px; background-color: #95a5a6; color: white;")
        self.btn_clear.clicked.connect(self.clear_plots)
        ctrl_layout.addWidget(self.btn_clear)
        

        ctrl_layout.addSpacing(5)

        self.btn_dark = QtWidgets.QPushButton("MEASURE DARK NOISE")
        self.btn_dark.setStyleSheet("height: 30px; background-color: #34495e; color: white;")
        self.btn_dark.clicked.connect(self._emit_dark_noise)
        ctrl_layout.addWidget(self.btn_dark)

        # Peaks and pathlength difference estimates
        self.lbl_tpsf_peak = QtWidgets.QLabel("TPSF Peak: -- ns")
        self.lbl_irf_peak = QtWidgets.QLabel("IRF Peak: -- ns")
        self.lbl_pathlength_diff = QtWidgets.QLabel("Pathlength Diff: -- cm")
        ctrl_layout.addWidget(self.lbl_tpsf_peak)
        ctrl_layout.addWidget(self.lbl_irf_peak)
        ctrl_layout.addWidget(self.lbl_pathlength_diff)

        # Action Buttons
        ctrl_layout.addStretch()

        self.btn_run = QtWidgets.QPushButton("START ACQUISITION")
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; height: 35px;")
        self.btn_run.clicked.connect(lambda: self._emit_measurement("acquire"))
        ctrl_layout.addWidget(self.btn_run)

        self.btn_live = QtWidgets.QPushButton("START LIVE SCOS")
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; height: 35px;")
        self.btn_live.clicked.connect(self._handle_live_toggle)
        ctrl_layout.addWidget(self.btn_live)

        # --- Display Area ---
        self.display_tabs = QtWidgets.QTabWidget()
        self.canvas_k2 = MplCanvas()
        self.canvas_intensity = MplCanvas()
        self.canvas_fft_k2 = MplCanvas()
        self.canvas_fft_int = MplCanvas()
        self.display_tabs.addTab(self.canvas_k2, "Contrast (K²)")
        self.display_tabs.addTab(self.canvas_intensity, "Intensity")
        self.display_tabs.addTab(self.canvas_fft_k2, "FFT (K²)")
        self.display_tabs.addTab(self.canvas_fft_int, "FFT (Intensity)")

        layout.addWidget(self.controls_group)
        layout.addWidget(self.display_tabs)

        # Init State
        self._update_visibility(self.mode_combo.currentText())
        self.integration_time_spin.valueChanged.connect(self._update_estimates)
        self._update_estimates()

    def _emit_dark_noise(self):
        # Try to get bit depth and pileup from the camera interface if possible
        bit_depth = getattr(self.interface, 'bit_depth', 8)
        pileup_correction = getattr(self.interface, 'pileup_correction', 0)
        int_time = self.integration_time_spin.value()
        self.noise_requested.emit(int_time, bit_depth, pileup_correction)

    def _make_label(self, text, bold=False, color=None):
        lbl = QtWidgets.QLabel(text)
        # Assign a color if provided, else cycle through a palette
        palette = ["#e67e22", "#1abc9c", "#9b59b6", "#f39c12", "#3498db", "#e74c3c", "#2ecc71"]
        if not hasattr(self, '_label_color_idx'):
            self._label_color_idx = 0
        if color is None:
            color = palette[self._label_color_idx % len(palette)]
            self._label_color_idx += 1
        style = f"color: {color};"
        if bold:
            style += " font-weight: bold;"
        lbl.setStyleSheet(style)
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
        if checked:
            self.btn_live.setText("STOP LIVE SCOS")
            self.btn_live.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; height: 35px;")
            self._emit_measurement("live")
        else:
            self.btn_live.setText("START LIVE SCOS")
            self.btn_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; height: 35px;")
            self.stop_live_requested.emit()

    def _emit_measurement(self, mode):
        params = {
            "mode": mode,  # 'acquire' or 'live'
            "type": "Gated SCOS" if "Scan" in self.mode_combo.currentText() else "PaLS-iSCOS",
            "int_time": self.integration_time_spin.value(),
            "n_frames": self.frames_per_step_spin.value(),
            "gate_start": self.gate_start_spin.value(),
            "gate_step": self.gate_step_spin.value(),
            "num_steps": self.num_steps_spin.value(),
            "roi": self.current_mask,
            "peak_ns": self.tpsf_peak_time
        }
        self.start_measurement.emit(params)

    def update_preview(self, frame, mask=None):
            """Updates the small preview with the SPAD frame and an ROI overlay."""
            self.roi_preview.ax.clear()
            
            if frame is not None:
                # Display the SPAD frame in grayscale
                self.roi_preview.ax.imshow(frame, cmap='gray', interpolation='nearest')
                
            if mask is not None:
                # Save the mask locally for the measurement logic
                self.current_mask = mask
                # Draw a red outline around the ROI pixels
                # This allows you to see the object inside the ROI
                self.roi_preview.ax.contour(mask, colors='red', linewidths=1)
                
            self.roi_preview.ax.axis('off')
            # self.roi_preview.set_tight_layout()
            self.roi_preview.draw_idle()
    def clean_plot(self):
        """Clears the SCOS plots. Used before starting a new acquisition."""
        self.canvas_k2.ax.clear()
        self.canvas_k2.draw_idle()
        self.canvas_intensity.ax.clear()
        self.canvas_intensity.draw_idle()
        self.canvas_fft_k2.ax.clear()
        self.canvas_fft_k2.draw_idle()
        self.canvas_fft_int.ax.clear()
        self.canvas_fft_int.draw_idle()
        
    def plot_scos_data(self, x_data, k2_data, int_data, xlabel="Gate Delay (ps)", clean_flag=False):
        """Plot time-resolved SCOS curve. Don't clear - accumulate across gate offsets."""
            # Clear for LIVE mode (xlabel contains "LIVE"), keep for scan mode
        if "LIVE" in xlabel or "Single" in xlabel:
            self.canvas_k2.ax.clear()
            self.canvas_intensity.ax.clear()
            self.canvas_fft_k2.ax.clear()
            self.canvas_fft_int.ax.clear()
            
        if clean_flag:
            self.canvas_k2.ax.clear()
            self.canvas_intensity.ax.clear()
            self.canvas_fft_k2.ax.clear()
            self.canvas_fft_int.ax.clear()
            
        # Plot K2
        self.canvas_k2.ax.plot(x_data, k2_data, '-', label=xlabel, linewidth=1.5, alpha=0.7)
        # Neon colors for contrast plot
        self.canvas_k2.ax.set_ylabel("Contrast (K²)", color="#00fff7")  # Neon cyan
        self.canvas_k2.ax.set_xlabel("Time (s)", color="#00ff85")      # Neon green
        self.canvas_k2.ax.tick_params(axis='x', colors="#00ff85")
        self.canvas_k2.ax.tick_params(axis='y', colors="#00fff7")
        self.canvas_k2.ax.legend(fontsize=8, loc='best')
        self.canvas_k2.ax.grid(True, alpha=0.3)
        self.canvas_k2.draw_idle()
        
        # Plot Intensity (log scale)
        self.canvas_intensity.ax.semilogy(x_data, int_data, '-', label=xlabel, linewidth=1.5, alpha=0.7)
        # Neon colors for intensity plot
        self.canvas_intensity.ax.set_ylabel("Intensity (Log)", color="#ff00e1")  # Neon magenta
        self.canvas_intensity.ax.set_xlabel("Time (s)", color="#ffe600")         # Neon yellow
        self.canvas_intensity.ax.tick_params(axis='x', colors="#ffe600")
        self.canvas_intensity.ax.tick_params(axis='y', colors="#ff00e1")
        self.canvas_intensity.ax.legend(fontsize=8, loc='best')
        self.canvas_intensity.ax.grid(True, alpha=0.3)
        self.canvas_intensity.draw_idle()
        
        # plot FFT of K2
        peak_freq, SNR, positive_magnitude, positive_freqs,_ = find_fft_peak(x_data, k2_data)
        self.canvas_fft_k2.ax.clear()
        self.canvas_fft_k2.ax.plot(positive_freqs, positive_magnitude, 'm--', label=f"FFT of K² (Peak: {peak_freq:.2f} Hz , SNR: {SNR:.2f})", linewidth=1, alpha=0.7)
        self.canvas_fft_k2.ax.set_ylabel("FFT(K²)", color='m')
        self.canvas_fft_k2.ax.set_xlabel("Frequency (Hz)", color='#f0f0f0')
        self.canvas_fft_k2.ax.tick_params(axis='y', labelcolor='m')
        self.canvas_fft_k2.ax.tick_params(axis='x', colors='#f0f0f0')
        self.canvas_fft_k2.ax.legend(fontsize=8, loc='upper right')
        self.canvas_fft_k2.ax.grid(True, alpha=0.3)
        self.canvas_fft_k2.draw_idle()

        # plot FFT of intensity
        peak_freq_int, SNR_int, positive_magnitude_int, positive_freqs_int, _ = find_fft_peak(x_data, int_data)
        self.canvas_fft_int.ax.clear()
        self.canvas_fft_int.ax.plot(positive_freqs_int, positive_magnitude_int, 'c--', label=f"FFT of Intensity (Peak: {peak_freq_int:.2f} Hz , SNR: {SNR_int:.2f})", linewidth=1, alpha=0.7)
        self.canvas_fft_int.ax.set_ylabel("FFT(Intensity)", color='c')
        self.canvas_fft_int.ax.set_xlabel("Frequency (Hz)", color='#f0f0f0')
        self.canvas_fft_int.ax.tick_params(axis='y', labelcolor='c')
        self.canvas_fft_int.ax.tick_params(axis='x', colors='#f0f0f0')
        self.canvas_fft_int.ax.legend(fontsize=8, loc='upper right')
        self.canvas_fft_int.ax.grid(True, alpha=0.3)
        self.canvas_fft_int.draw_idle()
        

    def on_activated(self):
        self.roi_requested.emit()

    def handle_tpsf_peak(self, val):
        # Check if val is a vector and take the last element as the peak time, otherwise use val directly
        if isinstance(val, (list, np.ndarray)):
            val = val[-1]
        self.tpsf_peak_time = val
        self.lbl_tpsf_peak.setText(f"TPSF Peak: {val:.2f} ps")
        self.handle_pathlength_diff()

    def handle_irf_peak(self, val):
        #check if val is a vetor and take the last element as the peak time, otherwise use val directly
        if isinstance(val, (list, np.ndarray)):
            val = val[-1]
        self.irf_peak_time = val
        self.lbl_irf_peak.setText(f"IRF Peak: {val:.2f} ps")
        self.handle_pathlength_diff()
    
    def handle_pathlength_diff(self):
        if self.tpsf_peak_time is not None and self.irf_peak_time is not None:
            # Calculate pathlength difference in cm
            time_diff_ps = self.tpsf_peak_time - self.irf_peak_time  # in ps
            time_diff_s = time_diff_ps * 1e-12  # convert to seconds
            speed_of_light_cm_s = 3e10  # speed of light in cm/s
            pathlength_diff_cm = time_diff_s * speed_of_light_cm_s  # in cm
            self.lbl_pathlength_diff.setText(f"Pathlength Diff: {pathlength_diff_cm:.2f} cm")
        else:
            self.lbl_pathlength_diff.setText("Pathlength Diff: -- cm")

    def _handle_run_clicked(self):
        mode = "gated" if "Scan" in self.mode_combo.currentText() else "intensity"
        self.analysis_requested.emit(mode)

    def update_roi_preview(self, mask):
        """Display the ROI mask on the preview canvas."""
        if mask is None:
            self.status_msg.emit("No ROI mask to display.")
            return
        
        self.roi_preview.ax.clear()
        
        # Create a dummy frame or use zeros for display
        dummy_frame = np.zeros((512, 512))
        self.roi_preview.ax.imshow(dummy_frame, cmap='gray')
        
        # Overlay the mask contour
        self.roi_preview.ax.contour(mask, colors='red', linewidths=2)
        self.roi_preview.ax.set_title("Selected ROI Mask")
        self.roi_preview.ax.axis('off')
        self.roi_preview.draw_idle()
        
        # Save the mask for SCOS
        self.current_mask = mask

    def clear_plots(self):
        """Clear the plot canvases before a new sweep."""
        self.canvas_k2.ax.clear()
        self.canvas_intensity.ax.clear()
        self.canvas_k2.draw_idle()
        self.canvas_intensity.draw_idle()
        self.canvas_fft_k2.ax.clear()
        self.canvas_fft_k2.draw_idle()
        self.canvas_fft_int.ax.clear()
        self.canvas_fft_int.draw_idle()