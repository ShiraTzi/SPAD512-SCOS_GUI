from PySide6 import QtWidgets, QtCore

class ControlsPanel(QtWidgets.QGroupBox):
    """The sidebar containing navigation, hardware status, and measurement triggers."""

    # --- Signals ---
    view_changed = QtCore.Signal(int)
    calibrate_requested = QtCore.Signal()
    calibrate_ms_requested = QtCore.Signal()
    calibrate_breakdown = QtCore.Signal()
    integration_time_changed = QtCore.Signal(float)
    save_path_changed = QtCore.Signal(str)
    # measure_dark_noise_requested = QtCore.Signal()  # Add this signal
    
    def __init__(self, parent=None):
        super().__init__("Measurement Controls", parent)
        self.setFixedWidth(250)

        # Main vertical layout for the sidebar
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 15, 10, 15)
        self.main_layout.setSpacing(10)
        # Make all text in the control panel bright for dark mode
        self.setStyleSheet('''
            QWidget, QLabel, QCheckBox, QRadioButton, QGroupBox, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox {
                color: #f8f8ff;
                font-weight: 500;
            }
            QGroupBox:title { color: #fff; font-weight: bold; }
        ''')

        # 1. --- View Selection ---
        self.main_layout.addWidget(self._make_header("Active View"))
        self.view_dropdown = QtWidgets.QComboBox()
        self.view_dropdown.addItems([
            "Live Intensity", 
            "TPSF Trace", 
            "IRF Trace", 
            "SCOS View",
            "SACS View"
        ])
        self.main_layout.addWidget(self.view_dropdown)
        self.main_layout.addSpacing(5)

        # 2. --- Camera Status Section ---
        self.main_layout.addWidget(self._make_header("Camera Status"))
        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.lbl_volt = QtWidgets.QLabel("Vex: -- | Vq: --")
        self.lbl_temp = QtWidgets.QLabel("Chip Temp: --°C")
        self.lbl_freq = QtWidgets.QLabel("Laser: -- MHz")

        status_labels = [self.lbl_fps, self.lbl_volt, self.lbl_temp, self.lbl_freq]
        for lbl in status_labels:
            lbl.setStyleSheet("font-family: monospace; color: #f8f8ff; font-size: 11px;")
            self.main_layout.addWidget(lbl)

        self.main_layout.addSpacing(5)

        # # 3. --- Exposure & Timing Section ---
        # self.main_layout.addWidget(self._make_header("Timing & Exposure"))
        # self.main_layout.addWidget(QtWidgets.QLabel("Integration Time:"))
        # self.spin_int_time = QtWidgets.QDoubleSpinBox()
        # self.spin_int_time.setRange(0.001, 1000.0)
        # self.spin_int_time.setValue(10.0)
        # self.spin_int_time.setSuffix(" ms")
        # self.main_layout.addWidget(self.spin_int_time)
        # self.main_layout.addSpacing(5)

        # 4. --- Hardware Calibration Section ---
        self.main_layout.addWidget(self._make_header("Hardware Calib"))
        self.btn_calib = QtWidgets.QPushButton("Calibrate Noises")
        self.btn_calib_ms = QtWidgets.QPushButton("Calibrate Master/Slave")
        self.btn_calib_breakdown = QtWidgets.QPushButton("Calibrate Breakdown")


        for btn in [self.btn_calib, self.btn_calib_ms, self.btn_calib_breakdown]:
            self.main_layout.addWidget(btn)
        self.main_layout.addSpacing(5)

        # 4b. --- Measurement Options ---
        self.main_layout.addWidget(self._make_header("Measurement Options"))

        self.check_master_only = QtWidgets.QCheckBox("Measure Only Master")
        self.check_master_only.setChecked(False)
        self.main_layout.addWidget(self.check_master_only)

        self.check_pileup = QtWidgets.QCheckBox("Enable Pileup Correction")
        self.check_pileup.setChecked(False)
        self.main_layout.addWidget(self.check_pileup)

        self.main_layout.addWidget(QtWidgets.QLabel("Bit Depth:"))
        self.spin_bit_depth = QtWidgets.QSpinBox()
        self.spin_bit_depth.setRange(8, 12)
        self.spin_bit_depth.setValue(8)
        self.spin_bit_depth.setSuffix(" bit")
        self.main_layout.addWidget(self.spin_bit_depth)
        self.main_layout.addStretch()

        # 5. --- Gate Profile Section ---
        self.main_layout.addWidget(self._make_header("Gate Profile"))
        self.btn_load_gate_profile = QtWidgets.QPushButton("Load Gate Profile")
        self.main_layout.addWidget(self.btn_load_gate_profile)

        # 6. --- Dark Noise Measurement ---
        # self.main_layout.addWidget(self._make_header("Dark Noise"))
        # self.btn_measure_dark_noise = QtWidgets.QPushButton("Measure Dark Noise")
        # self.main_layout.addWidget(self.btn_measure_dark_noise)
        # self.btn_measure_dark_noise.clicked.connect(self.measure_dark_noise_requested.emit)
        # --- Internal Signal Connections ---
        self.view_dropdown.currentIndexChanged.connect(self.view_changed.emit)
        self.btn_calib.clicked.connect(self.calibrate_requested.emit)
        self.btn_calib_ms.clicked.connect(self.calibrate_ms_requested.emit)
        self.btn_calib_breakdown.clicked.connect(self.calibrate_breakdown.emit)
        # self.spin_int_time.valueChanged.connect(self.integration_time_changed.emit)

        # 7. --- File Management ---
        self.main_layout.addWidget(self._make_header("Data Management"))
        self.btn_set_path = QtWidgets.QPushButton("Set Save Directory")
        self.lbl_save_path = QtWidgets.QLabel("Path: Not Set")
        self.lbl_save_path.setWordWrap(True)
        self.lbl_save_path.setStyleSheet("font-size: 10px; color: #f8f8ff;")

        self.main_layout.addWidget(self.btn_set_path)
        self.main_layout.addWidget(self.lbl_save_path)

        self.btn_set_path.clicked.connect(self._select_save_path)

    def _make_header(self, text):
        """Helper to create consistent section headers."""
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("""
            font-weight: bold;
            color: #fff;
            border-bottom: 1px solid #888;
            padding-bottom: 2px;
            margin-top: 5px;
        """)
        return lbl

    def update_specs(self, data):
        """Updates the hardware status labels via signals from the logic layer."""
        self.lbl_fps.setText(f"FPS: {data.get('fps', '--')}")
        self.lbl_volt.setText(f"Vex: {data.get('vex', '--')}V | Vq: {data.get('vq', '--')}V")
        self.lbl_temp.setText(f"Chip: {data.get('temp', '--')}°C")
        self.lbl_freq.setText(f"Laser: {data.get('laser', '--')} MHz")

    def set_all_enabled(self, enabled: bool):
        """Disables all interactive elements during a blocking measurement."""
        for btn in self.findChildren(QtWidgets.QPushButton):
            btn.setEnabled(enabled)
        self.view_dropdown.setEnabled(enabled)
        # self.spin_int_time.setEnabled(enabled)

    def set_mode(self, mode_name: str):
        """Optional: Can be used to change UI style based on mode."""
        pass

    def _select_save_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if path:
            self.lbl_save_path.setText(f"Path: {path}")
            self.save_path_changed.emit(path)