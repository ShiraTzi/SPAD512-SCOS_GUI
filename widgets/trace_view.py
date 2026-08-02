from PySide6 import QtWidgets, QtCore

from utils.TPSF_calculation import calculate_FWHM
from .canvas import MplCanvas

class TraceViewWidget(QtWidgets.QWidget):
    # Signals for the new buttons
    measure_requested = QtCore.Signal()
    import_requested = QtCore.Signal()
    find_requested = QtCore.Signal() # Only used for TPSF
    gate_profile_requested = QtCore.Signal()
    find_irf_requested = QtCore.Signal()

    def __init__(self, title="Trace", is_tpsf=True, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)

        # --- Sidebar for this specific view ---
        self.sidebar = QtWidgets.QGroupBox(f"{title} Controls")
        side_layout = QtWidgets.QVBoxLayout(self.sidebar)

        # Small ROI Preview Window
        side_layout.addWidget(QtWidgets.QLabel("ROI Preview:"))
        self.roi_preview = MplCanvas(width=3, height=3)
        self.roi_preview.ax.axis('off')
        side_layout.addWidget(self.roi_preview)

        # Buttons moved from the main sidebar
        if is_tpsf:
            self.btn_find = QtWidgets.QPushButton("Find TPSF")
            self.btn_find.clicked.connect(self.find_requested.emit)
            side_layout.addWidget(self.btn_find)
        else:
            self.btn_find_irf = QtWidgets.QPushButton("Find IRF")
            self.btn_find_irf.clicked.connect(self.find_irf_requested.emit)
            side_layout.addWidget(self.btn_find_irf)

        self.btn_measure = QtWidgets.QPushButton(f"Measure {title.split()[0]}")
        self.btn_import = QtWidgets.QPushButton(f"Import {title.split()[0]}")
        
        self.btn_measure.clicked.connect(self.measure_requested.emit)
        self.btn_import.clicked.connect(self.import_requested.emit)

        side_layout.addWidget(self.btn_measure)
        side_layout.addWidget(self.btn_import)
        side_layout.addStretch()

        # Main Plot
        self.trace_canvas = MplCanvas()
        
        layout.addWidget(self.sidebar, 1)
        layout.addWidget(self.trace_canvas, 3)

        self.title = title  # Store the title for later use

    def update_preview(self, frame, mask=None):
        """Updates the small ROI window."""
        self.roi_preview.ax.clear()
        self.roi_preview.ax.imshow(frame, cmap='hot')
        if mask is not None:
            self.roi_preview.ax.contour(mask, colors='yellow', linewidths=1)
        self.roi_preview.ax.axis('off')
        self.roi_preview.draw_idle()

    def plot_trace(self, x, y):
        self.trace_canvas.ax.clear()
        self.trace_canvas.ax.plot(x, y)
        self.trace_canvas.ax.set_title(f"{self.title} Trace")
        self.trace_canvas.ax.set_xlabel("Time (ns)")
        self.trace_canvas.ax.set_ylabel("Counts")
        self.trace_canvas.ax.grid(True, alpha=0.3)
        self.trace_canvas.draw_idle()

    def plot_trace_with_metrics(self, t_axis, raw, deconvolved, iteration_vec, metrics):
        """Plots raw, deconvolved, FWHM, and MSE in a 2x2 grid.
        t_axis in ps
        """
        t_axis_ns = t_axis / 1000  # Convert to ns for plotting
        self.trace_canvas.figure.clear()
        
        # Create 2x2 subplot grid
        axes = self.trace_canvas.figure.subplots(2, 2)
        
        # Top-left: Raw trace
        axes[0, 0].plot(t_axis_ns, raw, 'b-', linewidth=1)
        axes[0, 0].set_title("Raw Trace")
        axes[0, 0].set_xlabel("Time (ns)")
        axes[0, 0].set_ylabel("Counts")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-right: Deconvolved trace
        axes[0, 1].plot(t_axis_ns, deconvolved, 'r-', linewidth=1)
        axes[0, 1].set_title(f"Deconvolved (FWHM: {metrics['best_fwhm']:.2f} ps)")
        axes[0, 1].set_xlabel("Time (ns)")
        axes[0, 1].set_ylabel("Counts")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom-left: MSE vs iterations
        iter_list = list(iteration_vec)
        axes[1, 0].plot(iter_list, metrics['mse'], 'g-', marker='o', markersize=3)
        axes[1, 0].axvline(metrics['best_iterations'], color='red', linestyle='--', alpha=0.7, label=f"Best: {metrics['best_iterations']}")
        axes[1, 0].set_title(f"MSE (min: {metrics['best_mse']:.4f})")
        axes[1, 0].set_xlabel("Iterations")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bottom-right: FWHM vs iterations
        axes[1, 1].plot(iter_list, metrics['fwhm'], 'purple', marker='s', markersize=3)
        axes[1, 1].axvline(metrics['best_iterations'], color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title(f"FWHM vs Iterations")
        axes[1, 1].set_xlabel("Iterations")
        axes[1, 1].set_ylabel("FWHM (ps)")
        axes[1, 1].grid(True, alpha=0.3)
        
        self.trace_canvas.figure.tight_layout()
        self.trace_canvas.draw()
        
    def plot_trace_with_deconv(self, t_axis, raw, deconvolved):
        
        #calculate FWHM for the deconvolved trace
        best_fwhm = calculate_FWHM(deconvolved, t_axis)
        """Plots raw and deconvolved traces with metrics in the title."""
        t_axis_ns = t_axis / 1000  # Convert to ns for plotting 
        self.trace_canvas.ax.clear()
        self.trace_canvas.ax.plot(t_axis_ns, raw, 'b-', label='Raw', linewidth=1)
        self.trace_canvas.ax.plot(t_axis_ns, deconvolved, 'r-', label='Deconvolved', linewidth=1)
        self.trace_canvas.ax.set_title(f"Deconvolved Trace (FWHM: {best_fwhm:.2f} ps)")
        self.trace_canvas.ax.set_xlabel("Time (ns)")    
        self.trace_canvas.ax.set_ylabel("Counts")
        self.trace_canvas.ax.legend()
        self.trace_canvas.ax.grid(True, alpha=0.3)
        self.trace_canvas.draw_idle()
            
