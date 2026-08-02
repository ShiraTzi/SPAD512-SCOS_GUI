from PySide6 import QtCore, QtWidgets
import numpy as np
from widgets.canvas import MplCanvas

class LiveViewWidget(QtWidgets.QWidget):
    def __init__(self, camera_interface, parent=None):
        super().__init__(parent) 
        
        self.interface = camera_interface 
        self._im = None
        self.roi_contour = None 
        self.mask_overlay = None
        
        # Main Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        # 1. Canvas (The Image)
        self.canvas = MplCanvas()
        # Adding with stretch=9 tells the layout to give 90% of space here
        self.layout.addWidget(self.canvas, stretch=9)
        
        # 2. Stats Container (The bottom 10%)
        self.stats_container = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(self.stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        self.info_label = QtWidgets.QLabel("Intensity: N/A | Range: N/A | Cursor: (N/A, N/A)")
        self.stats_label = QtWidgets.QLabel("ROI Mean: N/A | ROI Var: N/A")
        self.mask_status_label = QtWidgets.QLabel("Mask: None")
        
        stats_layout.addWidget(self.info_label)
        stats_layout.addWidget(self.stats_label)
        stats_layout.addWidget(self.mask_status_label)
        
        # Force the stats container to only take the space it needs
        self.stats_container.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        
        # Adding with stretch=1
        self.layout.addWidget(self.stats_container, stretch=1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)    
        
    def on_mouse_move(self, event):
        if event.inaxes == self.canvas.ax and self._im is not None:
            x, y = int(event.xdata), int(event.ydata)
            frame = self._im.get_array()
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                intensity = frame[y, x]
                # Use masked data for range display if mask exists
                stats_frame = frame[self.mask_overlay] if self.mask_overlay is not None else frame
                f_min, f_max = np.min(stats_frame), np.max(stats_frame)
                
                self.info_label.setText(
                    f"Intensity: {intensity:.2f} | Range: [{f_min:.2f}, {f_max:.2f}] | Cursor: ({x}, {y})"
                )    
        
    def display_roi_mask(self, mask):
        """Overlays the ROI boundary on the current canvas."""
        # Clear all contour collections from the axes
        for collection in list(self.canvas.ax.collections):
            collection.remove()
        
        self.roi_contour = None
        
        # Draw the new ROI contour
        self.roi_contour = self.canvas.ax.contour(mask, colors='yellow', linewidths=2)
        self.canvas.draw_idle()
        self.mask_overlay = mask
        num_pixels = np.sum(mask)
        self.mask_status_label.setText(f"Mask: {num_pixels} pixels in ROI")
        
    def refresh(self):
        """Timer-based refresh of the intensity image."""
        if self.interface.is_measuring: 
            return
            
        frame = self.interface.get_live_intensity()
        if frame is None: 
            return

        if self._im is None:
            # initialization
            self._im = self.canvas.ax.imshow(frame, cmap="hot")
            cbar = self.canvas.fig.colorbar(self._im, ax=self.canvas.ax)
            cbar.ax.yaxis.set_tick_params(color='#fff')
            import matplotlib.pyplot as plt
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#fff')
            cbar.set_label('Intensity', color='#fff')
            self.canvas.ax.axis('off')
            # Instead of tight_layout(), use subplots_adjust to leave a specific gap at bottom
            self.canvas.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        else:
            self._im.set_data(frame)
            f_min, f_max = np.min(frame), np.max(frame)
            if f_max > f_min:
                self._im.set_clim(f_min, f_max)
        
        # Update Mean/Var if mask exists
        if self.mask_overlay is not None:
            masked_data = frame[self.mask_overlay]
            if masked_data.size > 0:
                m_val, v_val = np.mean(masked_data), np.var(masked_data)
                self.stats_label.setText(f"ROI Mean: {m_val:.2f} | ROI Var: {v_val:.2f}")
        
        self.canvas.draw_idle()

    def start_live(self): 
        # 100ms = 10Hz. Your code had 10ms, which is too fast for Matplotlib.
        self.timer.start(100) 
        
    def stop_live(self): 
        self.timer.stop()