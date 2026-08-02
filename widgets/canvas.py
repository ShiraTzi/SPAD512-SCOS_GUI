from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#232629")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor="#232629")
        # Set default text and grid color for dark mode
        # Set all axis and label colors for dark mode
        light = "#f0f0f0"
        self.ax.tick_params(colors=light)
        self.ax.xaxis.label.set_color(light)
        self.ax.yaxis.label.set_color(light)
        self.ax.title.set_color(light)
        self.ax.grid(True, color="#444", alpha=0.5)

    def set_labels_light(self):
        """Set all axis, tick, and legend label colors to light gray/white."""
        light = "#f0f0f0"
        self.ax.xaxis.label.set_color(light)
        self.ax.yaxis.label.set_color(light)
        self.ax.title.set_color(light)
        self.ax.tick_params(colors=light)
        legend = self.ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color(light)

    def set_title(self, title, **kwargs):
        # Always use bright color for dark mode
        kwargs.setdefault('color', '#f0f0f0')
        return self.ax.set_title(title, **kwargs)
        self.fig.tight_layout()