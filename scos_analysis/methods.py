# scos_analysis/methods.py
import numpy as np
from utils.SCOS_calculation import SCOS_Calculation

class BaseSCOS:
    def __init__(self, image_data, mask, frame_rate, **kwargs):
        self.image_data = image_data
        self.mask = mask
        self.frame_rate = frame_rate
        self.kwargs = kwargs
        self.results = None

    def get_gate_offset(self):
        """Must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement gate offset calculation")

    def run(self):
        # 1. Determine the gate offset specific to the method
        offset = self.get_gate_offset()
        
        # 2. Run the core SCOS calculation
        self.results = SCOS_Calculation(
            image_data=self.image_data,
            mask=self.mask,
            frame_rate=self.frame_rate,
            gate_offset=offset, # Inject the calculated offset
            **self.kwargs
        )
        return self.results

class GatedSCOS(BaseSCOS):
    def __init__(self, image_data, mask, frame_rate, set_gate, **kwargs):
        super().__init__(image_data, mask, frame_rate, **kwargs)
        self.set_gate = set_gate

    def get_gate_offset(self):
        # Gated SCOS simply uses the fixed gate
        return self.set_gate

class PaLSiSCOS(BaseSCOS):
    def __init__(self, image_data, mask, frame_rate, tpsf_peak, irf_peak, **kwargs):
        super().__init__(image_data, mask, frame_rate, **kwargs)
        self.tpsf_peak = tpsf_peak
        self.irf_peak = irf_peak

    def get_gate_offset(self):
        # PaLS-iSCOS calculates the offset from the IRF
        time_diff = self.irf_peak - self.tpsf_peak
        return time_diff