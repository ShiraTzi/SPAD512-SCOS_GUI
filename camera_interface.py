import os
from attrs import inspect
import numpy as np
import json
import matplotlib.pyplot as plt
import inspect
from scipy.interpolate import interp1d
from pathlib import Path
from time import time, perf_counter
from PySide6 import QtCore
from camera_wrapper import SPAD_Camera
from utils.SCOS_calculation import calculate_dark_noise, SCOS_Calculation, find_fft_peak
from utils.TPSF_calculation import TPSF_deconvolution_multiple_iterations 
from utils.SACS_Calculations import speckle_size_gs_framewise
from time import perf_counter

CONFIG_PATH = Path("config.json")  # choose your config path
MIN_STEP_SIZE = 18.6  # in ps
GATE_SIZE_FOR_SCOS = 10000  # in ps
GATE_SIZE_FOR_SACS =10000  # in ps, can be same as SCOS or different depending on needs
GATE_SIZE_TPSF = 6000  # in ps

class MeasurementWorker(QtCore.QObject):
    """
    Generic worker to run any hardware-blocking function in a background thread.
    """
    finished = QtCore.Signal(str, object) 
    error = QtCore.Signal(str)
    progress_val = QtCore.Signal(int)
    def __init__(self, task_id, func, args=(), kwargs={}):
        super().__init__()
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        # print(f"Worker started for task: {self.task_id}")
        try:
            # Only add progress_callback if the function signature accepts it
            sig = inspect.signature(self.func)
            if 'progress_callback' in sig.parameters:
                self.kwargs['progress_callback'] = self.progress_val.emit
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(self.task_id, result)
        except Exception as e:
            print(f"Error occurred in task {self.task_id}: {e}")
            self.error.emit(str(e))

class CameraInterface(QtCore.QObject):
    # --- Signals ---
    connection_status = QtCore.Signal(bool, str)
    measurement_started = QtCore.Signal(str)  
    measurement_finished = QtCore.Signal(str) 
    specs_updated = QtCore.Signal(dict)
    estimates_updated = QtCore.Signal(dict)
    status_msg = QtCore.Signal(str)
    progress_val = QtCore.Signal(int)
    roi_found = QtCore.Signal(np.ndarray)
    tpsf_data_ready = QtCore.Signal(np.ndarray, np.ndarray)
    irf_data_ready = QtCore.Signal(np.ndarray, np.ndarray)
    tpsf_deconv_ready = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray, object, dict)
    irf_deconv_ready = QtCore.Signal(object, object, object, object, dict)
    tpsf_peak_found = QtCore.Signal(float)
    irf_peak_found = QtCore.Signal(float)
    scos_data_ready = QtCore.Signal(object, object, object, str, bool)
    clean_scos_plot = QtCore.Signal()  # Signal to clear SCOS plots
    sacs_data_ready = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray, str, bool)  # time, size, peak, label, clean_flag
    
      
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = SPAD_Camera()
        
        # Internal State & Data Persistence
        self.is_measuring = False
        self.save_path = ""
        self.gate_profile = None   # For IRF storage
        self.gate_timevec = None   # For IRF storage
        self.found_offset = None   # Result of find_TPSF
        self.found_irf_offset = None   # Result of find_IRF
        self.found_gate_steps = None
        self.found_irf_gate_steps = None
        self.master_only = False
        self.pileup_correction = 0  # 0 = disabled, 1 = enabled
        self.bit_depth = 8  # Valid values: 8-12
        self.speckle_size = 1 # in pixels, for SACS speckle coherence calculation
        # self.btn_run.clicked.connect(self._handle_run_clicked)

        
        # Threading references
        self._thread = None
        self._worker = None
        
        self.tpsf_peak_time = None  # To store TPSF peak time for SCOS
        self.irf_peak_time = None
        
        self.backgroundImg = np.zeros((512, 512))  # Placeholder for dark noise background
        self.darkVarPerWindow = 0

        # Save path
        self.gate_offsets = {"tpsf": None, "irf": None}
        self._load_gate_offsets()
        self.last_roi_mask = None
        
        # self.camera_gain = 0.126  # or your gain value
        self.camera_gain = 1  # or your gain value


        # Add in __init__
        self.tpsf_peak_ns = 0.0  # Store peak in ns for SCOS
        
    def _handle_run_clicked(self):
        # print("CameraInterface:_handle_run_clicked called.")
        """Decides which measurement worker to trigger based on the UI mode."""
        if "Scan" in self.mode_combo.currentText():
            self.analysis_requested.emit("GATED_SWEEP")
        else:
            self.analysis_requested.emit("SINGLE_GATE_SCOS")
            
    def _load_gate_offsets(self):
        # print("CameraInterface: _load_gate_offsets called.")
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                self.gate_offsets.update(data.get("gate_offsets", {}))
            except Exception:
                pass  # ignore corrupt file 


    def _resample_gate_profile(self):
        # print("CameraInterface: _resample_gate_profile called.")
        """Resample gate profile to match hardware min gate step."""
        if self.gate_profile is None or self.gate_timevec is None:
            return
        
        # Get min gate step from hardware specs
        min_gate_step = MIN_STEP_SIZE/1e3  # in ns
        
        # Create new time vector with min_gate_step spacing
        t_start = self.gate_timevec[0]
        t_end = self.gate_timevec[-1]
        new_time = np.arange(t_start, t_end, min_gate_step)
        
        # Interpolate profile to new time base
        interpolator = interp1d(self.gate_timevec, self.gate_profile, 
                            kind='linear', bounds_error=False, fill_value=0)
        resampled_profile = interpolator(new_time)
        # Update stored values
        self.gate_timevec = new_time
        self.gate_profile = resampled_profile 
        
        
    def _save_gate_offset(self, mode, value):
        # print(f"CameraInterface: _save_gate_offset called for mode {mode} with value {value}.")
        self.gate_offsets[mode] = value
        data = {"gate_offsets": self.gate_offsets}
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
        
    def _save_gate_steps(self, mode, value):
        # print(f"CameraInterface: _save_gate_steps called for mode {mode} with value {value}.")
        self.gate_offsets[mode+"_steps"] = value
        data = {"gate_offsets": self.gate_offsets}
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
        
    def _run_in_background(self, task_id, func, *args, **kwargs):
        # print(f"CameraInterface: _run_in_background called for task {task_id}.")
        """Standardizes background task execution to keep UI responsive."""
        if self.is_measuring: return
        self.is_measuring = True
        self.measurement_started.emit(task_id)
        
        self._thread = QtCore.QThread()
        self._worker = MeasurementWorker(task_id, func, args, kwargs)
        self._worker.moveToThread(self._thread)
        
        self._worker.progress_val.connect(self.progress_val.emit)
        
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_worker_finished(self, task_id, result):
        # print(f"Worker finished for task: {task_id} with result: {type(result)}")
        self.status_msg.emit("Measurement Completed.")
        self.is_measuring = False
        self.progress_val.emit(100)
        file_name = ""
        # if there is a string result in the last position, set it as the name of the saved file
        # check if results is a list or tuple
        if isinstance(result, (list, tuple)) and any(isinstance(r, str) for r in result):
            if isinstance(result, list):
                r_last = result[-1]
                if isinstance(r_last, str):
                    file_name = r_last
            elif isinstance(result, tuple):
                r_last = result[-1]
                if isinstance(r_last, str):
                    file_name = r_last
                    
        # Logic routing based on task
        if task_id == "FIND_TPSF":
            offsets, steps = result
            if offsets: 
                self.found_offset = offsets
                self.found_gate_steps = steps
                self._save_gate_offset("tpsf", offsets)
                self._save_gate_steps("tpsf", steps)
        elif task_id == "FIND_IRF":
            offsets, steps = result
            if offsets: 
                self.found_irf_offset = offsets
                self.found_irf_gate_steps = steps
                self._save_gate_offset("irf", offsets)
                self._save_gate_steps("irf", steps)
        elif task_id in ["MEASURE_TPSF", "MEASURE_IRF"]:
            self._handle_measure_result(result, is_irf=(task_id == "MEASURE_IRF"))
        elif task_id in ["GATED_SCAN", "GATED_SCOS"]:  # ADD THIS
            self._handle_measure_result_SCOS(result)
        elif task_id == "PALS_ISCOS":
            self.status_msg.emit("PaLS-iSCOS Complete!")
            self._handle_measure_result_SCOS(result)
        elif task_id == "MEASURE_DARK_NOISE":
            self.status_msg.emit("Dark Noise Complete!")

        self.measurement_finished.emit(task_id)

    def _on_worker_error(self, err_msg):
        self.is_measuring = False
        self.status_msg.emit(f"Error: {err_msg}")
        self.measurement_finished.emit("Error")

    # =========================================================================
    # 1. HARDWARE SETTINGS & MONITORING
    # =========================================================================

    def update_hardware_integration(self, val_ms):
        # print(f"CameraInterface: update_hardware_integration called with {val_ms} ms.")
        """Sends command to camera and updates UI estimates."""
        if not self.camera._connected: return
        try:
            self.camera.SPAD1.t.sendall(f'INTT,{val_ms}'.encode('utf-8'))
            self.calculate_estimates(val_ms)
        except Exception as e:
            print(f"Integration Error: {e}")

    def calculate_estimates(self, int_time_ms):
        # print(f"CameraInterface: calculate_estimates called with {int_time_ms} ms.")
        """Calculates expected FPS and Exposure based on hardware readout."""
        readout_time_ms = 6.5 
        est_fps = 1000.0 / max(int_time_ms, readout_time_ms)
        self.estimates_updated.emit({
            "fps": est_fps,
            "exposure": int_time_ms * 1000.0 
        })

    def fetch_camera_specs(self):
        # print("CameraInterface: fetch_camera_specs called.")
        """Polling loop for sidebar data."""
        if self.is_measuring or not self.camera._connected: return
        try:
            vq, vex = self.camera.SPAD1.get_voltages()
            temps = self.camera.SPAD1.get_temps()
            freq_data = self.camera.SPAD1.get_freq()
            self.specs_updated.emit({
                "vq": str(vq), "vex": str(vex), 
                "temp": str(temps[3]) if len(temps)>3 else "--",
                "laser": f"{float(freq_data[2])/1e6:.2f}",
                "fps": f"{float(freq_data[1]):.2f}"
            })
        except: pass

    # =========================================================================
    # 2. CALIBRATION ROUTINES (Threaded)
    # =========================================================================

    def calibrate_noises(self):
        # print("CameraInterface: calibrate_noises called.")
        self.status_msg.emit("Calibrating Dark Noise...")
        self._run_in_background("CALIB_NOISE", self.camera.calibrate_noise)

    def calibrate_breakdown(self):
        # print("CameraInterface: calibrate_breakdown called.")
        self.status_msg.emit("Starting Breakdown Calib (~20s)...")
        self._run_in_background("CALIB_BRK", self.camera.calibrate_breakdown)

    def calibrate_master_slave(self):
        # print("CameraInterface: calibrate_master_slave called.")
        self.status_msg.emit("Starting M/S Offset Calib...")
        self._run_in_background("CALIB_MS", self.camera.calibrate_master_slave_offset)

    # =========================================================================
    # 3. TPSF & DECONVOLUTION WORKFLOW
    # =========================================================================

    def find_TPSF(self, bitDepth=None, intTime=20, gate_width=GATE_SIZE_TPSF):
        # print("CameraInterface: find_TPSF called.")
        if bitDepth is None:
            bitDepth = self.bit_depth
        self.status_msg.emit("Finding TPSF Peak...")
        freq_data = self.camera.SPAD1.get_freq()
        laser_period = 1e12 / float(freq_data[2]) if len(freq_data) > 2 else 50000.0
        # load ROI mask from logic if available
        roi_mask = self.camera.get_roi_mask()
        if self.master_only:
            self.status_msg.emit("Measuring Only Master SPADs.")
            master_mask= np.zeros_like(roi_mask, dtype=bool)
            # master pixels are at the left half of the array
            master_mask[:roi_mask.shape[1]//2, :] = True
            roi_mask = roi_mask & master_mask
        
        self._run_in_background("FIND_TPSF", self.camera.Find_TPSF, 
                                laser_period=laser_period, gate_width=gate_width, 
                                bitDepth=bitDepth, intTime=intTime, mask=roi_mask)
        self.status_msg.emit("Finding TPSF Peak Task Completed.")
    def find_IRF(self, bitDepth=None, intTime=20, gate_width=GATE_SIZE_TPSF):
        # print("CameraInterface: find_IRF called.")
        if bitDepth is None:
            bitDepth = self.bit_depth
        self.status_msg.emit("Finding IRF Peak...")
        freq_data = self.camera.SPAD1.get_freq()
        laser_period = 1e12 / float(freq_data[2]) if len(freq_data) > 2 else 50000.0
        # load ROI mask from logic if available
        roi_mask = self.camera.get_roi_mask()
        if self.master_only:
            self.status_msg.emit("Measuring Only Master SPADs.")
            master_mask= np.zeros_like(roi_mask, dtype=bool)
            # master pixels are at the left half of the array
            master_mask[:roi_mask.shape[1]//2, :] = True
            roi_mask = roi_mask & master_mask
        
        self._run_in_background("FIND_IRF", self.camera.Find_IRF, 
                                laser_period=laser_period, gate_width=gate_width, 
                                bitDepth=bitDepth, intTime=intTime, mask=roi_mask)
        
    def measure_TPSF(self, bitDepth=None, intTime=10, gate_width=GATE_SIZE_TPSF, iterations=5):
        # print("CameraInterface: measure_TPSF called.")
        if bitDepth is None:
            bitDepth = self.bit_depth
        if self.found_offset is None:
            self.status_msg.emit("Error: Find Peak First!")
            return
        roi_mask = self.camera.get_roi_mask()
        if self.master_only:
            self.status_msg.emit("Measuring Only Master SPADs.")
            master_mask= np.zeros_like(self.camera.get_roi_mask(), dtype=bool)
            # master pixels are at the left half of the array
            master_mask[:self.camera.get_roi_mask().shape[1]//2, :] = True
            roi_mask = roi_mask & master_mask
        self.status_msg.emit("Starting TPSF Measurement...")
        self._run_in_background("MEASURE_TPSF", self.camera.measure_TPSF, gate_offset=self.found_offset, gate_steps=self.found_gate_steps, intTime=intTime, bitDepth=bitDepth, gate_width=gate_width,iterations= iterations, mask=roi_mask) 

    
    def measure_IRF(self, bitDepth=None, intTime=10, gate_width=GATE_SIZE_TPSF, iterations=5):
        # print("CameraInterface: measure_IRF called.")
        if bitDepth is None:
            bitDepth = self.bit_depth
        if self.found_irf_offset is None:
            self.status_msg.emit("Error: Find IRF Peak First!")
            return
        roi_mask = self.camera.get_roi_mask()
        if self.master_only:
            self.status_msg.emit("Measuring Only Master SPADs.")
            master_mask= np.zeros_like(self.camera.get_roi_mask(), dtype=bool)
            # master pixels are at the left half of the array
            master_mask[:self.camera.get_roi_mask().shape[1]//2, :] = True
            roi_mask = roi_mask & master_mask
        self.status_msg.emit("Starting IRF Measurement...")
        self._run_in_background("MEASURE_IRF", self.camera.measure_TPSF, gate_offset=self.found_irf_offset, gate_steps=self.found_irf_gate_steps, intTime=intTime, bitDepth=bitDepth, gate_width=gate_width, iterations=iterations, mask=roi_mask) 
        
    def _handle_measure_result(self, result, is_irf, file_name=""):
        # print(f"CameraInterface: _handle_measure_result called for {'IRF' if is_irf else 'TPSF'} with result type {type(result)}.")
        t_axis, raw_intensity = result
        if raw_intensity is None: return
        iteration_vec = range(1, 501, 5)
        
        if is_irf:
            if self.gate_profile is not None:
                # flip gate and time axis
                gate_timevec_flipped = -self.gate_timevec[::-1]
                gate_profile_flipped = self.gate_profile[::-1]
                best_deconvolved_IRF, best_iterations, best_mse, best_fwhm, MSE_vec, FWHM_vec, iteration_vec = TPSF_deconvolution_multiple_iterations(
                    raw_intensity, t_axis, gate_profile_flipped, gate_timevec_flipped, iteration_vec
                )
                metrics = {"mse": MSE_vec, "fwhm": FWHM_vec, "best_mse": best_mse, "best_fwhm": best_fwhm, "best_iterations": best_iterations}
                self.irf_deconv_ready.emit(t_axis, raw_intensity, best_deconvolved_IRF, iteration_vec, metrics)
                self._save_measurement_TPSF(t_axis, raw_intensity, best_deconvolved_IRF, file_name or "IRF",iteration_vec , metrics)
                
                # Find peak of deconvolved IRF
                if best_deconvolved_IRF is not None:
                    peak_idx = np.argmax(best_deconvolved_IRF)
                    peak_time = t_axis[peak_idx]
                    self.irf_peak_time = peak_time  # Store for SCOS
                    self.irf_peak_found.emit(peak_time)  # Emit peak found signal
            else:
                self.irf_data_ready.emit(t_axis, raw_intensity)
                self._save_measurement_TPSF(t_axis, raw_intensity, None, file_name or "IRF", iteration_vec)
        else:
            if self.gate_profile is not None:
                # flip gate and time axis
                gate_timevec_flipped = -self.gate_timevec[::-1]
                gate_profile_flipped = self.gate_profile[::-1]
                best_deconvolved_TPSF, best_iterations, best_mse, best_fwhm, MSE_vec, FWHM_vec, iteration_vec = TPSF_deconvolution_multiple_iterations(
                    raw_intensity, t_axis, gate_profile_flipped, gate_timevec_flipped, iteration_vec
                )
                metrics = {
                    "mse": MSE_vec,
                    "fwhm": FWHM_vec,
                    "best_mse": best_mse,
                    "best_fwhm": best_fwhm,
                    "best_iterations": best_iterations
                }

                self.tpsf_deconv_ready.emit(t_axis, raw_intensity, best_deconvolved_TPSF, iteration_vec, metrics)  # Add raw_intensity
                self._save_measurement_TPSF(t_axis, raw_intensity, best_deconvolved_TPSF, file_name or "TPSF",iteration_vec,  metrics)
                
                # Find peak of deconvolved TPSF
                if best_deconvolved_TPSF is not None:
                    peak_idx = np.argmax(best_deconvolved_TPSF)
                    peak_time = t_axis[peak_idx]
                    self.tpsf_peak_time = peak_time
                    self.tpsf_peak_ns = peak_time / 1e3  # Store in ns for SCOS
                    self.tpsf_peak_found.emit(peak_time)  # Emit peak found signal
            else:
                self.tpsf_data_ready.emit(t_axis, raw_intensity)
                self._save_measurement_TPSF(t_axis, raw_intensity, None, file_name or "TPSF", iteration_vec)
    def _handle_measure_result_SCOS(self, result, file_name=""):
        # print(f"CameraInterface: _handle_measure_result_SCOS called with result type {type(result)}.")
        # Check if this is scan mode (lists of arrays) or single mode (flat arrays)
        is_scan = isinstance(result.get("gate_offsets"), (list, np.ndarray)) and len(result.get("gate_offsets", [])) > 1
        
        if is_scan:
            # Scan mode: already saved by _worker_scos_scan, just emit status
            self.status_msg.emit(f"SCOS Scan complete. {len(result['gate_offsets'])} gate offsets measured and saved.")
            return
        
        # Single-point mode: original logic
        K2_raw = result.get('K2_raw')
        K2_corrected = result.get('K2_corrected')
        time_vector = result.get('time_vector')
        image_data = result.get('image_data')
        mask = result.get('mask')
        
        # if K2_raw is None or time_vector is None:
        #     self.status_msg.emit("Incomplete SCOS result - skipping save.")
        #     return
        
        prefix = "SCOS_Single"
        folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
        # os.makedirs(folder, exist_ok=True)
        
        # # Save data
        # np.save(os.path.join(folder, "time_vector.npy"), time_vector)
        # np.save(os.path.join(folder, "K2_raw.npy"), K2_raw)
        # if K2_corrected is not None:
        #     np.save(os.path.join(folder, "K2_corrected.npy"), K2_corrected)
        # np.save(os.path.join(folder, "image_data.npy"), image_data)
        # if mask is not None:
        #     np.save(os.path.join(folder, "roi_mask.npy"), mask)
            
        # FFT plot for single mode
        try:
            peak_freq_raw, SNR_raw, pos_mag_raw, pos_freq_raw, _ = find_fft_peak(time_vector, K2_raw)
            peak_freq_corr, SNR_corr, pos_mag_corr, pos_freq_corr, _ = find_fft_peak(time_vector, K2_corrected) if K2_corrected is not None else (0, 0, None, None, None)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.semilogy(pos_freq_raw, pos_mag_raw, 'b-', label=f'K2 Raw (SNR: {SNR_raw:.1f})')
            if pos_mag_corr is not None:
                ax.semilogy(pos_freq_corr, pos_mag_corr, 'r-', label=f'K2 Corr (SNR: {SNR_corr:.1f})')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder, "K2_FFT.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"FFT plot failed: {e}")


    def _handle_measure_result_SACS(self, result, file_name=""):
        is_scan = isinstance(result.get("gate_offsets"), (list, np.ndarray)) and len(result.get("gate_offsets", [])) > 1
        if is_scan:
            self.status_msg.emit(f"SACS Scan complete. {len(result['gate_offsets'])} gate offsets measured and saved.")
            return
        
        
       
        

    # =========================================================================
    # 4. SCOS & LIVE VIEW
    # =========================================================================

    def get_live_intensity(self):
        return None if self.is_measuring else self.camera.get_live_intensity()

    # def run_automatic_roi(self):
    #     frame = self.get_live_intensity()
    #     if frame is not None:
    #         mask = find_mask(frame)
    #         self.roi_found.emit(mask)
    #         self.status_msg.emit("ROI Found.")
    #         return mask

    def calculate_scos_fps(self, n_frames, gate_pos):
        # print("CameraInterface: calculate_scos_fps called.")
        """Calculates actual hardware FPS during SCOS stack capture."""
        start = perf_counter()
        stack = self.camera.capture_gated_stack(n_frames, gate_pos)
        duration = perf_counter() - start
        fps = n_frames / duration if duration > 0 else 0
        return fps, stack

    # def Measure_Gated_SCOS(self):
    #     print("CameraInterface: Measure_Gated_SCOS called.")
    #     page = self.parent().scos_page
    #     peak_ps = page.tpsf_peak_time
    #     start_offset = page.gate_start_spin.value()
    #     step_size = page.gate_step_spin.value()
    #     num_steps = page.num_steps_spin.value()
    #     n_frames = page.frames_per_step_spin.value()
    #     int_time = page.integration_time_spin.value()
    #     self.status_msg.emit(f"Starting SCOS Sweep: {num_steps} steps, Peak={peak_ps:.0f} ps")

    #     roi_mask = self.camera.get_roi_mask()
    #     if roi_mask is None:
    #         roi_mask = np.ones((512, 512), dtype=bool)
    #     if self.master_only:
    #         roi_mask[:, 256:] = False

    #     self._run_in_background(
    #         "GATED_SCAN",
    #         self._worker_scos_scan,
    #         peak_ps, start_offset, step_size, num_steps, n_frames, int_time, roi_mask
    #     )
     
    # def Measure_PaLS_iSCOS(self):
    #     print("CameraInterface: Measure_PaLS_iSCOS called.")
    #     self._run_in_background("PALS_ISCOS", self.camera.Measure_PaLS_iSCOS, 100, 0)

    
    def measure_dark_noise_for_scos(self, n_frames=600, gate_pos=0):
        """
        Measure 600 frames of gated intensity for SCOS dark noise,
        using GATE_SIZE_FOR_SCOS and user-specified integration time.
        """
        # print("CameraInterface: measure_dark_noise_for_scos called.")
        int_time_ms = self.parent().controls.spin_int_time.value()
        self.status_msg.emit(f"Measuring {n_frames} dark frames (int. time: {int_time_ms} ms, gate: {GATE_SIZE_FOR_SCOS} ps)...")
        dark_frames = self.camera.measure_dark_noise_for_scos(
            n_frames=n_frames,
            gate_pos=gate_pos,
            int_time_ms=int_time_ms,
            bitDepth=self.bit_depth,
            pileup=self.pileup_correction
        )
        self.status_msg.emit("Dark noise measurement complete.")
        return dark_frames

    def process_dark_noise(self, dark_frames):
        # print("CameraInterface: process_dark_noise called.")
        self.backgroundImg, self.darkVarPerWindow = calculate_dark_noise(dark_frames)
        self.status_msg.emit("Dark noise calculated and stored.")

    def measure_and_process_dark_noise_background(self, int_time=10.0, bit_depth=8, pileup_correction=0):
        """Runs the dark noise measurement in a background thread, using provided parameters."""
        self._run_in_background(
            "MEASURE_DARK_NOISE",
            lambda progress_callback=None: self._worker_dark_noise(int_time, bit_depth, pileup_correction, progress_callback)
        )

    def _worker_dark_noise(self, int_time, bit_depth, pileup_correction, progress_callback=None):
        dark_frames = self.camera.measure_dark_noise_for_scos(
            n_frames=600,
            int_time_ms=int_time,
            bitDepth=bit_depth,
            pileup=pileup_correction
        )
        if dark_frames is not None:
            self.backgroundImg, self.darkVarPerWindow = calculate_dark_noise(dark_frames)
            self.status_msg.emit("Dark noise calibration successful.")
            if progress_callback:
                progress_callback(100)

    # def _worker_gated_sweep(self, progress_callback=None):
    #     # print("CameraInterface: _worker_gated_sweep called.")
    #     """Logic for the Gated SCOS Sweep (Scan)."""
    #     page = self.parent().scos_page
        
    #     # 1. Setup Parameters
    #     peak_ps = self.tpsf_peak_ns * 1000.0
    #     start_off = page.gate_start_spin.value()
    #     step_ps = page.gate_step_spin.value()
    #     num_steps = page.num_steps_spin.value()
    #     n_frames = page.frames_per_step_spin.value()
    #     int_time = page.integration_time_spin.value()
    #     mask = page.current_mask

    #     delays, k2_vals, int_vals = [], [], []

    #     for i in range(num_steps):
    #         # Calculate Gate Position: Peak + Start Offset + (Step Index * Step Size)
    #         curr_delay = peak_ps + start_off + (i * step_ps)
            
    #         # --- Hardware Acquisition & Precise FPS Timing ---
    #         t0 = perf_counter()
    #         stack = self.camera.acquire_scos_stack(int_time, n_frames, GATE_SIZE_FOR_SCOS, curr_delay,num_steps, i)  # This should be a blocking call that returns the acquired stack
    #         dt = perf_counter() - t0
            
    #         if stack is None: break
            
    #         # Calculate actual FPS for this specific acquisition
    #         actual_fps = n_frames / dt if dt > 0 else 1.0
            
    #         # --- SCOS Math ---
    #         camera_gain = self.camera_gain

    #         res = SCOS_Calculation(
    #             image_data=stack,
    #             camera_gain=camera_gain,
    #             mask=mask if mask is not None else np.ones((512,512), dtype=bool),
    #             black_level=0,
    #             frame_rate=actual_fps, # Use measured FPS
    #             backgroundImg=self.backgroundImg if self.backgroundImg is not None else 0,
    #             darkVarPerWindow=self.darkVarPerWindow if self.darkVarPerWindow is not None else 0
    #         )
            
    #         delays.append(curr_delay)
    #         k2_vals.append(np.mean(res['K2_corrected']))
    #         int_vals.append(np.mean(stack))
            
    #         # Update Progress Bar (if available)
    #         if progress_callback:
    #             progress_callback(int((i + 1) / num_steps * 100))
                
    #         # Emit data to update the plots LIVE during the scan
    #         self.scos_data_ready.emit(delays, k2_vals, int_vals, "Gate Delay (ps)")

    #     return {"delays": delays, "K2": k2_vals}
    # =========================================================================
    # 5. IO & SHUTDOWN
    # =========================================================================

    def _save_measurement_TPSF(self, t, raw, decon, prefix,iteration_vec, metrics=None):
        # print(f"CameraInterface: _save_measurement_TPSF called for {prefix}.")
        if not self.save_path: return
        folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
        os.makedirs(folder, exist_ok=True)
        
        np.save(os.path.join(folder, "t_axis.npy"), t)
        np.save(os.path.join(folder, "raw_counts.npy"), raw)
        if decon is not None: 
            np.save(os.path.join(folder, "deconvolved.npy"), decon)
            np.save(os.path.join(folder, "iteration_vec.npy"), iteration_vec)
        # Plot and save results
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Raw trace
        axes[0, 0].plot(t, raw, 'b-', linewidth=1)
        axes[0, 0].set_title("Raw TPSF")
        axes[0, 0].set_xlabel("Time (ns)")
        axes[0, 0].set_ylabel("Counts")
        
        # Deconvolved trace
        if decon is not None:
            axes[0, 1].plot(t, decon, 'r-', linewidth=1)
            axes[0, 1].set_title(f"Deconvolved (FWHM={metrics['best_fwhm']:.2f} ns)")
            axes[0, 1].set_xlabel("Time (ns)")
            axes[0, 1].set_ylabel("Counts")
        
        # MSE vs iterations
        if metrics and 'mse' in metrics:
            np.save(os.path.join(folder, "MSE_vec.npy"), metrics['mse'])
            iter_list = list(iteration_vec)
            axes[1, 0].plot(iter_list, metrics['mse'], 'g-', marker='o')
            axes[1, 0].set_title(f"MSE vs Iterations (min={metrics['best_mse']:.4f})")
            axes[1, 0].set_xlabel("Iterations")
            axes[1, 0].set_ylabel("MSE")
            axes[1, 0].grid(True)
        
        # FWHM vs iterations
        if metrics and 'fwhm' in metrics:
            np.save(os.path.join(folder, "FWHM_vec.npy"), metrics['fwhm'])
            iter_list = list(iteration_vec)
            axes[1, 1].plot(iter_list, metrics['fwhm'], 'purple', marker='s')
            axes[1, 1].set_title(f"FWHM vs Iterations")
            axes[1, 1].set_xlabel("Iterations")
            axes[1, 1].set_ylabel("FWHM (ns)")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "results.png"), dpi=150, bbox_inches='tight')
        plt.close()
        # Save metadata
        metadata = {
            "measurement_type": prefix,
            "bit_depth": self.bit_depth,
            "master_only": self.master_only,
            "pileup_correction": bool(self.pileup_correction),
            "metrics": {
                "best_mse": float(metrics['best_mse']) if metrics and 'best_mse' in metrics else None,
                "best_fwhm": float(metrics['best_fwhm']) if metrics and 'best_fwhm' in metrics else None,
                "best_iterations": int(metrics['best_iterations']) if metrics and 'best_iterations' in metrics else None
            } if metrics else None,
            "time_axis_length": len(t),
            "raw_counts_mean": float(np.mean(raw)),
            "raw_counts_max": float(np.max(raw))
        }
        if decon is not None:
            metadata["decon_available"] = True
            metadata["decon_peak"] = float(np.max(decon))
        
        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=lambda x: None)        


            
            
    def check_connection(self):
        # print("CameraInterface: check_connection called.")
        s = self.camera._connected
        self.connection_status.emit(s, "Camera Connected" if s else "CAMERA DISCONNECTED")

    def close(self):
        # print("CameraInterface: close called.")
        self.camera.close()

    # ===== Add this helper inside CameraInterface =====
    def _acquire_scos_stack(self, int_time, n_frames, gate_width_ps, gate_pos_ps):
        # print(f"CameraInterface: _acquire_scos_stack called with int_time={int_time} ms, n_frames={n_frames}, gate_width={gate_width_ps} ps, gate_pos={gate_pos_ps} ps.")
        if not getattr(self.camera, "SPAD1", None):
            return None
        return self.camera.SPAD1.get_gated_intensity(
            bitDepth=self.bit_depth,
            intTime=int_time,
            iterations=n_frames,
            gate_steps=1,
            gate_step_size=MIN_STEP_SIZE,
            gate_step_arbitrary=0,
            gate_width=gate_width_ps / 1e3,  # ps -> ns
            gate_offset=gate_pos_ps,
            gate_direction=0,
            gate_trig=0,
            overlap=0,
            stream=0,
            pileup=self.pileup_correction,
            im_width=self.camera.im_width
        )

    # ===== Replace these methods =====
    def Measure_SCOS_Dispatch(self, params):
        # print(f"CameraInterface: Measure_SCOS_Dispatch called with params: {params}.")
        """Dispatches to the correct worker, handling ROI and Peaks like TPSF functions."""
        mode = params.get("mode")
        m_type = params.get("type")

        # 1. Base peak
        peak_ps = self.tpsf_peak_ns * 1000.0

        # 2. ROI handling
        roi_mask = self.camera.get_roi_mask()
        if roi_mask is None and hasattr(self, "last_roi_mask"):
            roi_mask = self.last_roi_mask
        if roi_mask is None:
            roi_mask = np.ones((512, 512), dtype=bool)
            print(f"CameraInterface: No ROI mask found in mode {mode}, {m_type} measurement, using default full mask.")

        if self.master_only:
            self.status_msg.emit("SCOS: Using Master SPADs only.")
            master_mask = np.zeros_like(roi_mask, dtype=bool)
            master_mask[:roi_mask.shape[1] // 2, :] = True
            roi_mask = roi_mask & master_mask

        # 3. Params
        start_offset = params.get("gate_start", 0)
        n_frames = params.get("n_frames", 100)
        int_time = params.get("int_time", 10)
        abs_start = peak_ps + start_offset

        if mode == "live":
            self.is_live_scos_running = True
            self._run_in_background("LIVE_SCOS", self._worker_live_loop,
                                abs_start, n_frames, int_time, roi_mask)
        elif mode == "acquire":
            if "Gated" in m_type:
                step_size = params.get("gate_step", 500)
                num_steps = params.get("num_steps", 10)
                # CORRECT ORDER: peak, offset, step, num, n_frames, int_time, mask
                self._run_in_background("GATED_SCOS", self._worker_scos_scan,
                            peak_ps, start_offset, step_size, num_steps, n_frames, int_time, roi_mask)
            else:
                self._run_in_background("PALS_ISCOS", self._worker_scos_single,
                                    abs_start, n_frames, int_time, roi_mask)

    def stop_live_scos(self):
        # print("CameraInterface: stop_live_scos called.")
        self.is_live_scos_running = False

    # def _run_live_scos(self, gate_pos, n_frames, int_time, mask):
    #     # print(f"CameraInterface: _run_live_scos called with gate_pos={gate_pos} ps, n_frames={n_frames}, int_time={int_time} ms.")
    #     self.is_live_scos_running = True
    #     self._thread = QtCore.QThread()
    #     self._worker = MeasurementWorker(
    #         "LIVE_SCOS",
    #         self._worker_live_loop,
    #         args=(gate_pos, n_frames, int_time, mask)
    #     )
    #     self._worker.moveToThread(self._thread)
    #     self._thread.started.connect(self._worker.run)
    #     self._worker.finished.connect(self._thread.quit)
    #     self._thread.start()

    def _worker_live_loop(self, gate_pos, n_frames, int_time, mask):
        # print(f"CameraInterface: _worker_live_loop started with gate_pos={gate_pos} ps, n_frames={n_frames}, int_time={int_time} ms.")
        """Live SCOS - continuously update time-resolved K² and intensity curves."""
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)

        while self.is_live_scos_running:
            # Acquire stack at current gate position
            t0 = perf_counter()
            stack = self.camera.acquire_scos_stack(int_time, n_frames, GATE_SIZE_FOR_SCOS, gate_pos, num_gates=0, gate_index=0, bitDepth=self.bit_depth, pileup=self.pileup_correction)  # This should be a blocking call that returns the acquired stack
            duration = perf_counter() - t0
            
            if stack is None:
                break

            # Calculate actual FPS
            actual_fps = n_frames / duration if duration > 0 else 1.0
            
            bg = self.backgroundImg if self.backgroundImg is not None else 0
            dv = self.darkVarPerWindow if self.darkVarPerWindow is not None else 0

            # Run SCOS calculation - get full time-resolved curves
            res = SCOS_Calculation(
                stack, self.camera_gain, mask, 0, actual_fps, bg, dv, is_pileup=bool(self.pileup_correction)
            )
            
            # Emit full time-resolved data (clears and replots each time)
            label = f"LIVE: {gate_pos:.0f} ps (FPS: {actual_fps:.1f})"
            self.scos_data_ready.emit(
                res['time_vector'],
                res['K2_corrected'],
                np.mean(stack, axis=(0, 1)),
                label,
                False
            )
            
            QtCore.QThread.msleep(100)  # Update every 100ms

    def _worker_scos_scan(self, peak, offset, step, num, n_frames, int_time, mask, progress_callback=None):
        # print(f"CameraInterface: _worker_scos_scan called with peak={peak}, offset={offset}, step={step}, num={num}, n_frames={n_frames}, int_time={int_time}, mask={mask is not None}")
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)

        all_time, all_k2_raw, all_k2_corr, all_int, all_offsets, all_BFi, all_image_data = [], [], [], [], [], [], []
        last_fps = 1.0

        
        for i in range(num):
            curr_delay = peak + offset + (i * step)
            t0 = perf_counter()
            stack = self.camera.acquire_scos_stack(int_time, n_frames, GATE_SIZE_FOR_SCOS, curr_delay, num, i, bitDepth=self.bit_depth, pileup=self.pileup_correction)  # This should be a blocking call that returns the acquired stack
            duration = perf_counter() - t0
            if stack is None:
                break

            actual_fps = n_frames / duration if duration > 0 else 1.0
            last_fps = actual_fps

            res = SCOS_Calculation(
                image_data=stack,
                camera_gain=self.camera_gain,
                mask=mask,
                black_level=0,
                frame_rate=actual_fps,
                backgroundImg=self.backgroundImg if self.backgroundImg is not None else 0,
                darkVarPerWindow=self.darkVarPerWindow if self.darkVarPerWindow is not None else 0,
                is_pileup=bool(self.pileup_correction)
            )


            # Live plot
            self.scos_data_ready.emit(
                res["time_vector"],
                res["K2_corrected"],
                np.mean(stack, axis=(0, 1)),
                f"Gate Offset: {curr_delay:.0f} ps (FPS: {actual_fps:.1f})",
                False
            )

            all_time.append(res["time_vector"])
            all_k2_raw.append(res["K2_raw"])
            all_k2_corr.append(res["K2_corrected"])
            all_BFi.append(res.get("BFi", None))
            all_int.append(np.mean(stack, axis=(0, 1)))
            all_image_data.append(res.get("image_data", None))
            all_offsets.append(curr_delay)

            if progress_callback:
                progress_callback(int((i + 1) / num * 100))

        # Save once after loop
        if self.save_path and len(all_time) > 0:
            prefix = "SCOS_Scan"
            folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
            os.makedirs(folder, exist_ok=True)

            np.save(os.path.join(folder, "gate_offsets.npy"), np.array(all_offsets))
            np.save(os.path.join(folder, "time_vector.npy"), np.array(all_time, dtype=object))
            np.save(os.path.join(folder, "K2_raw.npy"), np.array(all_k2_raw, dtype=object))
            np.save(os.path.join(folder, "K2_corrected.npy"), np.array(all_k2_corr, dtype=object))
            np.save(os.path.join(folder, "intensity.npy"), np.array(all_int, dtype=object))
            np.save(os.path.join(folder, "image_data.npy"), np.array(all_image_data, dtype=object))


            # Optional BFi
            if any(v is not None for v in all_BFi):
                np.save(os.path.join(folder, "BFi.npy"), np.array(all_BFi, dtype=object))

            metadata = {
                "gain": self.camera_gain,
                "frame_rate": last_fps,
                "integration_time": int_time,
                "n_frames": n_frames,
                "bit_depth": self.bit_depth,
                "is_pileup_correction": bool(self.pileup_correction)
            }
            np.save(os.path.join(folder, "roi_mask.npy"), mask if mask is not None else np.array([]))
            if self.backgroundImg is not None:
                np.save(os.path.join(folder, "backgroundImg.npy"), self.backgroundImg)
            if self.darkVarPerWindow is not None:
                np.save(os.path.join(folder, "darkVarPerWindow.npy"), self.darkVarPerWindow)
            with open(os.path.join(folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, default=lambda x: None)

        return {
            "time_vector": all_time,
            "K2_raw": all_k2_raw,
            "K2_corrected": all_k2_corr,
            "intensity": all_int,
            "BFi": all_BFi,
            "gate_offsets": all_offsets,
            "roi_mask": mask,
            "backgroundImg": self.backgroundImg,
            "darkVarPerWindow": self.darkVarPerWindow,
            "metadata": {
                "gain": self.camera_gain,
                "frame_rate": last_fps,
                "integration_time": int_time,
                "n_frames": n_frames,
                "bit_depth": self.bit_depth,
                "is_pileup_correction": bool(self.pileup_correction)
            }
        }
    def _worker_scos_single(self, gate_pos, n_frames, int_time, mask, progress_callback=None):
        # print(f"CameraInterface: _worker_scos_single called with gate_pos={gate_pos} ps, n_frames={n_frames}, int_time={int_time} ms, mask={mask is not None}")
        """
        Measure SCOS at a single gate position with time-resolved output.
        """
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)

        # 1. Acquire stack and measure actual duration
        t0 = perf_counter()
        stack = self._acquire_scos_stack(int_time, n_frames, GATE_SIZE_FOR_SCOS, gate_pos)
        duration = perf_counter() - t0
        
        if stack is not None:
            # 2. Calculate actual FPS from measurement
            actual_fps = n_frames / duration if duration > 0 else 1.0
            
            bg = self.backgroundImg if self.backgroundImg is not None else 0
            dv = self.darkVarPerWindow if self.darkVarPerWindow is not None else 0

            # 3. Run SCOS with actual FPS (not estimated)
            res = SCOS_Calculation(
                stack, self.camera_gain, mask, 0, actual_fps, bg, dv, is_pileup=bool(self.pileup_correction)
            )
            
            # 4. Emit time-resolved curve
            label = f"Single Gate: {gate_pos:.0f} ps (FPS: {actual_fps:.1f})"
            self.scos_data_ready.emit(
                res['time_vector'],
                res['K2_corrected'],
                np.mean(stack, axis=(0, 1)),
                label,
                True
            )
            
            # save all data
            prefix = "SCOS_Single"
            folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, "time_vector.npy"), res['time_vector'])
            np.save(os.path.join(folder, "K2_corrected.npy"), res['K2_corrected'])
            np.save(os.path.join(folder, "K2_raw.npy"), res['K2_raw'])
            np.save(os.path.join(folder, "intensity.npy"), np.mean(stack, axis=(0, 1)))
            np.save(os.path.join(folder, "image_data.npy"), stack)  
            np.save(os.path.join(folder, "roi_mask.npy"), mask if mask is not None else np.array([]))
            if self.backgroundImg is not None:
                np.save(os.path.join(folder, "backgroundImg.npy"), self.backgroundImg)
            if self.darkVarPerWindow is not None:
                np.save(os.path.join(folder, "darkVarPerWindow.npy"), self.darkVarPerWindow)

            metadata = {
                "gain": self.camera_gain,
                "frame_rate": actual_fps,
                "integration_time": int_time,
                "n_frames": n_frames,
                "bit_depth": self.bit_depth,
                "is_pileup_correction": bool(self.pileup_correction),
                "gate_position_ps": gate_pos

            }
            with open(os.path.join(folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, default=lambda x: None)
            
            return {
                "time_vector": res['time_vector'],
                "K2_corrected": res['K2_corrected'],
                "K2_raw": res['K2_raw'],
                "intensity": np.mean(stack, axis=(0, 1)),
                "image_data": stack,
                "metadata": metadata
            }

        if progress_callback:
            progress_callback(100)

    # =========================================================================
    # 5. SACS MEASUREMENT WORKFLOW
    # =========================================================================
    def _worker_sacs_live(self, gate_pos, n_frames, int_time, mask, speckle_size=1, progress_callback=None):
        """
        Live SACS - continuously loop until stopped.
        """
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)
        
        is_live_running = True
        
        while is_live_running:
            # Check if stop was requested
            if not getattr(self, 'is_live_sacs_running', False):
                break
            
            # Acquire stack
            t0 = perf_counter()
            stack = self.camera.acquire_sacs_stack(int_time, n_frames, GATE_SIZE_FOR_SACS, gate_pos,
                                                    num_gates=0, gate_index=0,
                                                    bitDepth=self.bit_depth, pileup=self.pileup_correction)
            duration = perf_counter() - t0
            
            if stack is None:
                break
            
            actual_fps = n_frames / duration if duration > 0 else 1.0
            
            # transpose stack to (n_frames, height, width) if needed
            if stack.shape[0] != n_frames:
                stack = np.transpose(stack, (2, 0, 1))  # Now (n_frames, height, width)
                
            # Process frames
            size_vals, peak_vals, _ = speckle_size_gs_framewise(stack, speckle_size=speckle_size)

            
            time_vector = np.arange(len(size_vals)) / actual_fps
            size_vals = np.array(size_vals)
            peak_vals = np.array(peak_vals)
            
            # Emit live plot
            label = f"LIVE: {gate_pos:.0f} ps (FPS: {actual_fps:.1f})"
            self.sacs_data_ready.emit(
                time_vector, size_vals, peak_vals, label, False
            )
            
            QtCore.QThread.msleep(100)  # Update every 100ms
            
    def Measure_SACS_Dispatch(self, params):
        """Dispatch SACS measurement based on mode."""
        mode = params.get("mode", "acquire")
        meas_type = params.get("type", "")
        peak_ps = self.tpsf_peak_ns * 1000.0
        speckle_size = params.get("speckle_size", self.speckle_size)
        
        # ROI handling
        roi_mask = self.camera.get_roi_mask()
        if roi_mask is None:
            roi_mask = np.ones((512, 512), dtype=bool)
        
        if self.master_only:
            master_mask = np.zeros_like(roi_mask, dtype=bool)
            master_mask[:roi_mask.shape[0] // 2, :] = True
            roi_mask = roi_mask & master_mask
        
        if mode == "acquire":
            if "Scan" in meas_type:
                self._run_in_background(
                    "SACS_SCAN",
                    self._worker_sacs_stack,
                    peak_ps, 
                    params.get("gate_start", 0),
                    params.get("gate_step", 500),
                    params.get("num_steps", 10),
                    params.get("n_frames", 100),
                    params.get("int_time", 10),
                    roi_mask,
                    speckle_size
                )
            else:
                self._run_in_background(
                    "SACS_SINGLE",
                    self._worker_sacs_single,
                    peak_ps + params.get("gate_start", 0),
                    params.get("n_frames", 100),
                    params.get("int_time", 10),
                    roi_mask,
                    speckle_size
                )
        elif mode == "live":
            # For LIVE, only pass these 5 arguments
            self.is_live_sacs_running = True
            self._run_in_background(
                "SACS_LIVE",
                self._worker_sacs_live,
                peak_ps + params.get("gate_start", 0),
                params.get("n_frames", 100),
                params.get("int_time", 10),
                roi_mask,
                speckle_size
            )

    def stop_live_sacs(self):
        """Stop live SACS measurement."""
        if hasattr(self, '_thread') and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _worker_sacs_single(self, gate_pos, n_frames, int_time, mask, speckle_size=1, progress_callback=None):
        """
        Measure SACS at a single gate position with time-resolved output.
        """
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)

        # 1. Acquire stack and measure actual duration
        t0 = perf_counter()
        stack = self.camera.acquire_sacs_stack(int_time, n_frames, GATE_SIZE_FOR_SACS, gate_pos, 
                                                num_gates=1, gate_index=0, 
                                                bitDepth=self.bit_depth, pileup=self.pileup_correction)
        duration = perf_counter() - t0
        
        if stack is None:
            if progress_callback:
                progress_callback(100)
            return

        # 2. Calculate actual FPS
        actual_fps = n_frames / duration if duration > 0 else 1.0
        
        
        # 3. Process each frame to get size and peak metrics
        # transpose stack to (n_frames, height, width) if needed
        if stack.shape[0] != n_frames:
            stack = np.transpose(stack, (2, 0, 1))  # Now (n_frames, height, width)
        size_vals, peak_vals, _ = speckle_size_gs_framewise(stack, speckle_size=speckle_size)

        
        
        
        time_vector = np.arange(len(size_vals)) / actual_fps
        size_vals = np.array(size_vals)
        peak_vals = np.array(peak_vals)
        
        # 4. Emit live plot
        label = f"SACS Single: {gate_pos:.0f} ps (FPS: {actual_fps:.1f})"
        self.sacs_data_ready.emit(
            time_vector, size_vals, peak_vals, label, True
        )
        
        # 5. Save data - same pattern as SCOS_Single
        if self.save_path:
            prefix = "SACS_Single"
            folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
            os.makedirs(folder, exist_ok=True)
            
            np.save(os.path.join(folder, "time_vector.npy"), time_vector)
            np.save(os.path.join(folder, "size_val.npy"), size_vals)
            np.save(os.path.join(folder, "peak_val.npy"), peak_vals)
            np.save(os.path.join(folder, "image_data.npy"), stack)  
            np.save(os.path.join(folder, "roi_mask.npy"), mask if mask is not None else np.array([]))
            
            metadata = {
                "gain": self.camera_gain,
                "frame_rate": actual_fps,
                "integration_time": int_time,
                "n_frames": n_frames,
                "bit_depth": self.bit_depth,
                "is_pileup_correction": bool(self.pileup_correction),
                "speckle_size": speckle_size
            }
            with open(os.path.join(folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, default=lambda x: None)
        
        if progress_callback:
            progress_callback(100)
        
        self.status_msg.emit("SACS single measurement complete.")


    def _worker_sacs_stack(self, peak, offset, step, num, n_frames, int_time, mask, speckle_size=1, progress_callback=None):
        """
        Measure SACS across a stack of gate positions.
        """
        if mask is None:
            mask = np.ones((512, 512), dtype=bool)

        all_time, all_size, all_peak, all_offsets, all_image_data = [], [], [], [], []
        last_fps = 1.0
        
        for i in range(num):
            curr_delay = peak + offset + (i * step)
            
            # Acquire stack
            t0 = perf_counter()
            stack = self.camera.acquire_sacs_stack(int_time, n_frames, GATE_SIZE_FOR_SACS, curr_delay,
                                                    num_gates=num, gate_index=i,
                                                    bitDepth=self.bit_depth, pileup=self.pileup_correction)
            duration = perf_counter() - t0
            
            if stack is None:
                break
            
            actual_fps = n_frames / duration if duration > 0 else 1.0
            last_fps = actual_fps
            
            # Process frames with specified speckle_size
            size_vals = []
            peak_vals = []
            
            # transpose stack to (n_frames, height, width) if needed
            if stack.shape[0] != n_frames:
                stack = np.transpose(stack, (2, 0, 1))  # Now (n_frames, height, width)
            size_vals, peak_vals, _ = speckle_size_gs_framewise(stack, speckle_size=speckle_size)
            
            time_vector = np.arange(len(size_vals)) / actual_fps
            size_vals = np.array(size_vals)
            peak_vals = np.array(peak_vals)
            
            # Emit live plot (same as SCOS - don't clear)
            label = f"Gate: {curr_delay:.0f} ps (FPS: {actual_fps:.1f})"
            self.sacs_data_ready.emit(
                time_vector, size_vals, peak_vals, label, False
            )
            
            all_time.append(time_vector)
            all_size.append(size_vals)
            all_peak.append(peak_vals)
            all_image_data.append(stack)
            all_offsets.append(curr_delay)
            
            if progress_callback:
                progress_callback(int((i + 1) / num * 100))
        
        # Save once after loop
        if self.save_path and len(all_offsets) > 0:
            prefix = "SACS_Scan"
            folder = os.path.join(self.save_path, f"{prefix}_{int(time())}")
            os.makedirs(folder, exist_ok=True)

            np.save(os.path.join(folder, "gate_offsets.npy"), np.array(all_offsets))
            np.save(os.path.join(folder, "time_vector.npy"), np.array(all_time, dtype=object))
            np.save(os.path.join(folder, "size_val.npy"), np.array(all_size, dtype=object))
            np.save(os.path.join(folder, "peak_val.npy"), np.array(all_peak, dtype=object))
            np.save(os.path.join(folder, "image_data.npy"), np.array(all_image_data, dtype=object))
            np.save(os.path.join(folder, "roi_mask.npy"), mask if mask is not None else np.array([]))

            metadata = {
                "gain": self.camera_gain,
                "frame_rate": last_fps,
                "integration_time": int_time,
                "n_frames": n_frames,
                "bit_depth": self.bit_depth,
                "is_pileup_correction": bool(self.pileup_correction),
                "speckle_size": speckle_size
            }
            with open(os.path.join(folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, default=lambda x: None)

        self.status_msg.emit("SACS scan measurement complete.")