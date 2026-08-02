import os
import numpy as np
from PySide6 import QtCore
from typing import Optional, Tuple, List, Union
from Gated_SCOS_Calculation.SPAD512S import SPAD512S
from utils.SCOS_calculation import SCOS_Calculation

MAX_GATE_WIDTH = 60000  # in ps
MIN_STEP_SIZE = 18.6  # in ps
GATE_SIZE_FOR_SCOS = 10000  # in ps


class SPAD_Camera:
    def __init__(self, port=9999, im_width=512, auto_connect=True):
        self.port = port
        self.im_width = im_width
        self.SPAD1 = None
        self._connected = False
        self.roi_mask = None
        
        if auto_connect:
            self.connect()

    def connect(self):
        """Attempts to establish the socket connection."""
        try:
            # Initialize vendor client
            if self.SPAD1 is not None and hasattr(self.SPAD1, 't'):
                try:
                    self.SPAD1.t.close() # Force close the old dead socket
                except:
                    pass
            self.SPAD1 = SPAD512S(self.port)
            
            # Check if the underlying socket 't' was actually created
            if hasattr(self.SPAD1, 't') and self.SPAD1.t is not None:
                self._connected = True
                print(f"Trying to connect to SPAD on port {self.port}")
            else:
                self._connected = False
                print("Connection failed: Command Server not responding.")
        except Exception as e:
            self._connected = False
            print(f"Socket Error: {e}")

   
   
    def set_roi_mask(self, mask):
        # print("camera_wrapper: set_roi_mask called.")
        self.roi_mask = mask
        
    def get_roi_mask(self):
        # print("camera_wrapper: get_roi_mask called.")
        return self.roi_mask
    ###############################
    ##### Noise Calibration #####
    ###############################
    
    
    def calibrate_noise(self):
        # print("camera_wrapper: calibrate_noise called.")
        if not self._connected: return "Not Connected"
        # Calling the vendor method self.SPAD1.calib_noise()
        return self.SPAD1.calib_noise() 

    def calibrate_dead(self):
        # print("camera_wrapper: calibrate_dead called.")
        if not self._connected: return "Not Connected"
        return self.SPAD1.calib_dead()   
    
    # In camera_wrapper.py
    def calibrate_breakdown(self):
        # print("camera_wrapper: calibrate_breakdown called.")
        if not self._connected: return "Disconnected"
        return self.SPAD1.calib_breakdown() # Must return this!

    def calibrate_master_slave_offset(self):
        # print("camera_wrapper: calibrate_master_slave_offset called.")
        if not self._connected: return "Disconnected"
        return self.SPAD1.calib_mst_slv_off() # Must return this!
    
    ###############################
    ##### Camera Measurement #####
    ###############################
    
    def get_live_intensity(self) -> np.ndarray:
            # print("camera_wrapper: get_live_intensity called.")
            if not self._connected or self.SPAD1 is None:
                return np.zeros((512, self.im_width))
            
            try:
                # We MUST provide all 7 arguments required by the vendor's function
                counts = self.SPAD1.get_intensity(
                    iterations=1,
                    intTime=1.0,     # Integration time
                    bitDepth=8,      # 8-bit
                    overlap=0,       # New: overlap
                    timeout=0,       # New: timeout
                    pileup=0,        # New: pileup
                    im_width=self.im_width
                )
                
                # The vendor returns [512, width, iterations]
                # We want the 2D slice [512, width]
                return counts[:, :, 0]

            except Exception as e:
                print(f"CRITICAL: get_intensity failed: {e}")
                return np.zeros((512, self.im_width))

    def Find_TPSF(self, laser_period, gate_width, bitDepth, intTime, mask, progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed version with progress reporting and loop safety.
        """
        # print("camera_wrapper: Find_TPSF called.")
        if not self._connected:
            return None, None
            
        try:
            gate_offsets = 0 
            max_gate_offset = laser_period + 1000 
            found_TPSF = False
            start_TPSF_offset = 0
            JUMP_BACK_STEP = 5000
            JUMP_FORWARD_STEP = 15000
            
            frame = self.get_live_intensity()
            gate_amplitude = 5*np.percentile(frame[mask], 90) if mask is not None else 5*np.percentile(frame, 90)          
            print(f"Calculated gate_amplitude threshold: {gate_amplitude:.5f}")
            
            # Safety: Calculate approximate total iterations for progress tracking
            # Broadly: 1 iteration per 1000ps step across the period
            est_total_iters = int(max_gate_offset / JUMP_FORWARD_STEP) + 2 
            current_iter = 0

            while gate_offsets < max_gate_offset:
                current_iter += 1
                print(f"Find_TPSF iteration {current_iter}, gate_offset={gate_offsets}")
                # Report progress to the UI via the callback
                if progress_callback:
                    # Calculate percentage (capped at 95% until actually found)
                    percent = min(95, int((current_iter / est_total_iters) * 100))
                    progress_callback(percent)

                # Measure gated intensity
                intensity_data = self.SPAD1.get_gated_intensity(
                    bitDepth=bitDepth, intTime=intTime, iterations=1, 
                    gate_steps=1000, gate_step_size=MIN_STEP_SIZE, 
                    gate_step_arbitrary=0, gate_width=gate_width/1e3, 
                    gate_offset=gate_offsets, gate_direction=0, 
                    gate_trig=0, overlap=0, stream=0, pileup=0, 
                    im_width=self.im_width
                )
                
                
                print(f"Intensity data received, now analyzing...")
                # Check if TPSF found
                # cut according to the ROI the intensity data
                if mask is not None:
                    intensity_data_mean = np.mean(intensity_data[mask], axis=0)
                else:
                    intensity_data_mean = np.mean(intensity_data, axis=(0,1))
                    # print(f"Mean intensity across entire frame: {np.mean(intensity_data_mean[0]):.5f}")    
                max_intensity = np.max(intensity_data_mean) if intensity_data_mean is not None else 0
                min_intensity = np.min(intensity_data_mean) if intensity_data_mean is not None else 0
                print(f"Intensity data mean range: {min_intensity:.5f} to {max_intensity:.5f}")
                
                if intensity_data_mean is not None and np.any(intensity_data_mean > gate_amplitude):
                    indices = np.where(intensity_data_mean > gate_amplitude)[0]
                    
                    # If signal is high at the START of the window
                    if indices[0] == 0:
                        if gate_offsets < 0:
                            # We're in negative territory and still see signal high
                            # Jump to near end of laser period where rising edge likely is
                            gate_offsets = laser_period - JUMP_BACK_STEP
                            print(f"Signal high at negative offset, jumping to {gate_offsets} (laser_period - {JUMP_BACK_STEP}).")
                        else:
                            # Back up for finer resolution
                            gate_offsets -= JUMP_BACK_STEP
                            print(f"Signal starts high, backing up to {gate_offsets} to catch rising edge.")
                            if gate_offsets < -2*JUMP_FORWARD_STEP: 
                                print("Could not find rising edge within range.")
                                break
                    else:
                        # Signal rose within this measurement window
                        
                        rising_edge_step = indices[0] - 1
                        print(f"TPSF rising edge found at gate_offset={gate_offsets}")
                        start_TPSF_offset = (rising_edge_step * MIN_STEP_SIZE) + gate_offsets
                        if start_TPSF_offset- gate_width >= 0:
                            found_TPSF = True
                            break
                        else:
                            print(f"Rising edge too close to start, continuing search...")
                            gate_offsets += laser_period  # Move to next period

                        
                else:
                    print(f"No TPSF detected at gate_offset={gate_offsets}.")
                    gate_offsets += JUMP_FORWARD_STEP
                    
            if found_TPSF:
                Measurement_zero = gate_width
                gate_offsets_with_zero = start_TPSF_offset - Measurement_zero
                steps_needed = int((3 * gate_width) / MIN_STEP_SIZE) + 1
                progress_callback(100)
                print(f"TPSF found! start_TPSF_offset={start_TPSF_offset:.2f}, gate_offsets_with_zero={gate_offsets_with_zero:.2f}, steps_needed={steps_needed}")                
                return gate_offsets_with_zero, steps_needed
            else:
                print("CRITICAL: TPSF not found within laser period.")
                progress_callback(100)
                return None, None

        except Exception as e:
            print(f"CRITICAL: Find_TPSF failed: {e}")
            progress_callback(100)
            return None, None
        
    def Find_IRF(self, laser_period, gate_width, bitDepth, intTime, mask, progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed version with progress reporting and loop safety.
        """
        # print("camera_wrapper: Find_IRF called.")
        if not self._connected:
            return None, None
            
        try:
            gate_offsets = 0 
            max_gate_offset = laser_period + 1000 
            found_IRF = False
            start_IRF_offset = 0
            JUMP_BACK_STEP = 5000
            JUMP_FORWARD_STEP = 15000
            
            frame = self.get_live_intensity()
            gate_amplitude = 2*np.percentile(frame[mask], 80) if mask is not None else 2*np.percentile(frame, 80)   
            print(f"Calculated gate_amplitude threshold: {gate_amplitude:.5f}")        
            # Safety: Calculate approximate total iterations for progress tracking
            # Broadly: 1 iteration per 1000ps step across the period
            est_total_iters = int(max_gate_offset / JUMP_FORWARD_STEP) + 2 
            current_iter = 0

            while gate_offsets < max_gate_offset:
                current_iter += 1
                print(f"Find_IRF iteration {current_iter}, gate_offset={gate_offsets}")
                # Report progress to the UI via the callback
                if progress_callback:
                    # Calculate percentage (capped at 95% until actually found)
                    percent = min(95, int((current_iter / est_total_iters) * 100))
                    progress_callback(percent)

                # Measure gated intensity
                intensity_data = self.SPAD1.get_gated_intensity(
                    bitDepth=bitDepth, intTime=intTime, iterations=1, 
                    gate_steps=1000, gate_step_size=MIN_STEP_SIZE, 
                    gate_step_arbitrary=0, gate_width=gate_width/1e3, 
                    gate_offset=gate_offsets, gate_direction=0, 
                    gate_trig=0, overlap=0, stream=0, pileup=0, 
                    im_width=self.im_width
                )
                print(f"Intensity data received, now analyzing...")
                # Check if IRF found
                # cut according to the ROI the intensity data
                if mask is not None:
                    intensity_data_mean = np.mean(intensity_data[mask], axis=0)
                else:
                    intensity_data_mean = np.mean(intensity_data, axis=(0,1))
                
                
                
                if intensity_data_mean is not None and np.any(intensity_data_mean > gate_amplitude):
                    indices = np.where(intensity_data_mean > gate_amplitude)[0]
                    
                    # If signal is high at the START of the window
                    if indices[0] == 0:
                        if gate_offsets < 0:
                            # We're in negative territory and still see signal high
                            # Jump to near end of laser period where rising edge likely is
                            gate_offsets = laser_period - JUMP_BACK_STEP
                            print(f"Signal high at negative offset, jumping to {gate_offsets} (laser_period - {JUMP_BACK_STEP}).")
                        else:
                            # Back up for finer resolution
                            gate_offsets -= JUMP_BACK_STEP
                            print(f"Signal starts high, backing up to {gate_offsets} to catch rising edge.")
                            if gate_offsets < -2*JUMP_FORWARD_STEP: 
                                print("Could not find rising edge within range.")
                                break
                    else:
                        # Signal rose within this measurement window
                        
                        rising_edge_step = indices[0] - 1
                        print(f"IRF rising edge found at gate_offset={gate_offsets}")
                        start_IRF_offset = (rising_edge_step * MIN_STEP_SIZE) + gate_offsets
                        if start_IRF_offset- gate_width >= 0:
                            found_IRF = True
                            break
                        else:
                            print(f"Rising edge too close to start, continuing search...")
                            gate_offsets += laser_period  # Move to next period

                        
                else:
                    print(f"No IRF detected at gate_offset={gate_offsets}.")
                    gate_offsets += JUMP_FORWARD_STEP
                    
            if found_IRF:
                Measurement_zero = gate_width
                gate_offsets_with_zero = start_IRF_offset - Measurement_zero
                steps_needed = int((3 * gate_width) / MIN_STEP_SIZE) + 1
                progress_callback(100)
                print(f"IRF found! start_IRF_offset={start_IRF_offset:.2f}, gate_offsets_with_zero={gate_offsets_with_zero:.2f}, steps_needed={steps_needed}")                
                return gate_offsets_with_zero, steps_needed
            else:
                print("CRITICAL: IRF not found within laser period.")
                progress_callback(100)
                return None, None

        except Exception as e:
            print(f"CRITICAL: Find_IRF failed: {e}")
            progress_callback(100)
            return None, None
    
    def measure_TPSF(self, bitDepth, intTime, iterations, gate_steps, gate_width, 
                    gate_offset, mask, progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        # print("camera_wrapper: measure_TPSF called.")
        if not self._connected:
            return None, None
        print("Starting TPSF/IRF measurement...")
        if progress_callback:
            progress_callback(0)
        try:
            intensity_data = self.SPAD1.get_gated_intensity(bitDepth=bitDepth, intTime=intTime, iterations=iterations, 
                    gate_steps=gate_steps, gate_step_size=MIN_STEP_SIZE, 
                    gate_step_arbitrary=0, gate_width=gate_width/1e3, 
                    gate_offset=gate_offset, gate_direction=0, 
                    gate_trig=0, overlap=0, stream=0, pileup=0, 
                    im_width=self.im_width )
            t_data = np.arange(gate_steps) * MIN_STEP_SIZE + gate_offset
            if mask is not None:
                intensity_data_mean = np.mean(intensity_data[mask], axis=0)
            else:
                intensity_data_mean = np.mean(intensity_data, axis=(0,1))
            # take mean value over iterations
            intensity_data_mean_over_iterations = np.zeros(gate_steps)
            for i in range(iterations):
                intensity_data_mean_over_iterations += intensity_data_mean[i*gate_steps:(i+1)*gate_steps]
            intensity_data_mean_over_iterations /= iterations
            if progress_callback:
                progress_callback(100)
            return t_data, intensity_data_mean_over_iterations
        
        except Exception as e:
            print(f"CRITICAL: measure_TPSF failed: {e}")
            if progress_callback:
                progress_callback(100)
            return None, None

    # Shutdown procedure 
    
    def close(self):
        try:
            if self.SPAD1 is not None:
                # We access the socket 't' inside the vendor's object
                if hasattr(self.SPAD1, 't') and self.SPAD1.t is not None:
                    self.SPAD1.t.close()
                    print("SUCCESS: Socket closed.", flush=True)
                else:
                    print("WARNING: No socket found to close.", flush=True)
        except Exception as e:
            print(f"ERROR: Exception during socket close: {e}", flush=True)
        finally:
            self._connected = False
            print("CAMERA STATE: Disconnected.", flush=True)

    # def Measure_Gated_SCOS(self):
    #     print("camera_wrapper: Measure_Gated_SCOS called.")
    #     # 1. Acquire image stack (replace with your actual acquisition code)
    #     image_data = self.camera.capture_gated_stack(100, 0)  # shape: (H, W, N)
    #     # 2. Prepare other parameters (mask, gain, etc.)
    #     mask = self.get_roi_mask()  # or however you get the mask
    #     camera_gain = 0.126  # or your gain value
    #     black_level = 0  # or your black level
    #     frame_rate = self.get_frame_rate()  # or your frame rate

    #     # 3. Run SCOS calculation
    #     results = SCOS_Calculation(
    #         image_data,
    #         camera_gain,
    #         mask,
    #         black_level,
    #         frame_rate,
    #         self.backgroundImg,
    #         self.darkVarPerWindow
    #     )
    #     # 4. Emit results or update UI as needed
    #     self.status_msg.emit("Gated SCOS calculation complete.")
    #     self.scos_results_ready.emit(results)

    # def Measure_PaLS_iSCOS(self):
    #     print("camera_wrapper: Measure_PaLS_iSCOS called.")

    #     image_data = self.camera.capture_pals_iscos_stack(100, 0)
    #     mask = self.camera.get_roi_mask()
    #     camera_gain = self.camera.get_gain()
    #     black_level = self.camera.get_black_level()
    #     frame_rate = self.camera.get_frame_rate()

    #     results = SCOS_Calculation(
    #         image_data,
    #         camera_gain,
    #         mask,
    #         black_level,
    #         frame_rate,
    #         self.backgroundImg,
    #         self.darkVarPerWindow
    #     )
    #     self.status_msg.emit("PaLS-iSCOS calculation complete.")
    #     self.scos_results_ready.emit(results)
    #     return results

    def acquire_scos_stack(self, intTime, n_frames, gate_width_ps, gate_offset_ps, num_gates, gate_index, bitDepth=8, pileup=0, progress_callback=None):
        """
        Acquires a (H, W, N) stack at a single gate position.
        """
        # print("camera_wrapper: acquire_scos_stack called.")
        if not self._connected:
            return None

        gate_width_ns = gate_width_ps / 1000.0

        if progress_callback and num_gates > 0:
            progress_callback(gate_index / num_gates * 100)
            
        return self.SPAD1.get_gated_intensity(
            bitDepth=bitDepth,
            intTime=intTime,
            iterations=n_frames,
            gate_steps=1,
            gate_step_size=MIN_STEP_SIZE,
            gate_step_arbitrary=0,
            gate_width=gate_width_ns,
            gate_offset=gate_offset_ps,
            gate_direction=0,
            gate_trig=0,
            overlap=0,
            stream=1,
            pileup=pileup,
            im_width=self.im_width
        )
        
    def acquire_sacs_stack(self, intTime, n_frames, gate_width_ps, gate_offset_ps, num_gates, gate_index, bitDepth=8, pileup=0, progress_callback=None):
        """
        Acquires a (H, W, N) stack at a single gate position for SACS.
        """
        # print("camera_wrapper: acquire_sacs_stack called.")
        if not self._connected:
            return None

        gate_width_ns = gate_width_ps / 1000.0

        if progress_callback and num_gates > 0:
            progress_callback(gate_index / num_gates * 100)
            
        return self.SPAD1.get_gated_intensity(
            bitDepth=bitDepth,
            intTime=intTime,
            iterations=n_frames,
            gate_steps=1,
            gate_step_size=MIN_STEP_SIZE,
            gate_step_arbitrary=0,
            gate_width=gate_width_ns,
            gate_offset=gate_offset_ps,
            gate_direction=0,
            gate_trig=0,
            overlap=0,
            stream=2,  # Use a different stream for SACS if needed
            pileup=pileup,
            im_width=self.im_width
        )
        

    def measure_dark_noise_for_scos(self, n_frames=600, gate_pos=0, int_time_ms=10.0, bitDepth=8, pileup=0, progress_callback=None):
        """
        Actually acquire the dark frames from hardware.
        Create a stack of (H, W, N) dark frames at the specified gate position and integration time.
        """
        # print("camera_wrapper: measure_dark_noise_for_scos called.")

        if not self._connected:
            return None
        dark_frames = self.SPAD1.get_gated_intensity(
            bitDepth=bitDepth,
            intTime=int_time_ms,
            iterations=n_frames,
            gate_steps=1,
            gate_step_size=MIN_STEP_SIZE,
            gate_step_arbitrary=0,
            gate_width=GATE_SIZE_FOR_SCOS/1e3,  # ps to ns
            gate_offset=gate_pos,
            gate_direction=0,
            gate_trig=0,
            overlap=0,
            stream=0,
            pileup=pileup,
            im_width=self.im_width
        )
        return dark_frames