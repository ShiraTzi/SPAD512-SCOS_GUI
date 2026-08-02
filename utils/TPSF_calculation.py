from skimage.restoration import richardson_lucy
from scipy.interpolate import interp1d
import numpy as np

def TPSF_deconvolution(measured_TPSF, IRF, num_iterations=30):
    """
    Deconvolve the measured TPSF using the IRF via Richardson-Lucy algorithm.
    Args:
        measured_TPSF: 1D numpy array of the measured TPSF data.
        IRF: 1D numpy array of the Instrument Response Function.
        num_iterations: Number of iterations for the deconvolution algorithm.
        
    Returns:
        deconvolved_TPSF: 1D numpy array of the deconvolved TPSF.
        MSE between reconvolved TPSF and measured TPSF.
    """
    # assume TPSF and IRF are 1D numpy arrays with the same time resolution
    # Normalize IRF
    IRF_normalized = IRF / np.sum(IRF)
    
    # Perform Richardson-Lucy deconvolution
    deconvolved_TPSF = richardson_lucy(measured_TPSF, IRF_normalized, num_iter=num_iterations, clip=False)
    deconvolved_TPSF *= np.sum(measured_TPSF) / np.sum(deconvolved_TPSF)  # Scale to match total counts
    # Reconvolve to check error
    reconvolved_TPSF = np.convolve(deconvolved_TPSF, IRF_normalized, mode='same')
    mse = np.mean((measured_TPSF - reconvolved_TPSF) ** 2)
    
    return deconvolved_TPSF, mse

def TPSF_deconvolution_multiple_iterations(measured_TPSF,measured_TPSF_timevec , IRF, IRF_timevec, iteration_list):
    """
    Runs TPSF deconvolution over a list of iteration counts to find the best result.
    Args:
        measured_TPSF: 1D numpy array of the measured TPSF data.
        measured_TPSF_timevec: 1D numpy array of the time values corresponding to the measured TPSF.(ps)
        IRF: 1D numpy array of the Instrument Response Function.
        IRF_timevec: 1D numpy array of the time values corresponding to the IRF.(ns)
        iteration_list: List of integers specifying iteration counts to try.
        
    Returns:
        deconvolved_TPSF: 1D numpy array of the best deconvolved TPSF.
        best_iterations: Number of iterations that yielded the best result.
        best_mse: MSE of the best result.
        best_fwhm: FWHM of the best result.
        MSE for each iteration count tried.
        FWHM for each iteration count tried.
    """
    
    dt_TPSF = measured_TPSF_timevec[1] - measured_TPSF_timevec[0]
    dt_IRF = IRF_timevec[1] - IRF_timevec[0]
  
    # Resample IRF to match TPSF time vector
    interp_func = interp1d(IRF_timevec, IRF, kind='linear', fill_value="extrapolate")
    # IRF_resampled = interp_func(measured_TPSF_timevec)

        
    # Generate a symmetric time vector centered at 0 to keep the PSF centered in the array
    t_limit_ps = max(abs(IRF_timevec.min() * 1000), abs(IRF_timevec.max() * 1000))
    time_vec_gate_ps = np.arange(-t_limit_ps, t_limit_ps, dt_TPSF)
    gate_profile_interp = interp_func(time_vec_gate_ps / 1000) 
    gate_profile = gate_profile_interp / np.sum(gate_profile_interp)  # normalize
    
    best_mse = float('inf')
    best_deconvolved_TPSF = None
    best_iterations = 0
    best_fwhm = 0.0
    MSE_vec=[]
    FWHM_vec=[]
    
    for iterations in iteration_list:
        deconvolved_TPSF, mse = TPSF_deconvolution(measured_TPSF, gate_profile, num_iterations=iterations)
        MSE_vec.append(mse)
        FWHM_vec.append(calculate_FWHM(deconvolved_TPSF, measured_TPSF_timevec))
        if mse < best_mse:
            best_mse = mse
            best_deconvolved_TPSF = deconvolved_TPSF
            best_iterations = iterations
            best_fwhm = calculate_FWHM(deconvolved_TPSF, measured_TPSF_timevec)
            
    return best_deconvolved_TPSF, best_iterations, best_mse, best_fwhm, MSE_vec, FWHM_vec, iteration_list

def calculate_FWHM(signal, time_vector):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a given signal.
    Args:
        signal: 1D numpy array of the signal.
        time_vector: 1D numpy array of the corresponding time values.
        
    Returns:
        FWHM value in the same units as time_vector.
    """
    half_max = np.max(signal) / 2.0
    indices_above_half_max = np.where(signal >= half_max)[0]
    
    if len(indices_above_half_max) < 2:
        return 0.0  # FWHM cannot be determined
    
    left_idx = indices_above_half_max[0]
    right_idx = indices_above_half_max[-1]
    
    fwhm = time_vector[right_idx] - time_vector[left_idx]
    return fwhm



# def TPSF_deconvolution_multiple_iterations(measured_TPSF, measured_TPSF_timevec, IRF, IRF_timevec, iteration_list):
#     """
#     Runs TPSF deconvolution over a list of iteration counts to find the best result.
#     FIXED: Now centers the IRF to prevent time-shifts.
#     """
#     # 1. Get the time step from the TPSF
#     dt = measured_TPSF_timevec[1] - measured_TPSF_timevec[0]
    
#     # 2. CENTER the IRF: Find the peak and subtract that time 
#     # so the IRF peak is defined at t=0
#     peak_time_irf = IRF_timevec[np.argmax(IRF)]
#     centered_irf_time = IRF_timevec - peak_time_irf
    
#     # 3. Create a SYMMETRIC target grid centered at 0
#     # This ensures the peak is in the middle of the array, preventing shifts in RL
#     t_limit = max(abs(centered_irf_time.min()), abs(centered_irf_time.max()))
#     t_eval = np.arange(-t_limit, t_limit, dt)
    
#     # 4. Resample IRF onto the centered grid
#     interp_func = interp1d(centered_irf_time, IRF, kind='linear', 
#                            bounds_error=False, fill_value=0)
#     IRF_resampled = interp_func(t_eval)
    
#     # --- Rest of the logic remains the same ---
#     best_mse = float('inf')
#     best_deconvolved_TPSF = None
#     best_iterations = 0
#     best_fwhm = 0.0
#     MSE_vec = []
#     FWHM_vec = []
    
#     for iterations in iteration_list:
#         deconvolved_TPSF, mse = TPSF_deconvolution(measured_TPSF, IRF_resampled, num_iterations=iterations)
#         MSE_vec.append(mse)
#         FWHM_vec.append(calculate_FWHM(deconvolved_TPSF, measured_TPSF_timevec))
        
#         if mse < best_mse:
#             best_mse = mse
#             best_deconvolved_TPSF = deconvolved_TPSF
#             best_iterations = iterations
#             best_fwhm = calculate_FWHM(deconvolved_TPSF, measured_TPSF_timevec)
            
#     return best_deconvolved_TPSF, best_iterations, best_mse, best_fwhm, MSE_vec, FWHM_vec, iteration_list