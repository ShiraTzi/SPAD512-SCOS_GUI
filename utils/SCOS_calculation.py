import numpy as np
import os
import scipy.io as sio
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt




# Fiber-Based Ultra-High-Speed Diffuse Speckle Contrast Analysis System for Deep Blood Flow Sensing Using a Large SPAD Camera 


def SCOS_Calculation(
    image_data,
    camera_gain,  # unused for SPAD (kept for compatibility)
    mask,
    black_level,
    frame_rate,
    backgroundImg,
    darkVarPerWindow,
    window_size=7,
    nBits=8,
    is_pileup=False,
    save_dir=None
):
    ## DEBUG START (Check to see if darkVarPerWindow is being computed correctly)
    # darkVarPerWindow=uniform_filter(darkVarPerWindow**2, size=window_size)-uniform_filter(darkVarPerWindow, size=window_size)**2
    ## DEBUG END
    

    # SPAD gain = 1
    actualGain = 1.0

    Nmax_lookup = {1: 1, 4: 59, 6: 333, 7: 759, 8: 1701, 9: 3756, 10: 8218, 11: 17850, 12: 38529}
    N_max = 2 ** nBits - 1 if not is_pileup else Nmax_lookup.get(nBits, 2 ** nBits - 1)

    # --- 3-sigma bad pixel removal (Wang2025) ---
    roi_pixels = backgroundImg[mask]
    mean_dark = np.nanmean(roi_pixels)
    std_dark = np.nanstd(roi_pixels)
    
    good_pixel_mask = backgroundImg <= (mean_dark + 3 * std_dark)
    combined_mask = mask & good_pixel_mask

    # --- Pre-compute Mask Weights for Normalized Convolution ---
    # Convert mask to float (1.0 for valid, 0.0 for invalid)
    M_float = combined_mask.astype(float)
    
    # Calculate the local fraction of valid pixels in each window
    mask_weights = uniform_filter(M_float, size=window_size)
    
    # Replace 0s with NaN to avoid division by zero in windows that are 100% masked out
    mask_weights[mask_weights == 0] = np.nan

    num_frames = image_data.shape[2]
    K2_raw = np.zeros(num_frames)
    K2_corrected = np.zeros(num_frames)
    BFi = np.zeros(num_frames)

    for frame in range(num_frames):
        im_raw = image_data[:, :, frame].astype(float) - black_level
        Ic = im_raw - backgroundImg  # corrected intensity

        # --- Global mean (Kept strictly for final normalization) ---
        meanFrame = np.nanmean(Ic[combined_mask])
        if meanFrame <= 0:
            meanFrame = 1e-5
        meanFrame_sq = meanFrame**2

        # --- Apply Mask BEFORE local windowing (Normalized Convolution) ---
        # Zero out the bad pixels so they don't contribute to the sum
        Ic_masked = Ic * M_float
        Ic_sq_masked = (Ic**2) * M_float

        I_raw_masked = im_raw * M_float
        
        # Calculate local mean and variance using only valid pixels
        local_mean = uniform_filter(Ic_masked, size=window_size) / mask_weights
        mean_im_sq = uniform_filter(Ic_sq_masked, size=window_size) / mask_weights
        
        local_mean_raw = uniform_filter(I_raw_masked, size=window_size) / mask_weights
        
        var_im = mean_im_sq - local_mean**2

        # --- Shot noise (LOCAL) ---
        if is_pileup: # the camera correted for pileup
            # Pile-up correction is applied spatially based on local photon flux
            correction_factor = 1.0
        else:
            correction_factor = np.maximum(
                1.0 - local_mean_raw / N_max,
                0.0
            )
        shot_noise_var = actualGain * local_mean_raw * correction_factor

        # --- Corrected variance ---
        corrected_var = var_im - darkVarPerWindow - shot_noise_var

        # --- Final contrasts ---
        K2_raw[frame] = np.nanmean(var_im[combined_mask]) / meanFrame_sq
        K2_corrected[frame] = np.nanmean(corrected_var[combined_mask]) / meanFrame_sq

        # --- BFi ---
        BFi[frame] = 1 / max(K2_corrected[frame], 1e-6)

    time_vector = np.arange(1, num_frames + 1) / frame_rate

    return {
        'K2_raw': K2_raw,
        'K2_corrected': K2_corrected,
        'BFi': BFi,
        'time_vector': time_vector,
        'gainCalc': camera_gain,
        'image_data': image_data,
        'mask': combined_mask
    }





# def SCOS_Calculation(
#     image_data,
#     camera_gain,  # unused for SPAD (kept for compatibility)
#     mask,
#     black_level,
#     frame_rate,
#     backgroundImg,
#     darkVarPerWindow,
#     window_size=7,
#     nBits=8,
#     is_pileup=False,
#     save_dir=None
# ):
#     from scipy.ndimage import uniform_filter

#     ## DEBUG START (Check to see if darkVarPerWindow is being computed correctly)
#     # darkVarPerWindow=uniform_filter(darkVarPerWindow**2, size=window_size)-uniform_filter(darkVarPerWindow, size=window_size)**2
#     ## DEBUG END


#     # SPAD gain = 1
#     actualGain = 1.0

#     Nmax_lookup = {1: 1, 4: 59, 6: 333, 7: 759, 8: 1701, 9: 3756, 10: 8218, 11: 17850, 12: 38529}
#     N_max = 2 ** nBits - 1 if not is_pileup else Nmax_lookup.get(nBits, 2 ** nBits - 1)

#     # --- 3-sigma bad pixel removal (Wang2025) ---
#     roi_pixels = backgroundImg[mask]
#     mean_dark = np.nanmean(roi_pixels)
#     std_dark = np.nanstd(roi_pixels)
    
#     good_pixel_mask = backgroundImg <= (mean_dark + 3 * std_dark)
#     combined_mask = mask & good_pixel_mask

#     # --- Pre-compute Mask Weights for Normalized Convolution ---
#     # Convert mask to float (1.0 for valid, 0.0 for invalid)
#     M_float = combined_mask.astype(float)
    
#     # Calculate the local fraction of valid pixels in each window
#     mask_weights = uniform_filter(M_float, size=window_size)
    
#     # Replace 0s with NaN to avoid division by zero in windows that are 100% masked out
#     mask_weights[mask_weights == 0] = np.nan

#     num_frames = image_data.shape[2]
#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     for frame in range(num_frames):
#         im_raw = image_data[:, :, frame].astype(float) - black_level
#         Ic = im_raw - backgroundImg  # corrected intensity

#         # --- Global mean (Kept strictly for final normalization) ---
#         meanFrame = np.nanmean(Ic[combined_mask])
#         if meanFrame <= 0:
#             meanFrame = 1e-5
#         meanFrame_sq = meanFrame**2

#         # --- Apply Mask BEFORE local windowing (Normalized Convolution) ---
#         # Zero out the bad pixels so they don't contribute to the sum
#         Ic_masked = Ic * M_float
#         Ic_sq_masked = (Ic**2) * M_float

#         # Calculate local mean and variance using only valid pixels
#         local_mean = uniform_filter(Ic_masked, size=window_size) / mask_weights
#         mean_im_sq = uniform_filter(Ic_sq_masked, size=window_size) / mask_weights
        
#         var_im = mean_im_sq - local_mean**2

#         # --- Shot noise (LOCAL) ---
#         if is_pileup:
#             # Pile-up correction is applied spatially based on local photon flux
#             correction_factor = (1.0 - local_mean / N_max)
#         else:
#             correction_factor = 1.0

#         shot_noise_var = actualGain * local_mean * correction_factor

#         # --- Corrected variance ---
#         corrected_var = var_im - darkVarPerWindow - shot_noise_var

#         # --- Final contrasts ---
#         K2_raw[frame] = np.nanmean(var_im[combined_mask]) / meanFrame_sq
#         K2_corrected[frame] = np.nanmean(corrected_var[combined_mask]) / meanFrame_sq

#         # --- BFi ---
#         BFi[frame] = 1 / max(K2_corrected[frame], 1e-6)

#     time_vector = np.arange(1, num_frames + 1) / frame_rate

#     return {
#         'K2_raw': K2_raw,
#         'K2_corrected': K2_corrected,
#         'BFi': BFi,
#         'time_vector': time_vector,
#         'gainCalc': camera_gain,
#         'image_data': image_data,
#         'mask': combined_mask
#     }


# def SCOS_Calculation(
#     image_data,
#     camera_gain,  # unused for SPAD (kept for compatibility)
#     mask,
#     black_level,
#     frame_rate,
#     backgroundImg,
#     darkVarPerWindow,
#     window_size=7,
#     nBits=8,
#     is_pileup=False,
#     save_dir=None
# ):
#     import numpy as np
#     from scipy.ndimage import uniform_filter

#     # SPAD → gain = 1
#     actualGain = 1.0

#     Nmax_lookup = {1: 1, 4: 59, 6: 333, 7: 759, 8: 1701, 9: 3756, 10: 8218, 11: 17850, 12: 38529}
#     N_max = 2 ** nBits - 1 if not is_pileup else Nmax_lookup.get(nBits, 2 ** nBits - 1)

#     # --- Static Spatial Noise ---
#     MAX_FRAMES_FOR_SP = min(300, image_data.shape[2])
#     mean_im_raw = np.mean(image_data[:, :, :MAX_FRAMES_FOR_SP].astype(float), axis=2)

#     spIm = mean_im_raw - black_level - backgroundImg

#     mean_spIm_sq = uniform_filter(spIm**2, size=window_size)
#     sq_mean_spIm = uniform_filter(spIm, size=window_size)**2
#     spVar = mean_spIm_sq - sq_mean_spIm

#     # --- 3-sigma bad pixel removal (Wang2025) ---
#     # Compute mean and std in ROI of dark-corrected mean image
#     roi_pixels = spIm[mask]
#     mean_dark = np.nanmean(roi_pixels)
#     std_dark = np.nanstd(roi_pixels)
#     # Mask out pixels with high dark counts
#     bad_pixel_mask = spIm <= (mean_dark + 3 * std_dark)
#     # Combine with user mask
#     combined_mask = mask & bad_pixel_mask

#     num_frames = image_data.shape[2]
#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     for frame in range(num_frames):
#         im_raw = image_data[:, :, frame].astype(float) - black_level
#         Ic = im_raw - backgroundImg  # corrected intensity

#         # --- Global mean (as in your equation) ---
#         meanFrame = np.nanmean(Ic[combined_mask])
#         if meanFrame <= 0:
#             meanFrame = 1e-5
#         meanFrame_sq = meanFrame**2

#         # --- Local variance σ_I^2 ---
#         mean_im_sq = uniform_filter(Ic**2, size=window_size)
#         sq_mean_im = uniform_filter(Ic, size=window_size)**2
#         var_im = mean_im_sq - sq_mean_im

#         # --- Shot noise σ_s^2 = <Ic> (GLOBAL, not local) ---
#         if is_pileup:
#             correction_factor = (1.0 - meanFrame / N_max)
#         else:
#             correction_factor = 1.0

#         shot_noise_var = actualGain * meanFrame * correction_factor

#         # --- Corrected variance (ONLY the terms in your equation) ---
#         corrected_var = var_im - darkVarPerWindow - shot_noise_var

#         # --- Final contrasts ---
#         K2_raw[frame] = np.nanmean(var_im[combined_mask]) / meanFrame_sq
#         K2_corrected[frame] = np.nanmean(corrected_var[combined_mask]) / meanFrame_sq

#         # --- BFi ---
#         BFi[frame] = 1 / max(K2_corrected[frame], 1e-6)

#     time_vector = np.arange(1, num_frames + 1) / frame_rate

#     return {
#         'K2_raw': K2_raw,
#         'K2_corrected': K2_corrected,
#         'BFi': BFi,
#         'time_vector': time_vector,
#         'gainCalc': camera_gain,
#         'image_data': image_data,
#         'mask': combined_mask
#     }
# def SCOS_Calculation(
#     image_data,
#     camera_gain,
#     mask,
#     black_level,
#     frame_rate,
#     backgroundImg,
#     darkVarPerWindow,
#     window_size=7,
#     nBits=8,
#     save_dir=None
# ):
#     """
#     SCOS calculation workflow implementing noise models and 3-sigma bad pixel removal.
#     """
#     video_size = image_data.shape[:2]
#     assert mask.shape == video_size, "Mask size must match video size"

#     # --- Static Noise Calibration (K2_sp) ---
#     MAX_FRAMES_FOR_SP = min(600, image_data.shape[2])
#     # Cast to float to avoid integer underflow (e.g., 5 - 10 = 65535 in uint16)
#     mean_Isp = np.mean(image_data[:, :, :MAX_FRAMES_FOR_SP].astype(float), axis=2) - backgroundImg - black_level
#     mean_Isp[mean_Isp <= 0] = 1e-5

#     # Equation (5): Spatial Noise Contrast (K^2_sp)
#     # Calculated from the mean image to remove static surface/optical noise
#     var_sp = uniform_filter(mean_Isp**2, size=window_size) - uniform_filter(mean_Isp, size=window_size)**2
#     K2_sp_map = var_sp / (mean_Isp**2)

#     num_frames = image_data.shape[2]
#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     # --- Frame-by-Frame Processing ---
#     for frame in range(num_frames):
#         im = image_data[:, :, frame].astype(float)
#         im = im - backgroundImg - black_level

#         # ==========================================
#         # 3-Sigma Rule for Bad Pixel Removal
#         # ==========================================
#         # 1. Calculate global ROI stats for the current frame
#         masked_pixels = im[mask]
#         mean_val = np.mean(masked_pixels)
#         std_val = np.std(masked_pixels)

#         # 2. Define upper/lower limits (3 standard deviations from mean)
#         lower_bound = mean_val - 3 * std_val
#         upper_bound = mean_val + 3 * std_val
        
#         # 3. Create mask for pixels that are statistically 'valid'
#         good_pixel_mask = mask & (im >= lower_bound) & (im <= upper_bound)

#         # --- Windowed Local Statistics (<I_window> and sigma^2_window) ---
#         mean_I_win = uniform_filter(im, size=window_size)
#         var_I_win = uniform_filter(im**2, size=window_size) - mean_I_win**2

#         # Combine bad pixel filter with signal thresholding
#         valid = good_pixel_mask & (mean_I_win > 1e-5)

#         with np.errstate(divide='ignore', invalid='ignore'):
#             # Equation (1): Raw Contrast (K^2_raw)
#             K2_raw_map = np.where(mean_I_win**2 > 0, var_I_win / (mean_I_win**2 + 1e-10), 0)
            
#             # Equation (2): Shot Noise Contrast (K^2_s = G / <I>)
#             K2_s_map = np.where(mean_I_win > 0, camera_gain / (mean_I_win + 1e-10), 0)
            
#             # Equation (3+4): Read & Quantization Noise (K^2_rq)
#             # Uses darkVarPerWindow (<sigma^2_dark,window>)
#             K2_rq_map = np.where(mean_I_win**2 > 0, darkVarPerWindow / (mean_I_win**2 + 1e-10), 0)
            
#             # Equation (6): Corrected Flow Contrast (K^2_f)
#             # K^2_f = K^2_raw - K^2_s - K^2_rq - K^2_sp
#             K2_f_map = K2_raw_map - K2_s_map - K2_rq_map - K2_sp_map

#         # --- Spatial Averaging over the valid ROI ---
#         K2_raw[frame] = np.nanmean(K2_raw_map[valid])
#         K2_corrected[frame] = np.nanmean(K2_f_map[valid])
        
#         # BFi = 1 / K^2_corrected
#         BFi[frame] = 1 / K2_corrected[frame] if K2_corrected[frame] != 0 else 0

#     time_vector = np.arange(1, num_frames + 1) / frame_rate

#     return {
#         'K2_raw': K2_raw,
#         'K2_corrected': K2_corrected,
#         'BFi': BFi,
#         'time_vector': time_vector,
#         'gainCalc': camera_gain,
#         'image_data': image_data,
#         'mask': mask
#     }

def find_mask(mean_img, threshold=0.1):
    """
    Find Circle mask in the image data based on Intensity disteribution.
    Returns: binary mask
    """
    gaussian_filtered = uniform_filter(mean_img, size=15)
    norm_img = (gaussian_filtered - np.min(gaussian_filtered)) / (np.max(gaussian_filtered) - np.min(gaussian_filtered))
    # find FWHM
    HM=np.max(norm_img) / 2
    if HM < threshold:
        raise ValueError("No object found in the image.")
    # find circle center and radius
    y_indices, x_indices = np.where(norm_img >= HM)
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    radius = int(np.max(np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)))
    Y, X = np.ogrid[:mean_img.shape[0], :mean_img.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    return mask

def calculate_dark_noise(dark_frames, window_size=7):
    """
    Calculate background image and dark variance per window from dark frames.
    Args:
        dark_frames: ndarray, shape (H, W, N_dark)
        window_size: int, window size for uniform filter
    Returns:
        backgroundImg: mean of dark frames, shape (H, W)
        darkVarPerWindow: local variance, shape (H, W)
    """
    backgroundImg = np.mean(dark_frames, axis=2)
    var_r = np.var(dark_frames, axis=2)
    darkVarPerWindow = uniform_filter(var_r, size=window_size)
    return backgroundImg, darkVarPerWindow

# def find_fft_peak(time_vector, signal):
#     """
#     Find the peak in the FFT of the signal to determine dominant frequency.
#     Args:
#         time_vector: 1D numpy array of time values.
#         signal: 1D numpy array of signal values.
#     Returns:
#         peak_freq: Frequency corresponding to the peak in the FFT.
#         SNR: Signal-to-noise ratio at the peak frequency.
#         FFT_magnitude: Magnitude of the FFT.
#         FFT_freqs: Frequencies corresponding to the FFT.
#     """
#     N = len(signal)
#     dt = time_vector[1] - time_vector[0]
#     fft_vals = np.fft.fft(signal - np.mean(signal))
#     fft_freqs = np.fft.fftfreq(N, dt)

#     positive_freqs = fft_freqs[:N//2]
#     positive_magnitude = np.abs(fft_vals[:N//2])
    
#     peak_idx = np.argmax(positive_magnitude[1:]) + 1  # Exclude DC component
#     peak_freq = positive_freqs[peak_idx]
#     peak_magnitude = positive_magnitude[peak_idx]

#     noise_floor = np.median(positive_magnitude)
#     SNR = peak_magnitude / noise_floor if noise_floor != 0 else np.inf

#     return peak_freq, SNR, positive_magnitude, positive_freqs

# def find_fft_peak(time_vector, signal, freq_min=0.5, freq_max=2.5, exclude_width_bins=1):
#     sig = np.asarray(signal)

#     N = sig.size

#     fft_vals = np.fft.rfft(sig - np.mean(sig))
#     freqs = np.fft.rfftfreq(N, d=(time_vector[1] - time_vector[0]))
#     mags = np.abs(fft_vals)

#     # Restrict to desired frequency range
#     freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
#     if not np.any(freq_mask):
#         return np.nan, np.nan, mags, freqs

#     idxs = np.where(freq_mask)[0]
#     peak_rel = np.argmax(mags[idxs])
#     peak_idx = idxs[peak_rel]
#     peak_freq = freqs[peak_idx]
#     peak_mag = mags[peak_idx]

#     # Exclude peak ± exclude_width_bins for noise floor
#     noise_mask = freq_mask.copy()
#     lo = max(0, peak_idx - exclude_width_bins)
#     hi = min(mags.size, peak_idx + exclude_width_bins + 1)
#     noise_mask[lo:hi] = False

#     # Noise floor: median of remaining magnitudes in the range
#     noise_mags = mags[noise_mask]
#     noise_floor = np.median(noise_mags) if noise_mags.size > 0 else 1e-12

#     SNR = peak_mag / noise_floor if noise_floor != 0 else np.inf
#     snr_db = 20 * np.log10(max(SNR, 1e-12))
#     bpm = peak_freq * 60.0
#     return peak_freq, SNR, mags, freqs

def find_fft_peak(
    time_vector,
    signal,
    freq_min=0.5,
    freq_max=2.5,
    window="hann",
    mainlobe_bins=4,
    prior_freq_hz=None,
    prior_tol_hz=0.3,
    min_prominence_db=3.0,
    min_snr_db=12.0,
):
    """
    Same fixes as SNRCalculator.calc_snr, adapted to the time_vector-based
    signature. Notable bug fix: the original computed `snr_db` but returned
    the *linear* `SNR` ratio as the second element -- if any downstream code
    (e.g. your plotting) assumed that value was already in dB, your SNR
    numbers/units would be inconsistent with the SNRCalculator class version.
    This version returns snr_db explicitly, plus a diagnostics dict.
 
    Returns
    -------
    peak_freq : float
    snr_db : float
    mags : np.ndarray
    freqs : np.ndarray
    diag : dict with keys 'prominence_db', 'low_confidence', 'second_peak_freq'
    """
    sig = np.asarray(signal, dtype=np.float64)
    N = sig.size
    dt = time_vector[1] - time_vector[0]
    fr = 1.0 / dt
 
    # --- detrend + window (reduces spectral leakage) ---
    sig = sig - np.mean(sig)
    if window is not None:
        if window == "hann":
            win = np.hanning(N)
        elif window == "hamming":
            win = np.hamming(N)
        else:
            raise ValueError(f"Unsupported window: {window}")
        win = win / (win.mean() + 1e-12)
        sig = sig * win
 
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / fr)
    mags = np.abs(fft_vals)
 
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(freq_mask):
        diag = {
            "prominence_db": np.nan,
            "low_confidence": True,
            "second_peak_freq": np.nan,
        }
        return np.nan, np.nan, mags, freqs, diag
 
    idxs = np.where(freq_mask)[0]
    band_mags = mags[idxs]
    band_freqs = freqs[idxs]
 
    # --- restrict search to a neighborhood of a prior/expected freq ---
    if prior_freq_hz is not None:
        search_mask = np.abs(band_freqs - prior_freq_hz) <= prior_tol_hz
        if not np.any(search_mask):
            search_mask = np.ones_like(band_freqs, dtype=bool)
    else:
        search_mask = np.ones_like(band_freqs, dtype=bool)
 
    search_idxs = np.where(search_mask)[0]
    peak_rel = search_idxs[np.argmax(band_mags[search_idxs])]
    peak_idx = idxs[peak_rel]
    peak_freq = freqs[peak_idx]
    peak_mag = mags[peak_idx]
 
    # --- exclusion zone scaled to window mainlobe width ---
    half_width = max(1, mainlobe_bins // 2)
    noise_mask = freq_mask.copy()
    lo = max(0, peak_idx - half_width)
    hi = min(mags.size, peak_idx + half_width + 1)
    noise_mask[lo:hi] = False
 
    noise_mags = mags[noise_mask]
    noise_floor = np.median(noise_mags) if noise_mags.size > 0 else 1e-12
 
    snr = peak_mag / noise_floor if noise_floor != 0 else np.inf
    snr_db = 20 * np.log10(max(snr, 1e-12))
 
    # --- prominence: compare peak to the runner-up local max in-band ---
    rival_mags = mags[noise_mask]
    rival_freqs = freqs[noise_mask]
    if rival_mags.size > 0:
        rival_rel = np.argmax(rival_mags)
        second_mag = rival_mags[rival_rel]
        second_freq = rival_freqs[rival_rel]
        prominence_db = 20 * np.log10(max(peak_mag / max(second_mag, 1e-12), 1e-12))
    else:
        second_freq = np.nan
        prominence_db = np.inf
 
    low_confidence = (prominence_db < min_prominence_db) or (snr_db < min_snr_db)
 
    diag = {
        "prominence_db": float(prominence_db),
        "low_confidence": bool(low_confidence),
        "second_peak_freq": float(second_freq) if second_freq == second_freq else np.nan,
    }
 
    return float(peak_freq), float(snr), mags, freqs, diag
 

# def SCOS_Calculation(
#     image_data,                # shape (H, W, N)
#     camera_gain,
#     mask,
#     black_level,
#     frame_rate,
#     backgroundImg,             # from calculate_dark_noise
#     darkVarPerWindow,          # from calculate_dark_noise
#     window_size=7,
#     var_sp=None,
#     save_dir=None
# ):
#     """
#     SCOS calculation workflow for both Gated and PaLS-iSCOS.
#     """
#     video_size = image_data.shape[:2]
#     assert mask.shape == video_size, "Mask size must match video size"

#     # Compute pixel non-uniformity
#     MAX_FRAMES_FOR_SP = min(600, image_data.shape[2])
#     mean_Isp = np.mean(image_data[:, :, :MAX_FRAMES_FOR_SP], axis=2) - black_level
#     var_sp = uniform_filter(mean_Isp**2, size=window_size) - uniform_filter(mean_Isp, size=window_size)**2

#     gainCalc = camera_gain
#     num_frames = image_data.shape[2]
#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     for frame in range(num_frames):
#         im = image_data[:, :, frame]
#         im = im - backgroundImg - black_level
#         meanIm = np.mean(im[mask])
#         varIm = uniform_filter(im**2, size=window_size) - uniform_filter(im, size=window_size)**2

#         K2_raw[frame] = np.mean(varIm[mask]) / (meanIm**2)
#         noise_correction = gainCalc * meanIm + darkVarPerWindow[mask] + var_sp[mask] + 1/12
#         K2_corrected[frame] = np.mean(varIm[mask] - noise_correction) / (meanIm**2)
#         BFi[frame] = 1 / K2_corrected[frame]

#     time_vector = np.arange(1, num_frames + 1) / frame_rate

#     results = {
#         'K2_raw': K2_raw,
#         'K2_corrected': K2_corrected,
#         'BFi': BFi,
#         'time_vector': time_vector,
#         'gainCalc': gainCalc
#     }
#     return results

import numpy as np
from scipy.ndimage import uniform_filter

# def PaLS_iSCOS_Calculation(
#     image_data, 
#     camera_gain,
#     mask,
#     black_level,
#     frame_rate,
#     backgroundImg, 
#     darkVarPerWindow, 
#     window_size=7,
#     ref_variance=None,        # NEW: σ_r^2
#     phase_variance=None       # NEW: σ_phi^2 (optional)
# ):
#     """
#     PaLS-iSCOS calculation consistent with theoretical noise model
#     """

#     video_size = image_data.shape[:2]
#     num_frames = image_data.shape[2]

#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     quantization_noise = 1/12.0

#     for frame in range(num_frames):
#         im = image_data[:, :, frame].astype(float)
#         im = im - backgroundImg - black_level

#         # --- Local statistics ---
#         mean_I = uniform_filter(im, size=window_size)
#         var_I = uniform_filter(im**2, size=window_size) - mean_I**2

#         valid_pixels = mask & (mean_I > 1e-6)

#         # --- Noise model (variance domain) ---
#         shot_noise = camera_gain * mean_I
#         read_dark_noise = darkVarPerWindow
#         quant_noise = quantization_noise

#         # Reference noise (optional)
#         ref_noise = ref_variance if ref_variance is not None else 0

#         # Phase noise (optional, requires Is, Ir estimate)
#         if phase_variance is not None:
#             # crude approximation: assume mean_I ≈ I_T
#             phase_noise = phase_variance * mean_I**2
#         else:
#             phase_noise = 0

#         total_noise_var = (
#             shot_noise +
#             read_dark_noise +
#             quant_noise +
#             ref_noise +
#             phase_noise
#         )

#         # --- Raw contrast ---
#         with np.errstate(divide='ignore', invalid='ignore'):
#             K2_map = np.where(valid_pixels, var_I / (mean_I**2), np.nan)

#         # --- Noise-corrected contrast ---
#         with np.errstate(divide='ignore', invalid='ignore'):
#             K2_corr_map = np.where(
#                 valid_pixels,
#                 (var_I - total_noise_var) / (mean_I**2),
#                 np.nan
#             )

#         # --- ROI averaging ---
#         K2_raw[frame] = np.nanmean(K2_map)
#         K2_corrected[frame] = np.nanmean(K2_corr_map)

#         # --- BFi inversion (better model) ---
#         if np.isfinite(K2_corrected[frame]) and K2_corrected[frame] > 0:
#             # Use high-exposure approximation safely
#             BFi[frame] = 1.0 / K2_corrected[frame]
#         else:
#             BFi[frame] = np.nan

#     time_vector = np.arange(1, num_frames + 1) / frame_rate

#     return {
#         'K2_raw': K2_raw,
#         'K2_corrected': K2_corrected,
#         'BFi': BFi,
#         'time_vector': time_vector,
#     }


import numpy as np
from scipy.ndimage import uniform_filter

def PaLS_iSCOS_Differential(
    image_data,
    mask,
    black_level,
    frame_rate,
    backgroundImg,
    window_size=7,
    baseline_frames=50   # number of initial frames for baseline
):
    """
    PaLS-iSCOS using differential (temporal) noise removal

    No variance subtraction — only baseline removal in K² domain
    """

    num_frames = image_data.shape[2]

    K2 = np.zeros(num_frames)
    K2_diff = np.zeros(num_frames)
    BFi = np.zeros(num_frames)

    # --- Step 1: compute K² for all frames ---
    for frame in range(num_frames):
        im = image_data[:, :, frame].astype(float)
        im = im - backgroundImg - black_level

        mean_I = uniform_filter(im, size=window_size)
        var_I = uniform_filter(im**2, size=window_size) - mean_I**2

        valid_pixels = mask & (mean_I > 1e-6)

        with np.errstate(divide='ignore', invalid='ignore'):
            K2_map = np.where(valid_pixels, var_I / (mean_I**2), np.nan)

        K2[frame] = np.nanmean(K2_map)

    # --- Step 2: baseline estimation ---
    baseline_value = np.nanmean(K2[:baseline_frames])

    # --- Step 3: differential contrast ---
    K2_diff = K2 - baseline_value

    # --- Step 4: BFi (relative only!) ---
    # Avoid division by zero / negative values
    for i in range(num_frames):
        if np.isfinite(K2_diff[i]) and K2_diff[i] > 0:
            BFi[i] = 1.0 / K2_diff[i]
        else:
            BFi[i] = np.nan

    time_vector = np.arange(1, num_frames + 1) / frame_rate

    return {
        'K2_raw': K2,
        'K2_corrected': K2_diff,
        'BFi': BFi,
        'baseline_K2': baseline_value,
        'time_vector': time_vector
    }

