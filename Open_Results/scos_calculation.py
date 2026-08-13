import numpy as np
import os
import scipy.io as sio
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

import numpy as np
import os
import scipy.io as sio
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter

import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion

import numpy as np
from scipy.ndimage import uniform_filter


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
#     import numpy as np
#     from scipy.ndimage import uniform_filter

#     # SPAD → gain = 1
#     actualGain = 1.0

#     Nmax_lookup = {1: 1, 4: 59, 6: 333, 7: 759, 8: 1701, 9: 3756, 10: 8218, 11: 17850, 12: 38529}
#     N_max = 2 ** nBits - 1 if not is_pileup else Nmax_lookup.get(nBits, 2 ** nBits - 1)

#     # --- Static Spatial Noise ---
#     MAX_FRAMES_FOR_SP = min(600, image_data.shape[2])
#     mean_im_raw = np.mean(image_data[:, :, :MAX_FRAMES_FOR_SP].astype(float), axis=2)

#     spIm = mean_im_raw - black_level - backgroundImg

#     mean_spIm_sq = uniform_filter(spIm**2, size=window_size)
#     sq_mean_spIm = uniform_filter(spIm, size=window_size)**2
#     spVar = mean_spIm_sq - sq_mean_spIm

#     num_frames = image_data.shape[2]
#     K2_raw = np.zeros(num_frames)
#     K2_corrected = np.zeros(num_frames)
#     BFi = np.zeros(num_frames)

#     for frame in range(num_frames):
#         im_raw = image_data[:, :, frame].astype(float) - black_level
#         Ic = im_raw - backgroundImg  # corrected intensity

#         # --- Global mean (as in your equation) ---
#         meanFrame = np.nanmean(Ic[mask])
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
#         K2_raw[frame] = np.nanmean(var_im[mask]) / meanFrame_sq
#         K2_corrected[frame] = np.nanmean(corrected_var[mask]) / meanFrame_sq

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
#         'mask': mask
#     }
    
def SCOS_Calculation_CMOS(
    image_data,
    camera_gain,
    mask,
    black_level,
    frame_rate,
    backgroundImg,
    darkVarPerWindow,
    window_size=7,
    nBits=8,
    save_dir=None,
    mode="frame",
    t_line=8e-6
):
    """
    SCOS calculation workflow implementing noise models and 3-sigma bad pixel removal.
    mode: "frame" (default) - average over ROI, "line" - average over each row (columns)
    """
    video_size = image_data.shape[0:2]
    assert mask.shape == video_size, "Mask size must match video size"

    MAX_FRAMES_FOR_SP = min(600, image_data.shape[2])
    mean_Isp = np.mean(image_data[:, :, :MAX_FRAMES_FOR_SP].astype(float), axis=2) - backgroundImg
    mean_Isp[mean_Isp <= 0] = 1e-5

    var_sp = uniform_filter(mean_Isp**2, size=window_size) - uniform_filter(mean_Isp, size=window_size)**2
    K2_sp_map = var_sp / (mean_Isp**2)

    good_pixel_mask = mask
    num_frames = image_data.shape[2]

    if mode == "line":
        K2_raw = np.zeros((num_frames, image_data.shape[0]))
        K2_corrected = np.zeros((num_frames, image_data.shape[0]))
        BFi = np.zeros((num_frames, image_data.shape[0]))
    else:
        K2_raw = np.zeros(num_frames)
        K2_corrected = np.zeros(num_frames)
        BFi = np.zeros(num_frames)

    for frame in range(num_frames):
        im = image_data[:, :, frame].astype(float)
        im = im - backgroundImg

        mean_I_win = uniform_filter(im, size=window_size)
        var_I_win = uniform_filter(im**2, size=window_size) - mean_I_win**2

        # Apply mask: only use pixels within mask with sufficient signal
        valid = good_pixel_mask & (mean_I_win > 1e-5)

        with np.errstate(divide='ignore', invalid='ignore'):
            K2_raw_map = np.where(mean_I_win**2 > 0, var_I_win / (mean_I_win**2 + 1e-10), 0)
            K2_s_map = np.where(mean_I_win > 0, camera_gain / (mean_I_win + 1e-10), 0)
            K2_rq_map = np.where(mean_I_win**2 > 0, darkVarPerWindow / (mean_I_win**2 + 1e-10), 0)
            K2_f_map = K2_raw_map - K2_s_map - K2_rq_map - K2_sp_map

        if mode == "line":
            for row in range(im.shape[0]):
                valid_row = valid[row, :]
                if np.any(valid_row):
                    # Average all noise and signal terms over columns for this row
                    K2_raw[frame, row] = np.nanmean(K2_raw_map[row, :][valid_row])
                    K2_s = np.nanmean(K2_s_map[row, :][valid_row])
                    K2_rq = np.nanmean(K2_rq_map[row, :][valid_row])
                    K2_sp = np.nanmean(K2_sp_map[row, :][valid_row])
                    K2_corr = K2_raw[frame, row] - K2_s - K2_rq - K2_sp
                    K2_corrected[frame, row] = K2_corr
                    BFi[frame, row] = 1 / K2_corr if K2_corr != 0 else 0
                else:
                    K2_raw[frame, row] = np.nan
                    K2_corrected[frame, row] = np.nan
                    BFi[frame, row] = np.nan
        else:
            K2_raw[frame] = np.nanmean(K2_raw_map[valid])
            K2_corrected[frame] = np.nanmean(K2_f_map[valid])
            K2_corr_val = K2_corrected[frame]
            BFi[frame] = 1 / K2_corr_val if K2_corr_val != 0 else 0

    if mode == "line":
        # Flatten: (frames, rows) → (frames*rows,)
        num_rows = image_data.shape[0]
        K2_raw_flat = K2_raw.flatten()
        K2_corrected_flat = K2_corrected.flatten()
        BFi_flat = BFi.flatten()
        
        # Non-uniform time vector: rows sample faster than frames
        # Each row within a frame is sampled at frame_rate * num_rows
        time_vector = np.zeros(num_frames * num_rows)
        for frame_idx in range(num_frames):
            for row_idx in range(num_rows):
                flat_idx = frame_idx * num_rows + row_idx
                # Time = (frame_duration) * frame_idx + (row_time within frame)
                time_vector[flat_idx] = frame_idx / frame_rate + row_idx * t_line
        
        return {
            'K2_raw': K2_raw_flat,
            'K2_corrected': K2_corrected_flat,
            'BFi': BFi_flat,
            'time_vector': time_vector,
            'gainCalc': camera_gain,
            'image_data': image_data,
            'mask': mask
        }
    else:
        time_vector = np.arange(1, num_frames + 1) / frame_rate

        return {
            'K2_raw': K2_raw,
            'K2_corrected': K2_corrected,
            'BFi': BFi,
            'time_vector': time_vector,
            'gainCalc': camera_gain,
            'image_data': image_data,
            'mask': mask
        }



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

def square_roi_from_mask(mask):
    """Extract inscribed square ROI from circular mask."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    
    # Make square
    side = min(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    
    y0 = max(0, cy - side // 2)
    y1 = min(mask.shape[0], y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(mask.shape[1], x0 + side)
    
    square_mask = np.zeros_like(mask, dtype=bool)
    square_mask[y0:y1, x0:x1] = True
    return square_mask

def calculate_dark_noise(dark_frames, window_size=7, mode="frame", mask=None):
    """
    Calculate background image and dark variance per window from dark frames.
    mode: "frame" (default) - average over all, "line" - per row (columns)
    mask: optional binary mask to restrict computation to valid pixels
    """
    backgroundImg = np.mean(dark_frames, axis=2)
    var_r = np.var(dark_frames, axis=2, ddof=1)
    darkVarPerWindow = uniform_filter(var_r, size=window_size)
    
    if mode == "line":
        if mask is not None:
            # Keep the dark noise map 2D so it broadcasts with the frame/window maps.
            # Outside the ROI, mark values as NaN so they are ignored downstream.
            darkVarPerWindow = np.where(mask, darkVarPerWindow, np.nan)
        return backgroundImg, darkVarPerWindow
    else:
        return backgroundImg, darkVarPerWindow


def theoretical_gain(bit_depth, capacity_ke=10.4):
    gain= capacity_ke*1e3 / (2 ** bit_depth - 1)
    return gain
    
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

def find_fft_peak(time_vector, signal, freq_min=0.5, freq_max=2.5, exclude_width_bins=1):
    """
    Find the dominant FFT peak and SNR in a specified frequency range (default 0.5–2.5 Hz).
    - Peak: max magnitude in [freq_min, freq_max]
    - Noise: median of magnitudes in [freq_min, freq_max], excluding the peak ± exclude_width_bins
    Returns: (peak_freq, SNR, magnitudes, freqs)
    """
    import numpy as np

    t = np.asarray(time_vector)
    sig = np.asarray(signal)
    if sig.size < 2 or t.size < 2:
        return np.nan, np.nan, np.array([]), np.array([])

    dt = np.mean(np.diff(t))
    N = sig.size

    fft_vals = np.fft.rfft(sig - np.mean(sig))
    freqs = np.fft.rfftfreq(N, dt)
    mags = np.abs(fft_vals)

    # Restrict to desired frequency range
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(freq_mask):
        return np.nan, np.nan, mags, freqs

    idxs = np.where(freq_mask)[0]
    peak_rel = np.argmax(mags[idxs])
    peak_idx = idxs[peak_rel]
    peak_freq = freqs[peak_idx]
    peak_mag = mags[peak_idx]

    # Exclude peak ± exclude_width_bins for noise floor
    noise_mask = freq_mask.copy()
    lo = max(0, peak_idx - exclude_width_bins)
    hi = min(mags.size, peak_idx + exclude_width_bins + 1)
    noise_mask[lo:hi] = False

    # Noise floor: median of remaining magnitudes in the range
    noise_mags = mags[noise_mask]
    noise_floor = np.median(noise_mags) if noise_mags.size > 0 else 1e-12

    SNR = peak_mag / noise_floor if noise_floor != 0 else np.inf
    return peak_freq, SNR, mags, freqs





