
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import scipy as sp
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def _normalize_frames_by_polynomial(frames):
    """
    Normalize frames by dividing by a 2nd-degree polynomial fit to the mean image.
    Fits polynomials independently along X and Y axes and averages the result.
    
    Returns normalized frames with polynomial illumination component removed.
    """
    frames = np.asarray(frames, dtype=float)
    
    # Compute mean image
    mean_img = np.mean(frames, axis=0)
    height, width = mean_img.shape
    
    # Fit 2nd degree polynomial along X-axis (for each row)
    x_indices = np.arange(width)
    poly_x = np.zeros((height, width))
    for i in range(height):
        coeffs = np.polyfit(x_indices, mean_img[i, :], 2)
        poly_x[i, :] = np.polyval(coeffs, x_indices)
    
    # Fit 2nd degree polynomial along Y-axis (for each column)
    y_indices = np.arange(height)
    poly_y = np.zeros((height, width))
    for j in range(width):
        coeffs = np.polyfit(y_indices, mean_img[:, j], 2)
        poly_y[:, j] = np.polyval(coeffs, y_indices)
    
    # Average the two polynomial surfaces
    poly_surface = (poly_x + poly_y) / 2.0
    
    # Avoid division by very small values
    poly_surface = np.maximum(poly_surface, 1e-10)
    
    # Apply normalization to all frames
    normalized_frames = np.zeros_like(frames)
    for i in range(frames.shape[0]):
        normalized_frames[i] = frames[i] / poly_surface
    
    return normalized_frames
def _plot_2d_example_correlation(frame):
    """Plot a representative 2D autocorrelation and its center profiles."""
    frame = np.asarray(frame, dtype=float)
    ac2d = sp.signal.fftconvolve(frame, frame[::-1, ::-1], mode='full')
    if np.max(ac2d) > 0:
        ac2d = ac2d / np.max(ac2d)

    cy = ac2d.shape[0] // 2
    cx = ac2d.shape[1] // 2
    profile_x = ac2d[cy, :]
    profile_y = ac2d[:, cx]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Example 2D Autocorrelation')
    plt.imshow(ac2d, cmap='gray')
    plt.colorbar()
    plt.axhline(cy, color='red', linestyle='--', linewidth=1)
    plt.axvline(cx, color='red', linestyle='--', linewidth=1)

    plt.subplot(1, 2, 2)
    plt.title('Center Profiles')
    plt.plot(profile_x, label='horizontal')
    plt.plot(profile_y, label='vertical')
    plt.legend()
    plt.tight_layout()
    plt.show()

def _central_fwhm_from_profile(profile):
    """Return central-lobe FWHM (in pixels) as a robust fallback."""
    ac = np.asarray(profile, dtype=float)
    if ac.size < 5:
        return np.nan

    ac = ac - np.min(ac)
    peak = np.max(ac)
    if peak <= 0:
        return np.nan

    ac = ac / peak
    center = int(np.argmax(ac))
    half = 0.5

    # Find left crossing
    left = center
    while left > 0 and ac[left] >= half:
        left -= 1

    if left == center:
        return np.nan

    # Find right crossing
    right = center
    while right < ac.size - 1 and ac[right] >= half:
        right += 1

    if right == center:
        return np.nan
        # Linear interpolation around half-maximum crossings
    def _interp_x(i0, i1):
        y0 = ac[i0]
        y1 = ac[i1]
        if y1 == y0:
            return float(i0)
        return float(i0 + (half - y0) * (i1 - i0) / (y1 - y0))

    left_x = _interp_x(max(left, 0), min(left + 1, ac.size - 1))
    right_x = _interp_x(max(right - 1, 0), min(right, ac.size - 1))
    width = right_x - left_x
    return float(width) if width > 0 else np.nan

def _frame_speckle_coherence_2d(frame, speckle_size=1, plot_example=False):
    """Frame-wise speckle coherence using 2D autocorrelation.
    
    Computes two complementary metrics:
    1. Speckle size ampliude (peak metric)
    2. Central-lobe FWHM (width metric)
    """
    frame = np.asarray(frame, dtype=float)
    # # Normalize frames by dividing by polynomial fit to mean image
    # frame = _normalize_frames_by_polynomial(frame)
    ac2d = sp.signal.fftconvolve(frame, frame[::-1, ::-1], mode='full')
    if np.max(ac2d) > 0:
        ac2d = ac2d / np.max(ac2d)

    cy = ac2d.shape[0] // 2
    cx = ac2d.shape[1] // 2
    profile_x = ac2d[cy, :]
    profile_y = ac2d[:, cx]

    if plot_example:
        _plot_2d_example_correlation(frame)

    # Metric 1: Speckle size amplitude (peak metric)
    # idx_x = np.arange(profile_x.size)
    # non_center_x = idx_x[np.abs(idx_x - cx) > 0]
    # px = float(np.max(profile_x[non_center_x])) if non_center_x.size > 0 else np.nan
    indx_x= speckle_size+np.size(profile_x)//2
    px= float(profile_x[indx_x]) if 0 <= indx_x < profile_x.size else np.nan
    
    indx_y= speckle_size+np.size(profile_y)//2
    py= float(profile_y[indx_y]) if 0 <= indx_y < profile_y.size else np.nan
    # idx_y = np.arange(profile_y.size)
    # non_center_y = idx_y[np.abs(idx_y - cy) > 0]
    # py = float(np.max(profile_y[non_center_y])) if non_center_y.size > 0 else np.nan

    # Metric 2: Central-lobe FWHM
    dx = _central_fwhm_from_profile(profile_x)
    dy = _central_fwhm_from_profile(profile_y)

    # Compile results
    size_vals = [v for v in [dx, dy] if np.isfinite(v)]
    peak_vals = [v for v in [px, py] if np.isfinite(v)]

    size_val = float(np.mean(size_vals)) if len(size_vals) > 0 else np.nan
    peak_val = float(np.mean(peak_vals)) if len(peak_vals) > 0 else np.nan
    return size_val, peak_val, speckle_size

def square_roi_from_mask(mask):
    """Extract inscribed square ROI from circular mask."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    
    # Make square
    side = min(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    
    y0 = max(0, cy - side // 2)
    y1 = min(mask.shape[0], y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(mask.shape[1], x0 + side)
    
    return y0, y1, x0, x1

def speckle_size_gs_framewise(frames,speckle_size=1, upsample_factor=5, plot_example=False, example_frame_index=0, example_line_index=0):
    """
    Compute one speckle-size value per frame using 2D autocorrelation.
    Always returns both sizes and peak values.

    Returns:
        tuple: (sizes_arr, peaks_arr) - frame-wise speckle sizes and peak values
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3:
        raise ValueError("frames must have shape (num_frames, height, width).")

    sizes = []
    peak_values = []
    for i in tqdm(range(frames.shape[0]), desc="Frame-wise speckle size (2d)"):
        frame = frames[i]
        size_val, peak_val, speckle_size = _frame_speckle_coherence_2d(
            frame, 
            speckle_size=speckle_size,
            plot_example=plot_example and i == int(example_frame_index)
        )
        sizes.append(size_val)
        peak_values.append(peak_val)

    sizes_arr = np.asarray(sizes, dtype=float)
    peaks_arr = np.asarray(peak_values, dtype=float)
    return sizes_arr, peaks_arr, speckle_size


