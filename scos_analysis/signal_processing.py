
from scipy.signal import butter, filtfilt, medfilt
import numpy as np
from scipy.interpolate import interp1d

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

def estimate_fs(time_vector):
    t = np.asarray(time_vector, dtype=float)
    if t.size < 2:
        return np.nan
    dt = np.mean(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return 1.0 / dt


def apply_butter_bandpass(time_vector, trace, f_low_hz=0.5, f_high_hz=10.0, order=4):
    trace = np.asarray(trace, dtype=float)
    trace = oversample_then_median_back_to_original(time_vector, trace, upsample_factor=4)
    fs = estimate_fs(time_vector)
    if not np.isfinite(fs) or trace.size < 8:
        return trace, {"fs": fs, "butter_band": (None, None)}

    nyquist = 0.5 * fs
    low = float(np.clip(f_low_hz, 1e-3, 0.95 * nyquist))
    high = float(np.clip(f_high_hz, low + 1e-3, 0.95 * nyquist))

    if not (high > low):
        return trace.copy(), {"fs": fs, "butter_band": (None, None)}

    try:
        b, a = butter(order, [low, high], btype="bandpass", fs=fs)
        padlen = 3 * max(len(a), len(b))
        if trace.size > padlen:
            filtered = filtfilt(b, a, trace)
        else:
            filtered = trace.copy()
    except Exception:
        filtered = trace.copy()
        low, high = None, None

    
    return filtered, {"fs": fs, "butter_band": (low, high)}


def _odd(n):
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def oversample_then_median_back_to_original(time_vector, trace, upsample_factor=4):
    t = np.asarray(time_vector, dtype=float)
    x = np.asarray(trace, dtype=float)

    if t.size < 5 or x.size != t.size:
        return x.copy()

    valid = np.isfinite(t) & np.isfinite(x)
    if np.count_nonzero(valid) < 5:
        return x.copy()

    t_v = t[valid]
    x_v = x[valid]
    if not np.all(np.diff(t_v) > 0):
        order = np.argsort(t_v)
        t_v = t_v[order]
        x_v = x_v[order]

    n_up = int(max(t_v.size, upsample_factor * t_v.size))
    t_up = np.linspace(t_v[0], t_v[-1], n_up)
    x_up = np.interp(t_up, t_v, x_v)

    k = _odd(round(n_up / 100.0))
    if k >= n_up:
        k = _odd(n_up - 2)
    x_up_med = medfilt(x_up, kernel_size=k)

    x_back = np.interp(t, t_up, x_up_med)
    x_back[~np.isfinite(x_back)] = x[~np.isfinite(x_back)]
    return x_back

def compute_fwhm(signal, time_vec):
    """Robust FWHM calculation."""
    half_max = np.max(signal) / 2
    indices_above_half = np.where(signal >= half_max)[0]
    if len(indices_above_half) < 2:
        return np.nan
    left_idx = indices_above_half[0]
    right_idx = indices_above_half[-1]
    fwhm = time_vec[right_idx] - time_vec[left_idx]
    return fwhm
