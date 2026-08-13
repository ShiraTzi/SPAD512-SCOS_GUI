"""
SCOS Analysis Pipeline for Gated SCOS and PaLS-iSCOS Measurements
Processes multi-offset recordings with automatic ROI extraction and SNR/BPM calculation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import re
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Assuming SCOS_Calculation is in a local file scos_calculation.py
try:
    from scos_calculation import SCOS_Calculation, find_mask
except ImportError:
    def SCOS_Calculation(*args, **kwargs):
        print("Warning: scos_calculation module not found.")
        return {}
    def find_mask(*args, **kwargs):
        print("Warning: scos_calculation module not found.")
        return None
    
import numpy as np


MASTER_ONLY = False  # set True to restrict every recalculated ROI mask to the master-SPAD half of the sensor

class SNRCalculator:
    """
    Improvements over the original implementation:

    1. Windowing (Hann by default) before the FFT to reduce spectral leakage.
       This is the single biggest fix for the "SNR looks fine but BPM jumps"
       problem -- a rectangular window smears energy from strong low-freq
       drift/noise into neighboring bins, which can occasionally out-compete
       the true cardiac peak.

    2. Noise-floor exclusion width now scales with the window's mainlobe
       width (in bins) instead of a fixed `exclude_width_bins=1`. With a Hann
       window the true peak's energy spreads over ~4 bins, not 1, so the old
       code was leaking peak energy into its own "noise" estimate.

    3. Peak-picking can be constrained to a neighborhood around a prior/
       expected frequency (`prior_freq_hz`), instead of a free argmax over
       the whole band. This directly prevents the "wrong-bin jump" failure
       mode (e.g. locking onto 0.53 Hz instead of ~1.3 Hz).

    4. Prominence check: peak magnitude is compared not just to the median
       noise floor but also to the second-highest local maximum in the band.
       A low prominence (peak barely taller than the runner-up) is flagged
       as low-confidence even if SNR_db looks acceptable, since that's
       exactly the situation where trial-to-trial noise flips which bin wins.

    5. Returns extra diagnostics (`prominence_db`, `low_confidence`,
       `second_peak_freq`) so you can build a rejection/flagging rule
       downstream instead of silently trusting every estimate.
    """

    @staticmethod
    def calc_snr(
        signal,
        fr,
        freq_min=0.5,
        freq_max=2.5,
        window="hann",
        mainlobe_bins=4,
        prior_freq_hz=None,
        prior_tol_hz=0.3,
        min_prominence_db=3.0,
        min_snr_db=12.0,
    ):
        sig = np.asarray(signal, dtype=np.float64)
        N = sig.size

        # --- detrend + window (reduces spectral leakage) ---
        sig = sig - np.mean(sig)
        if window is not None:
            if window == "hann":
                win = np.hanning(N)
            elif window == "hamming":
                win = np.hamming(N)
            else:
                raise ValueError(f"Unsupported window: {window}")
            # normalize so windowing doesn't change overall magnitude scale
            win = win / (win.mean() + 1e-12)
            sig = sig * win

        fft_vals = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(N, d=1.0 / fr)
        mags = np.abs(fft_vals)

        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        if not np.any(freq_mask):
            return {
                "SNR_db": np.nan,
                "BPM": np.nan,
                "f_sound": np.nan,
                "prominence_db": np.nan,
                "low_confidence": True,
                "second_peak_freq": np.nan,
            }

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
        bpm = peak_freq * 60.0

        # --- prominence: compare peak to the runner-up local max in-band ---
        rival_mask = noise_mask.copy()
        rival_mags = mags[rival_mask]
        rival_freqs = freqs[rival_mask]
        if rival_mags.size > 0:
            rival_rel = np.argmax(rival_mags)
            second_mag = rival_mags[rival_rel]
            second_freq = rival_freqs[rival_rel]
            prominence_db = 20 * np.log10(max(peak_mag / max(second_mag, 1e-12), 1e-12))
        else:
            second_freq = np.nan
            prominence_db = np.inf

        low_confidence = (prominence_db < min_prominence_db) or (snr_db < min_snr_db)

        return {
            "SNR_db": float(snr_db),
            "BPM": float(bpm),
            "f_sound": float(peak_freq),
            "prominence_db": float(prominence_db),
            "low_confidence": bool(low_confidence),
            "second_peak_freq": float(second_freq) if second_freq == second_freq else np.nan,
        }

    # def calc_snr(self, signal, fr, num_harmonics=3):
    #     signal = np.asarray(signal, dtype=float)
    #     if len(signal) < 2:
    #         return {'SNR_db': np.nan, 'BPM': np.nan, 'f_sound': np.nan}
        
    #     signal = signal - np.mean(signal)
    #     fft_vals = np.fft.fft(signal) / fr
    #     freqs = np.fft.fftfreq(len(signal), d=1.0 / fr)
    #     half = len(fft_vals) // 2
    #     Pxx = np.abs(fft_vals[:half])
    #     f = freqs[:half]
        
    #     mask_fund = (f > 0.5) & (f < 2.5)
    #     mask_harm = (f > 0.5) & (f < 10.0)
    #     if not np.any(mask_fund) or not np.any(mask_harm):
    #         return {'SNR_db': np.nan, 'BPM': np.nan, 'f_sound': np.nan}
            
    #     f_fund = f[mask_fund]
    #     Pxx_fund = Pxx[mask_fund]
    #     f_harm = f[mask_harm]
    #     Pxx_harm = Pxx[mask_harm]
        
    #     idx_fund_local = int(np.argmax(Pxx_fund))
    #     f_max = float(f_fund[idx_fund_local])
    #     bpm = f_max * 60.0
        
    #     harmonic_indices = [int(np.argmin(np.abs(f_harm - f_max)))]
    #     for n in range(2, num_harmonics + 1):
    #         target = f_max * n
    #         if target >= 10.0:
    #             continue
    #         idx_candidates = np.where((f_harm >= target - 0.1) & (f_harm <= target + 0.1))[0]
    #         if idx_candidates.size > 0:
    #             local = int(np.argmax(Pxx_harm[idx_candidates]))
    #             harmonic_indices.append(int(idx_candidates[local]))
        
    #     harmonic_indices = sorted(set(harmonic_indices))
    #     Pxx_sound = float(np.mean(Pxx_harm[harmonic_indices])) if harmonic_indices else 1e-12
    #     noise_indices = [i for i in range(len(f_harm)) if i not in set(harmonic_indices)]
        
    #     if noise_indices:
    #         Pxx_noise = float(np.median(Pxx_harm[noise_indices]))
    #         Pxx_noise = max(Pxx_noise, 1e-12)
    #     else:
    #         Pxx_noise = 1e-12
            
    #     snr = Pxx_sound / Pxx_noise
    #     snr_db = 20 * np.log10(max(snr, 1e-12))
    #     return {
    #         'SNR_db': float(snr_db),
    #         'BPM': float(bpm),
    #         'f_sound': float(f_max)
    #     }

class SCOSAnalyzer:
    def __init__(self, base_path, output_csv='scos_analysis_results.csv', recalc_SCOS=False, master_only=None):
        self.base_path = Path(base_path)
        self.output_csv = output_csv
        self.results = []
        self.recalc_SCOS = recalc_SCOS
        self.master_only = MASTER_ONLY if master_only is None else master_only
        self.snr_calc = SNRCalculator()
        self.metric_map = {
            'K2_raw': 'raw',
            'K2_corrected': 'corrected',
            'BFi': 'BFi',
            'intensity': 'intensity'
        }
        self.mask=None  # Initialize mask to None; will be computed when needed
    @staticmethod
    def apply_master_only_mask(mask, master_only=MASTER_ONLY):
        """Restrict a mask to the master-SPAD half of the sensor (left half of columns)."""
        if not master_only or mask is None:
            return mask
        master_mask = np.zeros_like(mask, dtype=bool)
        master_mask[:mask.shape[1] // 2, :] = True  # columns, not rows
        return mask & master_mask
    
    def extract_power_level(self, name):
        """Extract power level string (e.g., -30%) from folder name."""
        match = re.search(r'([-+]?[0-9]+%|[0-9]+p[0-9]+%)', name)
        return match.group(1) if match else None

    def find_matching_tpsf(self, power_level):
        """Find TPSF file/folder matching the power level string."""
        if power_level:
            candidates = list(self.base_path.glob(f'TPSF_*{power_level}*'))
            if candidates: return candidates[0]
            candidates = list(self.base_path.rglob(f'TPSF_*{power_level}*'))
            if candidates: return candidates[0]
        
        all_tpsf = list(self.base_path.glob('TPSF_*'))
        if all_tpsf: return all_tpsf[0]
        all_tpsf_recursive = list(self.base_path.rglob('TPSF_*'))
        if all_tpsf_recursive: return all_tpsf_recursive[0]
        return None

    def get_peak_ns_from_metadata(self, folder_path):
        """Extracts peak position in ns from metadata.json."""
        if not folder_path or not folder_path.exists():
            return None
        metadata = self.load_metadata(folder_path)
        if metadata.get("measurement_type") == "TPSF" or metadata.get("measurement_type") == "IRF":
            time_vec=self.load_npy(folder_path, 't_axis.npy', allow_none=True)
            decon_data=self.load_npy(folder_path, 'deconvolved.npy', allow_none=True)
            if time_vec is not None and decon_data is not None:
                peak_idx = np.argmax(decon_data)
                peak_value = time_vec[peak_idx]
            # print("start time:", time_vec[0], "peak idx:", peak_idx, "peak value:", peak_value)
            return peak_value
        return None   

    def plot_results(self, time_vec, k2_raw, k2_corr, bfi, intensity, save_path, filename, offset_value=None):
        """Plot and save SCOS results (K2_raw, K2_corrected, BFi, intensity)"""
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 12))
            title_suffix = f" (Offset: {offset_value})" if offset_value is not None else ""
            
            axes[0].plot(time_vec, k2_raw, linewidth=1.3, color='blue')
            axes[0].set_title(f'K2_raw Per Frame{title_suffix}')
            axes[0].set_ylabel('K2_raw')

            axes[1].plot(time_vec, k2_corr, linewidth=1.3, color='green')
            axes[1].set_title(f'K2_corrected Per Frame{title_suffix}')
            axes[1].set_ylabel('K2_corrected')

            axes[2].plot(time_vec, bfi if bfi is not None else [], linewidth=1.3, color='red')
            axes[2].set_title(f'BFi Per Frame{title_suffix}')
            axes[2].set_ylabel('BFi')

            axes[3].plot(time_vec, intensity, linewidth=1.3, color='orange')
            axes[3].set_title(f'Intensity Per Frame{title_suffix}')
            axes[3].set_ylabel('Intensity')

            for ax in axes:
                ax.set_xlabel('Time [s]')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = Path(save_path) / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Plot saved: {output_path}")
        except Exception as e:
            print(f"    Error saving plot: {e}")

    def find_folders(self, pattern):
        return sorted([d for d in self.base_path.glob(pattern) if d.is_dir()])

    def load_metadata(self, folder_path):
        metadata_path = folder_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def load_npy(self, folder_path, filename, allow_none=False):
        file_path = folder_path / filename
        if file_path.exists():
            return np.load(file_path, allow_pickle=True)
        if allow_none: return None
        raise FileNotFoundError(f"Missing {filename} in {folder_path}")

    def save_npy(self, folder_path, filename, arr):
        np.save(folder_path / filename, arr)

    def process_scos_scan(self, folder_path, force_recalc=False):
        power_level = self.extract_power_level(folder_path.name)
        print(f"\nProcessing Gated SCOS: {folder_path.name}")

        try:
            if force_recalc:
                image_data = self.load_npy(folder_path, 'image_data.npy')
                image_data = np.asarray(image_data)
                if image_data.dtype == object:
                    image_data = image_data.astype(float)  # coerce object-dtype 4D arrays

                gate_offsets = self.load_npy(folder_path, 'gate_offsets.npy', allow_none=True)
                metadata = self.load_metadata(folder_path)
                backgroundImg = self.load_npy(folder_path, 'backgroundImg.npy', allow_none=True)
                darkVarPerWindow = self.load_npy(folder_path, 'darkVarPerWindow.npy', allow_none=True)

                n_offsets = image_data.shape[0] if image_data.ndim == 4 else 1

                if self.mask is None:  # Compute mask once, from the first offset's mean frame
                    representative_stack = image_data[0] if image_data.ndim == 4 else image_data
                    mean_image = np.mean(representative_stack, axis=2)
                    mask = find_mask(mean_image)
                    mask = self.apply_master_only_mask(mask, self.master_only)
                    self.mask = mask
                else:
                    mask = self.mask

                self.save_npy(folder_path, 'roi_mask.npy', mask)
                print(f"    Recomputed ROI mask: {mask.sum()} / {mask.size} pixels"
                    f"{' (MASTER_ONLY)' if self.master_only else ''}")

                all_time, all_k2_raw, all_k2_corr, all_BFi, all_int = [], [], [], [], []

                for i in range(n_offsets):
                    stack = image_data[i] if image_data.ndim == 4 else image_data
                    print(f"    Offset {i+1}/{n_offsets}: stack={stack.shape}, mask pixels={mask.sum()}, "
                        f"backgroundImg={backgroundImg.shape if backgroundImg is not None else 'N/A'}, "
                        f"darkVarPerWindow={darkVarPerWindow.shape if darkVarPerWindow is not None else 'N/A'}")

                    res = SCOS_Calculation(
                        image_data=stack,               # <-- now a proper 3D (H, W, n_frames) stack
                        camera_gain=metadata.get('gain', 1.0),
                        mask=mask,                       # <-- 2D mask now matches a 3D stack correctly
                        black_level=0,
                        frame_rate=metadata.get('frame_rate', 100.0),
                        nBits=metadata.get('bit_depth', 8),
                        is_pileup=metadata.get('is_pileup_correction', False),
                        backgroundImg=backgroundImg if backgroundImg is not None else 0,
                        darkVarPerWindow=darkVarPerWindow if darkVarPerWindow is not None else 0,
                    )
                    all_time.append(res['time_vector'])
                    all_k2_raw.append(res['K2_raw'])
                    all_k2_corr.append(res['K2_corrected'])
                    all_BFi.append(res.get('BFi', None))
                    all_int.append(np.mean(stack, axis=(0, 1)))

                self.save_npy(folder_path, 'time_vector.npy', np.array(all_time, dtype=object))
                self.save_npy(folder_path, 'K2_raw.npy', np.array(all_k2_raw, dtype=object))
                self.save_npy(folder_path, 'K2_corrected.npy', np.array(all_k2_corr, dtype=object))
                self.save_npy(folder_path, 'BFi.npy', np.array(all_BFi, dtype=object))
                self.save_npy(folder_path, 'intensity.npy', np.array(all_int, dtype=object))
            else:
                all_time = self.load_npy(folder_path, 'time_vector.npy')
                all_k2_raw = self.load_npy(folder_path, 'K2_raw.npy')
                all_k2_corr = self.load_npy(folder_path, 'K2_corrected.npy')
                all_BFi = self.load_npy(folder_path, 'BFi.npy', allow_none=True)
                all_int = self.load_npy(folder_path, 'intensity.npy')
                gate_offsets = self.load_npy(folder_path, 'gate_offsets.npy', allow_none=True)

            mean_ints = [np.mean(intensity) for intensity in all_int]
            peak_idx = np.argmax(mean_ints)
            TPSF_folder = self.find_matching_tpsf(power_level)
            peak_offset = self.get_peak_ns_from_metadata(TPSF_folder) if TPSF_folder else peak_idx
            # print (f"   Gate offsets: {gate_offsets if gate_offsets is not None else 'N/A'}, Peak offset: {peak_offset}")
            print(f" TPSF folder: {TPSF_folder if TPSF_folder else 'N/A'}, image data folder: {folder_path}")
            n_offsets = len(all_k2_corr)
            for i in range(n_offsets):
                offset = gate_offsets[i] if gate_offsets is not None else i
                rel_offset_ns = float(offset - peak_offset) if gate_offsets is not None else None
                print(f"    Processing offset {i+1}/{n_offsets} (Gate offset: {offset}, Relative offset: {rel_offset_ns})")
                time_vec = np.asarray(all_time[i]).flatten()
                fr_value = 1.0 / np.mean(np.diff(time_vec)) if len(time_vec) > 1 else 100.0
                
                metrics = [
                    ('K2_raw', np.asarray(all_k2_raw[i]).flatten()),
                    ('K2_corrected', np.asarray(all_k2_corr[i]).flatten()),
                    ('intensity', np.asarray(all_int[i]).flatten())
                ]
                
                plot_fn = f"{folder_path.name}_offset{i}_results.png"
                self.plot_results(time_vec, metrics[0][1], metrics[1][1], 
                                 np.asarray(all_BFi[i]).flatten() if all_BFi is not None and all_BFi[i] is not None else None, 
                                 np.asarray(all_int[i]).flatten(), folder_path, plot_fn, offset_value=offset)

                for m_key, arr in metrics:
                    snr = self.snr_calc.calc_snr(arr, fr=fr_value)
                    self.results.append({
                        'folder': folder_path.name,
                        'gate_offset': float(offset),
                        'gate_index': int(i),
                        'relative_offset_ns': rel_offset_ns,
                        'method': 'Gated SCOS',
                        'metric': self.metric_map.get(m_key, m_key),
                        'SNR_dB': snr['SNR_db'],
                        'BPM': snr['BPM'],
                        'f_max_Hz': snr['f_sound']
                    })
            print(f"✓ Processed {n_offsets} offsets")
        except Exception as e:
            print(f"✗ Error: {e}")

    def process_scos_single(self, folder_path, ref_name=None, force_recalc=False):
        folder_label = f"{ref_name}/{folder_path.name}" if ref_name else folder_path.name
        print(f"\nProcessing PaLS-iSCOS: {folder_label}")
        
        try:
            # Timing calculation logic
            power_level = self.extract_power_level(ref_name) if ref_name else None
            tpsf_folder = self.find_matching_tpsf(power_level)
            
            # Find IRF folder within the specific ref folder
            ref_folder_path = folder_path.parent
            irf_folders = list(ref_folder_path.glob('IRF_*'))
            irf_folder = irf_folders[0] if irf_folders else None

            tpsf_peak = self.get_peak_ns_from_metadata(tpsf_folder)
            irf_peak = self.get_peak_ns_from_metadata(irf_folder)
            print(f"    [Timing] TPSF folder: {tpsf_folder}, IRF folder: {irf_folder}")
            
            rel_offset_ns = None
            if tpsf_peak is not None and irf_peak is not None:
                rel_offset_ns = round(float(irf_peak - tpsf_peak), 4)
                print(f"    [Timing] IRF: {irf_peak}ns, TPSF: {tpsf_peak}ns -> Relative: {rel_offset_ns}ns")
            else:
                print("    [Timing] Could not determine relative offset (missing IRF or TPSF metadata)")
                print(f"    IRF folder: {irf_folder}, TPSF folder: {tpsf_folder}")
            if force_recalc:
                image_data = self.load_npy(folder_path, 'image_data.npy')
                metadata = self.load_metadata(folder_path)
                backgroundImg = self.load_npy(folder_path, 'backgroundImg.npy', allow_none=True)
                darkVarPerWindow = self.load_npy(folder_path, 'darkVarPerWindow.npy', allow_none=True)
                if self.mask is None:  # Compute mask only if it hasn't been computed yet
                    mask = find_mask(np.mean(image_data, axis=2))
                    mask = self.apply_master_only_mask(mask, self.master_only)
                    self.mask = mask
                else:
                    mask = self.mask
                self.save_npy(folder_path, 'roi_mask.npy', mask)
                print(f"    Recomputed ROI mask: {mask.sum()} / {mask.size} pixels"
                    f"{' (MASTER_ONLY)' if self.master_only else ''}")
                print(f"Loaded image data = {image_data.shape}, mask shape = {mask.shape if mask is not None else 'N/A'}, backgroundImg shape = {backgroundImg.shape if backgroundImg is not None or any(backgroundImg) else 'N/A'}, darkVarPerWindow shape = {darkVarPerWindow.shape if darkVarPerWindow is not None or any(darkVarPerWindow) else 'N/A'}")
                # print(f"mean backgroundImg: {np.mean(backgroundImg) if backgroundImg is not None else 'N/A'}, mean darkVarPerWindow: {np.mean(darkVarPerWindow) if darkVarPerWindow is not None else 'N/A'}")
                res = SCOS_Calculation(
                    image_data=image_data,
                    camera_gain=metadata.get('gain', 1.0),
                    mask=mask if mask is not None else np.ones(image_data.shape[:2], dtype=bool),
                    black_level=0,
                    frame_rate=metadata.get('frame_rate', 100.0),
                    nBits=metadata.get('bit_depth', 8),
                    is_pileup=metadata.get('is_pileup_correction', False),
                    backgroundImg=backgroundImg,
                    darkVarPerWindow=darkVarPerWindow
                )
                self.save_npy(folder_path, 'time_vector.npy', res['time_vector'])
                self.save_npy(folder_path, 'K2_raw.npy', res['K2_raw'])
                self.save_npy(folder_path, 'K2_corrected.npy', res['K2_corrected'])
                self.save_npy(folder_path, 'BFi.npy', res.get('BFi', None))
                self.save_npy(folder_path, 'intensity.npy', np.mean(image_data, axis=(0, 1)))
            
            res_data = {
                'time_vector': self.load_npy(folder_path, 'time_vector.npy'),
                'K2_raw': self.load_npy(folder_path, 'K2_raw.npy'),
                'K2_corrected': self.load_npy(folder_path, 'K2_corrected.npy'),
                'BFi': self.load_npy(folder_path, 'BFi.npy', allow_none=True),
                'intensity': self.load_npy(folder_path, 'intensity.npy')
            }

            time_vec = np.asarray(res_data['time_vector']).flatten()
            fr_value = 1.0 / np.mean(np.diff(time_vec)) if len(time_vec) > 1 else 100.0
            
            plot_fn = f"{folder_path.name}_results.png"
            self.plot_results(time_vec, res_data['K2_raw'], res_data['K2_corrected'], 
                             res_data['BFi'], res_data['intensity'], folder_path, plot_fn, offset_value=rel_offset_ns)

            for m_key in ['K2_raw', 'K2_corrected', 'intensity']:
                arr = np.asarray(res_data[m_key]).flatten()
                snr = self.snr_calc.calc_snr(arr, fr=fr_value)
                self.results.append({
                    'folder': folder_label,
                    'gate_offset': None,
                    'gate_index': None,
                    'relative_offset_ns': rel_offset_ns,
                    'method': 'PaLS-iSCOS',
                    'metric': self.metric_map.get(m_key, m_key),
                    'SNR_dB': snr['SNR_db'],
                    'BPM': snr['BPM'],
                    'f_max_Hz': snr['f_sound']
                })
            print(f"✓ Processed {folder_label}")
        except Exception as e:
            print(f"✗ Error: {e}")

    def run(self):
        print("\n" + "="*60)
        print("SCOS ANALYSIS PIPELINE")
        print("="*60)
        
        scos_scans = self.find_folders('SCOS_Scan_*')
        for folder in scos_scans:
            self.process_scos_scan(folder, force_recalc=self.recalc_SCOS)

        ref_folders = self.find_folders('ref-*')
        if not ref_folders:
            ref_folders = self.find_folders('*gate*')
        if not ref_folders:
            ref_folders = self.find_folders('*ref*')
        for ref_folder in ref_folders:
            singles = [d for d in ref_folder.glob('SCOS_Single_*') if d.is_dir()]
            for single in singles:
                self.process_scos_single(single, ref_name=ref_folder.name, force_recalc=self.recalc_SCOS)

        if self.results:
            df = pd.DataFrame(self.results)
            cols = ['folder', 'gate_offset', 'gate_index', 'relative_offset_ns', 'method', 'metric', 'SNR_dB', 'BPM', 'f_max_Hz']
            df = df.rename(columns={'SNR_db': 'SNR_dB', 'f_sound': 'f_max_Hz'})
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(self.output_csv, index=False)
            print(f"\n✓ Summary saved to {self.output_csv}")
        else:
            print("\nNo results generated")

if __name__ == '__main__':
    # For command-line usage: python scos_analysis.py [base_path] [--recalc]
    import sys
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    recalc_SCOS = '--recalc' in sys.argv
    master_only = '--master-only' in sys.argv
    output_csv = str(Path(base_path) / 'scos_analysis_results.csv')
    
    print(f"Starting SCOS analysis in: {base_path}")
    if recalc_SCOS:
        print("Recalculation of SCOS metrics is ENABLED")
    if master_only:
        print("MASTER_ONLY mode is ENABLED (restricting ROI to master-SPAD half)")
    analyzer = SCOSAnalyzer(base_path, output_csv=output_csv, recalc_SCOS=recalc_SCOS, master_only=master_only)
    analyzer.run()
    
    
#     """
# SCOS Analysis Pipeline for Gated SCOS and PaLS-iSCOS Measurements
# Processes multi-offset recordings with automatic ROI extraction and SNR/BPM calculation
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# from scos_calculation import SCOS_Calculation

# # SNR/BPM calculation (copied from autocorr_analysis.py)
# class SNRCalculator:
#     @staticmethod
#     def calc_snr(signal, fr, num_harmonics=3):
#         signal = np.asarray(signal, dtype=float)
#         signal = signal - np.mean(signal)
#         fft_vals = np.fft.fft(signal) / fr
#         freqs = np.fft.fftfreq(len(signal), d=1.0 / fr)
#         half = len(fft_vals) // 2
#         Pxx = np.abs(fft_vals[:half])
#         f = freqs[:half]
#         mask_fund = (f > 0.5) & (f < 2.5)
#         mask_harm = (f > 0.5) & (f < 10.0)
#         if not np.any(mask_fund) or not np.any(mask_harm):
#             return {'SNR_db': np.nan, 'BPM': np.nan, 'f_sound': np.nan}
#         f_fund = f[mask_fund]
#         Pxx_fund = Pxx[mask_fund]
#         f_harm = f[mask_harm]
#         Pxx_harm = Pxx[mask_harm]
#         idx_fund_local = int(np.argmax(Pxx_fund))
#         f_max = float(f_fund[idx_fund_local])
#         bpm = f_max * 60.0
#         harmonic_indices = [int(np.argmin(np.abs(f_harm - f_max)))]
#         for n in range(2, num_harmonics + 1):
#             target = f_max * n
#             if target >= 10.0:
#                 continue
#             idx_candidates = np.where((f_harm >= target - 0.1) & (f_harm <= target + 0.1))[0]
#             if idx_candidates.size > 0:
#                 local = int(np.argmax(Pxx_harm[idx_candidates]))
#                 harmonic_indices.append(int(idx_candidates[local]))
#         harmonic_indices = sorted(set(harmonic_indices))
#         Pxx_sound = float(np.mean(Pxx_harm[harmonic_indices])) if harmonic_indices else 1e-12
#         noise_indices = [i for i in range(len(f_harm)) if i not in set(harmonic_indices)]
#         if noise_indices:
#             Pxx_noise = float(np.median(Pxx_harm[noise_indices]))
#             Pxx_noise = max(Pxx_noise, 1e-12)
#         else:
#             Pxx_noise = 1e-12
#         snr = Pxx_sound / Pxx_noise
#         snr_db = 20 * np.log10(max(snr, 1e-12))
#         return {
#             'SNR_db': float(snr_db),
#             'BPM': float(bpm),
#             'f_sound': float(f_max)
#         }

# class SCOSAnalyzer:
#         def extract_power_level(self, name):
#             """Extract power level string (e.g., -30%) from folder name."""
#             import re
#             match = re.search(r'([-+]?[0-9]+%|[0-9]+p[0-9]+%)', name)
#             return match.group(1) if match else None

#         def find_matching_tpsf(self, power_level):
#             """Find TPSF file/folder matching the power level string. If not found, use the only TPSF if just one exists."""
#             if power_level:
#                 candidates = list(self.base_path.glob(f'TPSF_*{power_level}*'))
#                 if candidates:
#                     return candidates[0]
#                 candidates = list(self.base_path.rglob(f'TPSF_*{power_level}*'))
#                 if candidates:
#                     return candidates[0]
#             # Fallback: if only one TPSF exists, use it
#             all_tpsf = list(self.base_path.glob('TPSF_*'))
#             if len(all_tpsf) == 1:
#                 return all_tpsf[0]
#             all_tpsf_recursive = list(self.base_path.rglob('TPSF_*'))
#             if len(all_tpsf_recursive) == 1:
#                 return all_tpsf_recursive[0]
#             return None
    
#         def __init__(self, base_path, output_csv='scos_analysis_results.csv', recalc_SCOS=False):
#             self.base_path = Path(base_path)
#             self.output_csv = output_csv
#             self.results = []
#             self.recalc_SCOS = recalc_SCOS
#             self.snr_calc = SNRCalculator()
            
#         def plot_results(self, time_vec, k2_raw, k2_corr, bfi, intensity, save_path, filename, offset_value=None):
#             """Plot and save SCOS results (K2_raw, K2_corrected, BFi, intensity)"""
#             import matplotlib.pyplot as plt
#             try:
#                 fig, axes = plt.subplots(4, 1, figsize=(12, 12))
#                 title_suffix = f" (Offset: {offset_value})" if offset_value is not None else ""
#                 axes[0].plot(time_vec, k2_raw, linewidth=1.3, color='blue')
#                 axes[0].set_title(f'K2_raw Per Frame{title_suffix}')
#                 axes[0].set_xlabel('Time [s]')
#                 axes[0].set_ylabel('K2_raw')
#                 axes[0].grid(True, alpha=0.3)

#                 axes[1].plot(time_vec, k2_corr, linewidth=1.3, color='green')
#                 axes[1].set_title(f'K2_corrected Per Frame{title_suffix}')
#                 axes[1].set_xlabel('Time [s]')
#                 axes[1].set_ylabel('K2_corrected')
#                 axes[1].grid(True, alpha=0.3)

#                 axes[2].plot(time_vec, bfi, linewidth=1.3, color='red')
#                 axes[2].set_title(f'BFi Per Frame{title_suffix}')
#                 axes[2].set_xlabel('Time [s]')
#                 axes[2].set_ylabel('BFi')
#                 axes[2].grid(True, alpha=0.3)

#                 axes[3].plot(time_vec, intensity, linewidth=1.3, color='orange')
#                 axes[3].set_title(f'Intensity Per Frame{title_suffix}')
#                 axes[3].set_xlabel('Time [s]')
#                 axes[3].set_ylabel('Intensity')
#                 axes[3].grid(True, alpha=0.3)

#                 plt.tight_layout()
#                 output_path = Path(save_path) / filename
#                 plt.savefig(output_path, dpi=150, bbox_inches='tight')
#                 plt.close()
#             except Exception as e:
#                 print(f"    Error saving plot: {e}")
#         """Main pipeline for SCOS analysis on Gated SCOS and PaLS-iSCOS measurements"""
    

#         def find_folders(self, pattern):
#             """Find folders matching pattern"""
#             return sorted([d for d in self.base_path.glob(pattern) if d.is_dir()])

#         def load_metadata(self, folder_path):
#             metadata_path = folder_path / 'metadata.json'
#             if metadata_path.exists():
#                 with open(metadata_path, 'r') as f:
#                     return json.load(f)
#             return {}

#         def load_npy(self, folder_path, filename, allow_none=False):
#             import os
#             # print(f"[DEBUG] Listing files in: {folder_path}")
#             # try:
#             #     print(os.listdir(folder_path))
#             # except Exception as e:
#             #     print(f"[DEBUG] Could not list files: {e}")
#             file_path = folder_path / filename
#             # print(f"[DEBUG] Checking for file: {file_path}")
#             if file_path.exists():
#                 return np.load(file_path, allow_pickle=True)
#             if allow_none:
#                 return None
#             raise FileNotFoundError(f"Missing {filename} in {folder_path}")

#         def save_npy(self, folder_path, filename, arr):
#             np.save(folder_path / filename, arr)

#         def process_scos_scan(self, folder_path, force_recalc=False):
#             # Extract power level from SCOS_Scan folder name
#             power_level = self.extract_power_level(folder_path.name)
#             tpsf_path = self.find_matching_tpsf(power_level) if power_level else None
#             print(f"    [TPSF Matching] Power level: {power_level}, TPSF: {tpsf_path}")
#             print(f"\n{'='*60}")
#             print(f"Processing SCOS_Scan: {folder_path.name}")
#             print(f"{'='*60}")
#             try:
#                 if force_recalc:
#                     print("    Force recalculation enabled: Recomputing SCOS metrics for all offsets...")
#                     image_data = self.load_npy(folder_path, 'image_data.npy')
#                     gate_offsets = self.load_npy(folder_path, 'gate_offsets.npy', allow_none=True)
#                     mask = self.load_npy(folder_path, 'roi_mask.npy', allow_none=True)
#                     backgroundImg = self.load_npy(folder_path, 'backgroundImg.npy', allow_none=True)
#                     darkVarPerWindow = self.load_npy(folder_path, 'darkVarPerWindow.npy', allow_none=True)
#                     metadata = self.load_metadata(folder_path)
#                     frame_rate = metadata.get('frame_rate', 100.0)
#                     camera_gain = metadata.get('gain', 1.0)
#                     bit_depth = metadata.get('bit_depth', 8)
#                     is_pileup = metadata.get('is_pileup_correction', False)
#                     n_offsets = image_data.shape[0] if image_data.ndim == 4 else 1
#                     all_time, all_k2_raw, all_k2_corr, all_BFi, all_int = [], [], [], [], []
#                     for i in range(n_offsets):
#                         stack = image_data[i] if image_data.ndim == 4 else image_data
#                         res = SCOS_Calculation(
#                             image_data=stack,
#                             camera_gain=camera_gain,
#                             mask=mask if mask is not None else np.ones(stack.shape[:2], dtype=bool),
#                             black_level=0,
#                             frame_rate=frame_rate,
#                             backgroundImg=backgroundImg if backgroundImg is not None else 0,
#                             darkVarPerWindow=darkVarPerWindow if darkVarPerWindow is not None else 0,
#                             nBits=bit_depth,
#                             is_pileup=is_pileup
#                         )
#                         all_time.append(res['time_vector'])
#                         all_k2_raw.append(res['K2_raw'])
#                         all_k2_corr.append(res['K2_corrected'])
#                         all_BFi.append(res.get('BFi', None))
#                         all_int.append(np.mean(stack, axis=(0, 1)))
#                     self.save_npy(folder_path, 'time_vector.npy', np.array(all_time, dtype=object))
#                     self.save_npy(folder_path, 'K2_raw.npy', np.array(all_k2_raw, dtype=object))
#                     self.save_npy(folder_path, 'K2_corrected.npy', np.array(all_k2_corr, dtype=object))
#                     self.save_npy(folder_path, 'BFi.npy', np.array(all_BFi, dtype=object))
#                     self.save_npy(folder_path, 'intensity.npy', np.array(all_int, dtype=object))
#                 else:
#                     print("    Loading existing SCOS metrics for all offsets (if available)...")
#                     all_time = self.load_npy(folder_path, 'time_vector.npy')
#                     all_k2_raw = self.load_npy(folder_path, 'K2_raw.npy')
#                     all_k2_corr = self.load_npy(folder_path, 'K2_corrected.npy')
#                     all_BFi = self.load_npy(folder_path, 'BFi.npy', allow_none=True)
#                     all_int = self.load_npy(folder_path, 'intensity.npy')
#                     gate_offsets = self.load_npy(folder_path, 'gate_offsets.npy', allow_none=True)
#                 n_offsets = len(all_k2_corr)
#                 for i in range(n_offsets):
#                     offset = gate_offsets[i] if gate_offsets is not None and i < len(gate_offsets) else i
#                     gate_index = i
#                     # Each metric: K2_raw, K2_corrected, BFi, intensity
#                     k2_raw = np.asarray(all_k2_raw[i]).flatten() if all_k2_raw[i] is not None else None
#                     k2_corr = np.asarray(all_k2_corr[i]).flatten() if all_k2_corr[i] is not None else None
#                     bfi = np.asarray(all_BFi[i]).flatten() if all_BFi is not None and all_BFi[i] is not None else None
#                     intensity = np.asarray(all_int[i]).flatten() if all_int[i] is not None else None
#                     time_vec = np.asarray(all_time[i]).flatten() if all_time[i] is not None else None
#                     # Plot all metrics for this offset
#                     if time_vec is not None and k2_raw is not None and k2_corr is not None and bfi is not None and intensity is not None:
#                         plot_filename = f"{folder_path.name}_offset{gate_index}_results.png"
#                         self.plot_results(time_vec, k2_raw, k2_corr, bfi, intensity, folder_path, plot_filename, offset_value=offset)
#                     # Try to get frame rate for SNR/BPM
#                     fr_value = 1.0 / np.mean(np.diff(time_vec)) if time_vec is not None and len(time_vec) > 1 else 100.0
#                     metrics = [
#                         ('K2_raw', k2_raw),
#                         ('K2_corrected', k2_corr),
#                         ('BFi', bfi),
#                         ('intensity', intensity)
#                     ]
#                     for metric, arr in metrics:
#                         if arr is None:
#                             continue
#                         snr = self.snr_calc.calc_snr(arr, fr=fr_value)
#                         for idx, val in enumerate(arr):
#                             self.results.append({
#                                 'folder': folder_path.name,
#                                 'gate_offset': float(offset),
#                                 'gate_index': int(gate_index),
#                                 'relative_offset_ns': None,
#                                 'method': 'SCOS',
#                                 'metric': metric,
#                                 'frame': idx,
#                                 'value': float(val),
#                                 'SNR_dB': snr['SNR_db'],
#                                 'BPM': snr['BPM'],
#                                 'f_max_Hz': snr['f_sound']
#                             })
#                 print(f"✓ Processed {n_offsets} offsets in {folder_path.name}")
#             except Exception as e:
#                 print(f"✗ Error: {e}")

#         def process_scos_single(self, folder_path, ref_name=None, force_recalc=False):
#             # Extract power level from parent ref folder name
#             power_level = self.extract_power_level(ref_name) if ref_name else None
#             tpsf_path = self.find_matching_tpsf(power_level) if power_level else None
#             print(f"    [TPSF Matching] Power level: {power_level}, TPSF: {tpsf_path}")
#             print(f"\n{'='*60}")
#             print(f"Processing SCOS_Single: {folder_path.name}")
#             print(f"{'='*60}")
#             try:
#                 if force_recalc:
#                     print("    Force recalculation enabled: Recomputing SCOS metrics...")
#                     # Always recalculate and overwrite, regardless of file existence
#                     image_data = self.load_npy(folder_path, 'image_data.npy')
#                     mask = self.load_npy(folder_path, 'roi_mask.npy', allow_none=True)
#                     backgroundImg = self.load_npy(folder_path, 'backgroundImg.npy', allow_none=True)
#                     darkVarPerWindow = self.load_npy(folder_path, 'darkVarPerWindow.npy', allow_none=True)
#                     metadata = self.load_metadata(folder_path)
#                     frame_rate = metadata.get('frame_rate', 100.0)
#                     camera_gain = metadata.get('gain', 1.0)
#                     bit_depth = metadata.get('bit_depth', 8)
#                     is_pileup = metadata.get('is_pileup_correction', False)
#                     res = SCOS_Calculation(
#                         image_data=image_data,
#                         camera_gain=camera_gain,
#                         mask=mask if mask is not None else np.ones(image_data.shape[:2], dtype=bool),
#                         black_level=0,
#                         frame_rate=frame_rate,
#                         backgroundImg=backgroundImg if backgroundImg is not None else 0,
#                         darkVarPerWindow=darkVarPerWindow if darkVarPerWindow is not None else 0,
#                         nBits=bit_depth,
#                         is_pileup=is_pileup
#                     )
#                     self.save_npy(folder_path, 'time_vector.npy', res['time_vector'])
#                     self.save_npy(folder_path, 'K2_raw.npy', res['K2_raw'])
#                     self.save_npy(folder_path, 'K2_corrected.npy', res['K2_corrected'])
#                     self.save_npy(folder_path, 'BFi.npy', res.get('BFi', None))
#                     self.save_npy(folder_path, 'intensity.npy', np.mean(image_data, axis=(0, 1)))
#                     # Always plot and save after calculation
#                     k2_raw = np.asarray(res['K2_raw']).flatten() if res['K2_raw'] is not None else None
#                     k2_corr = np.asarray(res['K2_corrected']).flatten() if res['K2_corrected'] is not None else None
#                     bfi = np.asarray(res['BFi']).flatten() if isinstance(res['BFi'], np.ndarray) else None
#                     intensity = np.asarray(res['intensity']).flatten() if res['intensity'] is not None else None
#                     time_vec = np.asarray(res['time_vector']).flatten() if res['time_vector'] is not None else None
#                     if time_vec is not None and k2_raw is not None and k2_corr is not None and bfi is not None and intensity is not None:
#                         plot_filename = f"{folder_path.name}_results.png"
#                         self.plot_results(time_vec, k2_raw, k2_corr, bfi, intensity, folder_path, plot_filename)
#                 else:
#                     print("    Loading existing SCOS metrics (if available)...")
#                     res = {
#                         'time_vector': self.load_npy(folder_path, 'time_vector.npy'),
#                         'K2_raw': self.load_npy(folder_path, 'K2_raw.npy'),
#                         'K2_corrected': self.load_npy(folder_path, 'K2_corrected.npy'),
#                         'BFi': self.load_npy(folder_path, 'BFi.npy', allow_none=True),
#                         'intensity': self.load_npy(folder_path, 'intensity.npy')
#                     }
#                     k2_raw = np.asarray(res['K2_raw']).flatten() if res['K2_raw'] is not None else None
#                     k2_corr = np.asarray(res['K2_corrected']).flatten() if res['K2_corrected'] is not None else None
#                     bfi = np.asarray(res['BFi']).flatten() if isinstance(res['BFi'], np.ndarray) else None
#                     intensity = np.asarray(res['intensity']).flatten() if res['intensity'] is not None else None
#                     time_vec = np.asarray(res['time_vector']).flatten() if res['time_vector'] is not None else None
#                     if time_vec is not None and k2_raw is not None and k2_corr is not None and bfi is not None and intensity is not None:
#                         plot_filename = f"{folder_path.name}_results.png"
#                         self.plot_results(time_vec, k2_raw, k2_corr, bfi, intensity, folder_path, plot_filename)
#                 # Only one result per SCOS_Single measurement for CSV
#                 folder_label = f"{ref_name}/{folder_path.name}" if ref_name else folder_path.name
#                 self.results.append({
#                     'folder': folder_label,
#                     'method': 'SCOS_Single',
#                     'mean_K2_raw': float(np.mean(k2_raw)) if k2_raw is not None else None,
#                     'mean_K2_corrected': float(np.mean(k2_corr)) if k2_corr is not None else None,
#                     'mean_BFi': float(np.mean(bfi)) if bfi is not None else None,
#                     'mean_intensity': float(np.mean(intensity)) if intensity is not None else None
#                 })
#                 print(f"✓ Processed {folder_label}")
#             except Exception as e:
#                 print(f"✗ Error: {e}")
#             # Extract power level from parent ref folder name
#             power_level = self.extract_power_level(ref_name) if ref_name else None
#             tpsf_path = self.find_matching_tpsf(power_level) if power_level else None
#             print(f"    [TPSF Matching] Power level: {power_level}, TPSF: {tpsf_path}")
#             print(f"\n{'='*60}")
#             print(f"Processing SCOS_Single: {folder_path.name}")
#             print(f"{'='*60}")
#             try:
#                 if force_recalc:
#                     print("    Force recalculation enabled: Recomputing SCOS metrics...")
#                     # Always recalculate and overwrite, regardless of file existence
#                     image_data = self.load_npy(folder_path, 'image_data.npy')
#                     mask = self.load_npy(folder_path, 'roi_mask.npy', allow_none=True)
#                     backgroundImg = self.load_npy(folder_path, 'backgroundImg.npy', allow_none=True)
#                     darkVarPerWindow = self.load_npy(folder_path, 'darkVarPerWindow.npy', allow_none=True)
#                     metadata = self.load_metadata(folder_path)
#                     frame_rate = metadata.get('frame_rate', 100.0)
#                     camera_gain = metadata.get('gain', 1.0)
#                     bit_depth = metadata.get('bit_depth', 8)
#                     is_pileup = metadata.get('is_pileup_correction', False)
#                     res = SCOS_Calculation(
#                         image_data=image_data,
#                         camera_gain=camera_gain,
#                         mask=mask if mask is not None else np.ones(image_data.shape[:2], dtype=bool),
#                         black_level=0,
#                         frame_rate=frame_rate,
#                         backgroundImg=backgroundImg if backgroundImg is not None else 0,
#                         darkVarPerWindow=darkVarPerWindow if darkVarPerWindow is not None else 0,
#                         nBits=bit_depth,
#                         is_pileup=is_pileup
#                     )
#                     self.save_npy(folder_path, 'time_vector.npy', res['time_vector'])
#                     self.save_npy(folder_path, 'K2_raw.npy', res['K2_raw'])
#                     self.save_npy(folder_path, 'K2_corrected.npy', res['K2_corrected'])
#                     self.save_npy(folder_path, 'BFi.npy', res.get('BFi', None))
#                     self.save_npy(folder_path, 'intensity.npy', np.mean(image_data, axis=(0, 1)))
#                 else:
#                     print("    Loading existing SCOS metrics (if available)...")
#                     res = {
#                         'time_vector': self.load_npy(folder_path, 'time_vector.npy'),
#                         'K2_raw': self.load_npy(folder_path, 'K2_raw.npy'),
#                         'K2_corrected': self.load_npy(folder_path, 'K2_corrected.npy'),
#                         'BFi': self.load_npy(folder_path, 'BFi.npy', allow_none=True),
#                         'intensity': self.load_npy(folder_path, 'intensity.npy')
#                     }
#                 # Output per-frame, per-metric, include ref_name if provided
#                 k2_raw = np.asarray(res['K2_raw']).flatten() if res['K2_raw'] is not None else None
#                 k2_corr = np.asarray(res['K2_corrected']).flatten() if res['K2_corrected'] is not None else None
#                 bfi = np.asarray(res['BFi']).flatten() if isinstance(res['BFi'], np.ndarray) else None
#                 intensity = np.asarray(res['intensity']).flatten() if res['intensity'] is not None else None
#                 time_vec = np.asarray(res['time_vector']).flatten() if res['time_vector'] is not None else None
#                 folder_label = f"{ref_name}/{folder_path.name}" if ref_name else folder_path.name
#                 # Plot all metrics for this SCOS_Single, save in SCOS_Single folder
#                 if time_vec is not None and k2_raw is not None and k2_corr is not None and bfi is not None and intensity is not None:
#                     plot_filename = f"{folder_path.name}_results.png"
#                     self.plot_results(time_vec, k2_raw, k2_corr, bfi, intensity, folder_path, plot_filename)
#                 fr_value = 1.0 / np.mean(np.diff(time_vec)) if time_vec is not None and len(time_vec) > 1 else 100.0
#                 metrics = [
#                     ('K2_raw', k2_raw),
#                     ('K2_corrected', k2_corr),
#                     ('BFi', bfi),
#                     ('intensity', intensity)
#                 ]
#                 for metric, arr in metrics:
#                     if arr is None:
#                         continue
#                     snr = self.snr_calc.calc_snr(arr, fr=fr_value)
#                     for idx, val in enumerate(arr):
#                         self.results.append({
#                             'folder': folder_label,
#                             'gate_offset': None,
#                             'gate_index': None,
#                             'relative_offset_ns': None,
#                             'method': 'SCOS_Single',
#                             'metric': metric,
#                             'frame': idx,
#                             'value': float(val),
#                             'SNR_dB': snr['SNR_db'],
#                             'BPM': snr['BPM'],
#                             'f_max_Hz': snr['f_sound']
#                         })
#                 print(f"✓ Processed {folder_label}")
#             except Exception as e:
#                 print(f"✗ Error: {e}")

#         def run(self):
#             print("\n" + "="*60)
#             print("SCOS ANALYSIS PIPELINE")
#             print("="*60)
#             print(f"Base path: {self.base_path}\n")
#             if not self.recalc_SCOS:
#                 print("Using existing SCOS metrics if available. Use --recalc to force recomputation.")
#             else:
#                 print("--recalc specified: All SCOS metrics will be recomputed and resaved.")
#             # Process SCOS_Scan
#             scos_scans = self.find_folders('SCOS_Scan_*')
#             print(f"Found {len(scos_scans)} SCOS_Scan folders")
#             for folder in scos_scans:
#                 self.process_scos_scan(folder, force_recalc=self.recalc_SCOS)

#             # Process all SCOS_Single folders inside each ref-* folder
#             ref_folders = self.find_folders('ref-*')
#             scos_singles = []
#             for ref_folder in ref_folders:
#                 singles = [d for d in ref_folder.glob('SCOS_Single_*') if d.is_dir()]
#                 for single in singles:
#                     # Save plots/results in each SCOS_Single folder
#                     self.process_scos_single(single, ref_name=ref_folder.name, force_recalc=self.recalc_SCOS)
#                 scos_singles.extend(singles)
#             print(f"\nFound {len(scos_singles)} SCOS_Single folders (inside ref-*)")

#             # Save results: one row per SCOS measurement
#             if self.results:
#                 df = pd.DataFrame(self.results)
#                 df.to_csv(self.output_csv, index=False)
#                 print(f"\n{'='*60}")
#                 print(f"✓ Results saved to {self.output_csv}")
#                 print(f"{'='*60}")
#                 print(f"\nSummary ({len(df)} SCOS measurements):")
#                 print(df.describe())
#             else:
#                 print("\nNo results generated")

# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python scos_analysis.py <base_path> [--recalc]")
#         sys.exit(1)
#     base_path = sys.argv[1]
#     recalc_SCOS = '--recalc' in sys.argv
#     if recalc_SCOS:
#         print("Force recalculation enabled: All SCOS metrics will be recomputed even if existing files are found.")
#     else:
#         print("Using existing SCOS metrics if available. Use --recalc to force recomputation.")
#     output_csv = str(Path(base_path) / 'scos_analysis_results.csv')
#     analyzer = SCOSAnalyzer(base_path, output_csv=output_csv, recalc_SCOS=recalc_SCOS)
#     analyzer.run()
