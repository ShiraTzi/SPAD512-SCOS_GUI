"""
Sensitivity test analysis for SCOS_Single measurements.

This script compares a baseline segment against a tourniquet segment
(and, optionally, a post-release "late" segment) using corrected rBFI,
defined as 1 / K^2_corrected, or the raw contrast (sqrt(K2_corrected)).

It records the time gate relative to the experiment's gate-1 reference
and the coherence gate relative to the matched IRF peak for ref/late rows.

This is a merge of the previous two-stage and three-stage sensitivity
scripts, with two new options:

* stages=2 or stages=3
    - base and tourniquet segments are ALWAYS present.
    - stages=2: the tourniquet segment is the last `window_seconds`
      of the recording (this is the behavior of the original
      "two-stage" script).
    - stages=3: the tourniquet segment is a *middle* window (padded by
      `window_cut_transition_seconds` on either side), and a separate
      post-release "late" segment (the last `window_seconds` of the
      recording) is also computed (this is the behavior of the
      original "three-stage" script).

* correct_intensity=True/False (default False)
    - When True, an intensity-artifact correction (derived from the
      Gated measurements, then interpolated across gate/coherence
      offsets for ISCOS/PaLS-iSCOS) is applied to the tourniquet
      window's contrast/rBFI, exactly as in the original three-stage
      script's calibration pass.
    - When False (default), no correction is applied
      (contrast_multiplier = rbfi_multiplier = 1.0), matching the
      original two-stage script's behavior.

ROI handling
------------
Same fix as `scos_analysis.py`: instead of trusting a (possibly stale)
`roi_mask.npy` saved on disk, the mask is recomputed directly from the
image data via `find_mask()` whenever `--recalc` is used. The mask is
computed once per run (from the first folder that needs it) and reused
for every subsequent folder, since all measurements share the same
physical camera field of view.
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Assuming SCOS_Calculation / find_mask are in a local file scos_calculation.py
try:
    from scos_calculation import SCOS_Calculation, find_mask
except ImportError:
    def SCOS_Calculation(*args, **kwargs):
        print("Warning: scos_calculation module not found.")
        return {}
    def find_mask(*args, **kwargs):
        print("Warning: scos_calculation module not found.")
        return None


MASTER_ONLY = False  # set True to restrict every recalculated ROI mask to the master-SPAD half of the sensor


class SensitivityTestAnalyzer:
    def __init__(
        self,
        base_path,
        output_csv="sensitivity_test_results.csv",
        recalc_SCOS=False,
        window_seconds=30.0,
        metric="contrast",
        stages=2,
        correct_intensity=False,
        window_cut_transition_seconds=5.0,
        master_only=None,
    ):
        self.base_path = Path(base_path)
        self.output_csv = output_csv
        self.recalc_SCOS = recalc_SCOS
        self.window_seconds = float(window_seconds)
        self.window_cut_transition_seconds = float(window_cut_transition_seconds)
        self.results = []
        self._time_gate_reference_ps = None
        self.metric = str(metric).lower() if metric is not None else "contrast"

        if int(stages) not in (2, 3):
            raise ValueError("stages must be 2 or 3")
        self.stages = int(stages)
        self.correct_intensity = bool(correct_intensity)

        # ROI mask cache, computed once per run and reused across folders (see fix above)
        self.mask = None
        self.master_only = MASTER_ONLY if master_only is None else master_only

        # Intensity-artifact calibration model (only populated if correct_intensity=True)
        self.gated_baseline_model = None

    # ------------------------------------------------------------------ #
    # Generic I/O helpers
    # ------------------------------------------------------------------ #
    def load_metadata(self, folder_path):
        metadata_path = folder_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as file_handle:
                return json.load(file_handle)
        return {}

    def load_npy(self, folder_path, filename, allow_none=False):
        file_path = folder_path / filename
        if file_path.exists():
            return np.load(file_path, allow_pickle=True)
        if allow_none:
            return None
        raise FileNotFoundError(f"Missing {filename} in {folder_path}")

    def save_npy(self, folder_path, filename, array_value):
        np.save(folder_path / filename, array_value)

    def find_measurement_folders(self):
        return sorted([folder for folder in self.base_path.rglob("SCOS_Single_*") if folder.is_dir()])

    @staticmethod
    def apply_master_only_mask(mask, master_only=MASTER_ONLY):
        """Restrict a mask to the master-SPAD half of the sensor (left half of columns)."""
        if not master_only or mask is None:
            return mask
        master_mask = np.zeros_like(mask, dtype=bool)
        master_mask[:mask.shape[1] // 2, :] = True  # columns, not rows
        return mask & master_mask

    # ------------------------------------------------------------------ #
    # Folder / metadata matching helpers
    # ------------------------------------------------------------------ #
    def extract_power_level(self, text_value):
        match = re.search(r"(\d+(?:\.\d+)?%)", str(text_value))
        return match.group(1) if match else None

    def find_matching_tpsf(self, anchor_folder=None, power_level=None):
        if power_level is None and anchor_folder is not None:
            for candidate in [anchor_folder, *anchor_folder.parents]:
                power_level = self.extract_power_level(candidate.name)
                if power_level:
                    break

        if anchor_folder is not None:
            search_roots = [anchor_folder] + list(anchor_folder.parents)
            for root_folder in search_roots:
                if power_level:
                    candidates = list(root_folder.glob(f"TPSF_*{power_level}*"))
                else:
                    candidates = list(root_folder.glob("TPSF_*"))
                if candidates:
                    return candidates[0]

        if power_level:
            candidates = list(self.base_path.glob(f"TPSF_*{power_level}*"))
        else:
            candidates = list(self.base_path.glob("TPSF_*"))
        if len(candidates) == 1:
            return candidates[0]

        if power_level:
            candidates = list(self.base_path.rglob(f"TPSF_*{power_level}*"))
        else:
            candidates = list(self.base_path.rglob("TPSF_*"))
        if len(candidates) == 1:
            return candidates[0]

        return candidates[0] if candidates else None

    def find_matching_irf(self, anchor_folder):
        if anchor_folder is None:
            return None

        search_roots = [anchor_folder.parent, *anchor_folder.parents]
        for root_folder in search_roots:
            candidates = list(root_folder.glob("IRF_*"))
            if candidates:
                return candidates[0]

        return None

    def load_gate_offsets(self, folder_path):
        offsets_path = folder_path / "gate_offsets.npy"
        if offsets_path.exists():
            return np.asarray(np.load(offsets_path, allow_pickle=True)).flatten()
        return None

    def infer_time_gate_reference_ps(self):
        if self._time_gate_reference_ps is not None:
            return self._time_gate_reference_ps

        gate_1_values = []
        for folder_path in self.find_measurement_folders():
            folder_text = f"{folder_path.parent.name} {folder_path.name}".lower()
            # accept several common naming variants for the 'gate 1' folder
            # e.g. 'gate 1', 'gated 1', 'gate1', 'gated-1'
            if not re.search(r"\bgat(?:e|ed)?\s*-?\s*1\b", folder_text):
                continue
            metadata = self.load_metadata(folder_path)
            gate_position_ps = metadata.get("gate_position_ps")
            if gate_position_ps is not None:
                gate_1_values.append(float(gate_position_ps))

        if gate_1_values:
            self._time_gate_reference_ps = float(np.median(gate_1_values))
        else:
            self._time_gate_reference_ps = None

        return self._time_gate_reference_ps

    def get_peak_ps_from_metadata(self, folder_path):
        if not folder_path or not folder_path.exists():
            return None

        metadata = self.load_metadata(folder_path)
        if metadata.get("measurement_type") not in {"TPSF", "IRF"}:
            return None

        time_vec = self.load_npy(folder_path, "t_axis.npy", allow_none=True)
        decon_peak = metadata.get("decon_peak")

        if time_vec is not None and decon_peak is not None:
            time_vec = np.asarray(time_vec).flatten()
            if time_vec.size > 0:
                return float(time_vec[0] + float(decon_peak))

        deconvolved = self.load_npy(folder_path, "deconvolved.npy", allow_none=True)
        if time_vec is None or deconvolved is None:
            return None

        time_vec = np.asarray(time_vec).flatten()
        deconvolved = np.asarray(deconvolved).flatten()
        if time_vec.size == 0 or deconvolved.size == 0:
            return None

        peak_index = int(np.argmax(deconvolved))
        if peak_index >= time_vec.size:
            return None

        return float(time_vec[peak_index])

    def get_peak_components_from_arrays(self, folder_path):
        if not folder_path or not folder_path.exists():
            return None, None, None

        metadata = self.load_metadata(folder_path)
        time_vec = self.load_npy(folder_path, "t_axis.npy", allow_none=True)
        deconvolved = self.load_npy(folder_path, "deconvolved.npy", allow_none=True)
        decon_peak = metadata.get("decon_peak")

        if time_vec is not None and deconvolved is not None:
            time_vec = np.asarray(time_vec).flatten()
            deconvolved = np.asarray(deconvolved).flatten()
            if time_vec.size > 0 and deconvolved.size > 0:
                peak_index = int(np.argmax(deconvolved))
                if peak_index < time_vec.size:
                    start_time_ps = float(time_vec[0])
                    peak_ps = float(time_vec[peak_index])
                    decon_peak_ps = float(peak_ps - start_time_ps)
                    return decon_peak_ps, start_time_ps, peak_ps

        if time_vec is None or decon_peak is None:
            return None, None, None

        time_vec = np.asarray(time_vec).flatten()
        if time_vec.size == 0:
            return None, None, None

        start_time_ps = float(time_vec[0])
        decon_peak_ps = float(decon_peak)
        peak_ps = float(start_time_ps + decon_peak_ps)
        return decon_peak_ps, start_time_ps, peak_ps

    def get_metadata_decon_peak_ps(self, folder_path):
        if not folder_path or not folder_path.exists():
            return None

        metadata = self.load_metadata(folder_path)
        decon_peak = metadata.get("decon_peak")
        return float(decon_peak) if decon_peak is not None else None

    def get_peak_offset_ps_from_metadata(self, folder_path):
        if not folder_path or not folder_path.exists():
            return None

        metadata = self.load_metadata(folder_path)
        decon_peak = metadata.get("decon_peak")
        if decon_peak is not None:
            return float(decon_peak)

        return self.get_peak_ps_from_metadata(folder_path)

    def extract_folder_label(self, folder_path):
        parts = [folder_path.parent.name, folder_path.name]
        return " / ".join(part for part in parts if part)

    def extract_measurement_type(self, folder_path):
        text = f"{folder_path.parent.name} {folder_path.name}".lower()
        if "late" in text and "ref" in text:
            return "late_ref"
        if "gate" in text:
            return "gate"
        if "ref" in text:
            return "early_ref"
        return "unknown"

    def extract_method_label(self, folder_path):
        text = f"{folder_path.parent.name} {folder_path.name}".lower()
        if "late" in text and "ref" in text:
            return "PaLS-iSCOS"
        if "ref" in text:
            return "ISCOS"
        if "gate" in text:
            return "Gated"
        return "Unknown"

    def extract_measurement_index(self, folder_path):
        text = f"{folder_path.parent.name} {folder_path.name}".lower()
        match = re.search(r"\b(\d+)\b", text)
        return int(match.group(1)) if match else None

    # ------------------------------------------------------------------ #
    # Signal-processing helpers
    # ------------------------------------------------------------------ #
    def calculate_corrected_rbfi(self, corrected_contrast):
        corrected_contrast = np.asarray(corrected_contrast, dtype=float).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            rbfi = 1.0 / np.square(corrected_contrast)
        rbfi[~np.isfinite(rbfi)] = np.nan
        return rbfi

    def normalize_to_baseline(self, rbfi, baseline_mean):
        rbfi = np.asarray(rbfi, dtype=float).flatten()
        if not np.isfinite(baseline_mean) or baseline_mean == 0:
            return np.full_like(rbfi, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = rbfi / baseline_mean
        normalized[~np.isfinite(normalized)] = np.nan
        return normalized

    def summarize_window(self, values, time_vec, start_s, end_s):
        time_vec = np.asarray(time_vec, dtype=float).flatten()
        values = np.asarray(values, dtype=float).flatten()

        if time_vec.size == 0 or values.size == 0 or start_s is None or end_s is None:
            return {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
                "mask": np.zeros(time_vec.size, dtype=bool),
            }

        mask = (time_vec >= start_s) & (time_vec <= end_s)
        window_values = values[mask]
        window_values = window_values[np.isfinite(window_values)]

        return {
            "mean": float(np.nanmean(window_values)) if window_values.size else np.nan,
            "std": float(np.nanstd(window_values)) if window_values.size else np.nan,
            "count": int(window_values.size),
            "mask": mask,
        }

    def compute_stage_windows(self, time_vec):
        """
        Base and tourniquet windows are always computed. The 'late'
        (post-release) window is only computed for stages=3.

        stages=2: tourniquet = last `window_seconds` of the recording.
        stages=3: tourniquet = a middle window padded on both sides by
                  `window_cut_transition_seconds`; late = last
                  `window_seconds` of the recording.
        """
        time_vec = np.asarray(time_vec, dtype=float).flatten()
        start_time = float(time_vec[0])
        end_time = float(time_vec[-1])

        base_start = start_time
        base_end = min(start_time + self.window_seconds, end_time)

        if self.stages == 3:
            tourniquet_start = base_end + self.window_cut_transition_seconds
            tourniquet_end = min(
                tourniquet_start + self.window_seconds - self.window_cut_transition_seconds,
                end_time,
            )
            late_start = max(end_time - self.window_seconds + self.window_cut_transition_seconds, start_time)
            late_end = end_time
        else:
            tourniquet_start = max(end_time - self.window_seconds, start_time)
            tourniquet_end = end_time
            late_start = None
            late_end = None

        return {
            "base_start": base_start,
            "base_end": base_end,
            "tourniquet_start": tourniquet_start,
            "tourniquet_end": tourniquet_end,
            "late_start": late_start,
            "late_end": late_end,
        }

    # ------------------------------------------------------------------ #
    # Intensity-artifact calibration (only used if correct_intensity=True)
    # ------------------------------------------------------------------ #
    def calibrate_intensity_artifact(self):
        """
        Pass 1: learn the intensity ratio R = tourniquet_I / base_I from
        the Gated measurements, as a function of gate delay, so it can be
        interpolated and applied to ISCOS/PaLS-iSCOS measurements later
        (via their coherence gate) as well as to the Gated measurements
        themselves.
        """
        print("\n[Pass 1] Calibrating intensity artifact from Gated measurements...")
        time_gate_reference_ps = self.infer_time_gate_reference_ps()

        gated_delays = []
        gated_base_Is = []
        gated_tourniquet_Is = []

        for folder_path in self.find_measurement_folders():
            if self.extract_method_label(folder_path) != "Gated":
                continue

            metadata = self.load_metadata(folder_path)
            gate_position_ps = metadata.get("gate_position_ps")
            if gate_position_ps is None or time_gate_reference_ps is None:
                continue
            time_gate_ps = round(float(gate_position_ps) - time_gate_reference_ps, 4)

            time_vec = self.load_npy(folder_path, "time_vector.npy", allow_none=True)
            intensity = self.load_npy(folder_path, "intensity.npy", allow_none=True)
            if time_vec is None or intensity is None:
                continue
            time_vec = np.asarray(time_vec).flatten()
            intensity = np.asarray(intensity).flatten()
            if time_vec.size == 0 or intensity.size == 0:
                continue

            windows = self.compute_stage_windows(time_vec)
            base_mask = (time_vec >= windows["base_start"]) & (time_vec < windows["base_end"])
            tourniquet_mask = (time_vec >= windows["tourniquet_start"]) & (time_vec < windows["tourniquet_end"])

            base_I = np.nanmean(intensity[base_mask]) if np.any(base_mask) else np.nan
            tourniquet_I = np.nanmean(intensity[tourniquet_mask]) if np.any(tourniquet_mask) else np.nan
            if not np.isfinite(base_I) or base_I == 0 or not np.isfinite(tourniquet_I):
                continue

            gated_delays.append(time_gate_ps)
            gated_base_Is.append(base_I)
            gated_tourniquet_Is.append(tourniquet_I)

        if not gated_delays:
            print("    No usable Gated measurements found; intensity correction will fall back to 1.0 (no correction).")
            self.gated_baseline_model = None
            return

        from scipy.interpolate import interp1d

        delays = np.array(gated_delays, dtype=float)
        base_Is = np.array(gated_base_Is, dtype=float)
        tourniquet_Is = np.array(gated_tourniquet_Is, dtype=float)

        sorted_idx = np.argsort(delays)
        delays_sorted = delays[sorted_idx]
        base_Is_sorted = base_Is[sorted_idx]
        tourniquet_Is_sorted = tourniquet_Is[sorted_idx]

        unique_delays, unique_indices = np.unique(delays_sorted, return_index=True)
        unique_base_Is = base_Is_sorted[unique_indices]
        unique_tourniquet_Is = tourniquet_Is_sorted[unique_indices]
        ratio = unique_tourniquet_Is / unique_base_Is

        if len(unique_delays) > 1:
            self.gated_baseline_model = interp1d(unique_delays, ratio, kind="linear", fill_value="extrapolate")
        else:
            constant_ratio = float(ratio[0])
            self.gated_baseline_model = lambda x, _r=constant_ratio: _r

        # save (rather than plt.show, which would block a batch run) a calibration plot
        try:
            output_parent = Path(self.output_csv).parent if self.output_csv else Path(self.base_path)
            delay_range = np.linspace(min(unique_delays), max(unique_delays), 100)
            interpolated_ratio = self.gated_baseline_model(delay_range)
            plt.figure(figsize=(8, 5))
            plt.plot(unique_delays, ratio, "o", label="Measured tourniquet/baseline intensity")
            plt.plot(delay_range, interpolated_ratio, "-", label="Interpolated model")
            plt.xlabel("Gate Position (ps)")
            plt.ylabel("Intensity ratio (tourniquet / baseline)")
            plt.title("Intensity artifact calibration")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_parent / "intensity_calibration.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as error:
            print(f"    Warning: could not save calibration plot: {error}")

        print(f"    Intensity artifact calibration complete ({len(unique_delays)} gate position(s)).")

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot_results(self, time_vec, values, save_path, filename, title_suffix="", base_mask=None, tourniquet_mask=None, late_mask=None):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(time_vec, values, color="navy", linewidth=1.4, label="Signal")

            if base_mask is not None and np.any(base_mask):
                ax.axvspan(time_vec[base_mask][0], time_vec[base_mask][-1], color="seagreen", alpha=0.12, label="Base window")

            if tourniquet_mask is not None and np.any(tourniquet_mask):
                ax.axvspan(time_vec[tourniquet_mask][0], time_vec[tourniquet_mask][-1], color="darkorange", alpha=0.12, label="Tourniquet window")

            if late_mask is not None and np.any(late_mask):
                ax.axvspan(time_vec[late_mask][0], time_vec[late_mask][-1], color="lightblue", alpha=0.12, label="Late window")

            ax.set_title(f"{title_suffix}".strip())
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Signal (normalized)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            plt.tight_layout()
            output_path = Path(save_path) / filename
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as error:
            print(f"    Error saving plot: {error}")

    # ------------------------------------------------------------------ #
    # Per-measurement processing
    # ------------------------------------------------------------------ #
    def process_measurement(self, folder_path, force_recalc=False):
        folder_label = self.extract_folder_label(folder_path)
        measurement_type = self.extract_measurement_type(folder_path)
        method_label = self.extract_method_label(folder_path)
        measurement_index = self.extract_measurement_index(folder_path)
        print(f"\nProcessing sensitivity measurement: {folder_label}")

        try:
            metadata = self.load_metadata(folder_path)
            power_level = (
                self.extract_power_level(folder_label)
                or self.extract_power_level(folder_path.parent.name)
                or self.extract_power_level(folder_path.name)
            )

            tpsf_folder = self.find_matching_tpsf(folder_path.parent, power_level=power_level)
            tpsf_peak_ps = self.get_peak_ps_from_metadata(tpsf_folder)

            gate_position_ps = metadata.get("gate_position_ps")
            gate_position_ps_value = float(gate_position_ps) if gate_position_ps is not None else None
            gate_offsets = self.load_gate_offsets(folder_path)

            gate_value_ps = None
            if gate_offsets is not None and gate_offsets.size > 0:
                if measurement_index is not None and 0 <= measurement_index < gate_offsets.size:
                    gate_value_ps = float(gate_offsets[measurement_index])
                else:
                    gate_value_ps = float(gate_offsets[0])
            elif gate_position_ps_value is not None:
                gate_value_ps = gate_position_ps_value

            time_gate_reference_ps = self.infer_time_gate_reference_ps()
            time_gate_ps = None
            if gate_value_ps is not None and time_gate_reference_ps is not None:
                time_gate_ps = round(gate_value_ps - time_gate_reference_ps, 4)

            irf_folder = self.find_matching_irf(folder_path)
            irf_decon_peak_ps, irf_start_ps, irf_peak_ps = self.get_peak_components_from_arrays(irf_folder)
            tpsf_decon_peak_ps, tpsf_start_ps, tpsf_peak_ps = self.get_peak_components_from_arrays(tpsf_folder)
            irf_peak_offset_ps = self.get_metadata_decon_peak_ps(irf_folder)
            tpsf_peak_offset_ps = self.get_metadata_decon_peak_ps(tpsf_folder)
            coherence_gate_ps = None
            if method_label in {"ISCOS", "PaLS-iSCOS"} and irf_peak_ps is not None and tpsf_peak_ps is not None:
                coherence_gate_ps = round(irf_peak_ps - tpsf_peak_ps, 4)

            if force_recalc:
                image_data = self.load_npy(folder_path, "image_data.npy")
                background_img = self.load_npy(folder_path, "backgroundImg.npy", allow_none=True)
                dark_var_per_window = self.load_npy(folder_path, "darkVarPerWindow.npy", allow_none=True)

                # --- ROI fix (same as scos_analysis.py) ---
                # Recompute the mask from the actual image data via find_mask()
                # instead of trusting a possibly-stale roi_mask.npy on disk.
                # Computed once per run and reused for every subsequent folder,
                # since all measurements share the same physical camera FOV.
                if self.mask is None:
                    mean_image = np.mean(image_data, axis=2)
                    mask = find_mask(mean_image)
                    mask = self.apply_master_only_mask(mask, self.master_only)
                    self.mask = mask
                else:
                    mask = self.mask
                self.save_npy(folder_path, "roi_mask.npy", mask)
                print(
                    f"    Recomputed ROI mask: {mask.sum()} / {mask.size} pixels"
                    f"{' (MASTER_ONLY)' if self.master_only else ''}"
                )

                res = SCOS_Calculation(
                    image_data=image_data,
                    camera_gain=metadata.get("gain", 1.0),
                    mask=mask if mask is not None else np.ones(image_data.shape[:2], dtype=bool),
                    black_level=0,
                    frame_rate=metadata.get("frame_rate", 100.0),
                    backgroundImg=background_img if background_img is not None else 0,
                    darkVarPerWindow=dark_var_per_window if dark_var_per_window is not None else 0,
                    nBits=metadata.get("bit_depth", 8),
                    is_pileup=metadata.get("is_pileup_correction", False),
                )

                self.save_npy(folder_path, "time_vector.npy", res["time_vector"])
                self.save_npy(folder_path, "K2_corrected.npy", res["K2_corrected"])
                self.save_npy(folder_path, "K2_raw.npy", res.get("K2_raw", None))
                self.save_npy(folder_path, "BFi.npy", res.get("BFi", None))
                self.save_npy(folder_path, "intensity.npy", np.mean(image_data, axis=(0, 1)))

            time_vec = np.asarray(self.load_npy(folder_path, "time_vector.npy")).flatten()
            k2_corrected = np.asarray(self.load_npy(folder_path, "K2_corrected.npy")).flatten()
            intensity_raw = self.load_npy(folder_path, "intensity.npy", allow_none=True)
            intensity = np.asarray(intensity_raw).flatten() if intensity_raw is not None else None

            with np.errstate(invalid="ignore"):
                contrast = np.sqrt(np.maximum(k2_corrected.astype(float).flatten(), 0.0))

            if time_vec.size == 0 or contrast.size == 0:
                raise ValueError("Missing time vector or contrast data")

            windows = self.compute_stage_windows(time_vec)
            base_start, base_end = windows["base_start"], windows["base_end"]
            tourniquet_start, tourniquet_end = windows["tourniquet_start"], windows["tourniquet_end"]
            late_start, late_end = windows["late_start"], windows["late_end"]
            has_late_stage = self.stages == 3

            # --- optional intensity correction, applied only to the tourniquet window ---
            contrast_multiplier = 1.0
            rbfi_multiplier = 1.0
            R = 1.0
            if self.correct_intensity and intensity is not None:
                base_mask_R = (time_vec >= base_start) & (time_vec < base_end)
                tourniquet_mask_R = (time_vec >= tourniquet_start) & (time_vec < tourniquet_end)

                if method_label == "Gated":
                    # Gated SCOS uses the actual measured deviation
                    base_I = np.nanmean(intensity[base_mask_R]) if np.any(base_mask_R) else np.nan
                    tourniquet_I = np.nanmean(intensity[tourniquet_mask_R]) if np.any(tourniquet_mask_R) else np.nan
                    if np.isfinite(base_I) and base_I != 0 and np.isfinite(tourniquet_I):
                        R = tourniquet_I / base_I
                    # Homodyne physics:
                    contrast_multiplier = R
                    rbfi_multiplier = 1.0 / (R ** 2) if R != 0 else 1.0

                elif method_label in {"ISCOS", "PaLS-iSCOS"} and self.gated_baseline_model is not None:
                    # ISCOS/PaLS-iSCOS require the interpolated sample-intensity model
                    gate_to_query = coherence_gate_ps if coherence_gate_ps is not None else 0
                    try:
                        R = float(self.gated_baseline_model(gate_to_query))
                    except Exception:
                        R = 1.0  # fall back to no correction
                    # Heterodyne physics:
                    contrast_multiplier = np.sqrt(R) if R > 0 else 1.0
                    rbfi_multiplier = 1.0 / R if R > 0 else 1.0

                print(f"  Intensity correction ON — R={R:.6g}, contrast x{contrast_multiplier:.6g}, rBFI x{rbfi_multiplier:.6g}")

            # --- raw (uncorrected) window summaries, for reference/QA ---
            base_contrast_summary = self.summarize_window(contrast, time_vec, base_start, base_end)
            tourniquet_contrast_summary = self.summarize_window(contrast * contrast_multiplier, time_vec, tourniquet_start, tourniquet_end)
            late_contrast_summary = self.summarize_window(contrast, time_vec, late_start, late_end) if has_late_stage else None

            rbfi_raw = self.calculate_corrected_rbfi(contrast)
            base_rbfi_summary = self.summarize_window(rbfi_raw, time_vec, base_start, base_end)
            tourniquet_rbfi_summary = self.summarize_window(rbfi_raw * rbfi_multiplier, time_vec, tourniquet_start, tourniquet_end)
            late_rbfi_summary = self.summarize_window(rbfi_raw, time_vec, late_start, late_end) if has_late_stage else None

            if intensity is not None:
                base_intensity_summary = self.summarize_window(intensity, time_vec, base_start, base_end)
                tourniquet_intensity_summary = self.summarize_window(intensity, time_vec, tourniquet_start, tourniquet_end)
                late_intensity_summary = self.summarize_window(intensity, time_vec, late_start, late_end) if has_late_stage else None
            else:
                empty = {"mean": np.nan, "std": np.nan, "count": 0, "mask": np.zeros(0, dtype=bool)}
                base_intensity_summary = tourniquet_intensity_summary = empty
                late_intensity_summary = empty if has_late_stage else None

            # --- normalized metric (drives percentage_difference + plot) ---
            if self.metric == "rbfi":
                baseline_mean = base_rbfi_summary["mean"]
                if not np.isfinite(baseline_mean) or baseline_mean == 0:
                    raise ValueError("Invalid baseline mean rBFI for normalization")

                normalized = self.normalize_to_baseline(rbfi_raw, baseline_mean)
                base_summary = self.summarize_window(normalized, time_vec, base_start, base_end)
                tourniquet_summary = self.summarize_window(normalized * rbfi_multiplier, time_vec, tourniquet_start, tourniquet_end)
                late_summary = self.summarize_window(normalized, time_vec, late_start, late_end) if has_late_stage else None

                normalized_tourniquet_mean = tourniquet_summary["mean"]
                normalized_tourniquet_std = tourniquet_summary["std"]
                percentage_difference = float((normalized_tourniquet_mean - 1.0) * 100.0) if np.isfinite(normalized_tourniquet_mean) else np.nan

                title_suffix = f"Normalized rBFI ({folder_label})"
                plot_filename = f"{folder_path.name}_sensitivity_rbfi.png"
            else:
                baseline_mean = base_contrast_summary["mean"]
                if not np.isfinite(baseline_mean) or baseline_mean == 0:
                    raise ValueError("Invalid baseline mean contrast for normalization")

                with np.errstate(divide="ignore", invalid="ignore"):
                    normalized = contrast / float(baseline_mean)
                normalized[~np.isfinite(normalized)] = np.nan

                base_summary = self.summarize_window(normalized, time_vec, base_start, base_end)
                tourniquet_summary = self.summarize_window(normalized * contrast_multiplier, time_vec, tourniquet_start, tourniquet_end)
                late_summary = self.summarize_window(normalized, time_vec, late_start, late_end) if has_late_stage else None

                normalized_tourniquet_mean = tourniquet_summary["mean"]
                normalized_tourniquet_std = tourniquet_summary["std"]
                percentage_difference = float((normalized_tourniquet_mean - 1.0) * 100.0) if np.isfinite(normalized_tourniquet_mean) else np.nan

                title_suffix = f"Normalized contrast ({folder_label})"
                plot_filename = f"{folder_path.name}_sensitivity_contrast.png"

            self.plot_results(
                time_vec,
                normalized,
                folder_path,
                plot_filename,
                title_suffix=title_suffix,
                base_mask=base_summary["mask"],
                tourniquet_mask=tourniquet_summary["mask"],
                late_mask=late_summary["mask"] if has_late_stage else None,
            )

            row = {
                "folder": folder_label,
                "folder_path": str(folder_path),
                "method": method_label,
                "measurement_type": measurement_type,
                "measurement_index": measurement_index,
                "stages": self.stages,
                "correct_intensity": self.correct_intensity,
                "gate_position_ps": gate_position_ps_value if gate_position_ps_value is not None else np.nan,
                "tpsf_peak_ps": tpsf_peak_ps,
                "irf_peak_ps": irf_peak_ps,
                "tpsf_peak_offset_ps": tpsf_peak_offset_ps,
                "irf_peak_offset_ps": irf_peak_offset_ps,
                "time_gate_reference_ps": time_gate_reference_ps,
                "time_gate_ps": time_gate_ps if time_gate_ps is not None else np.nan,
                "coherence_gate_ps": coherence_gate_ps if coherence_gate_ps is not None else np.nan,
                "window_seconds": self.window_seconds,
                "window_cut_transition_seconds": self.window_cut_transition_seconds if has_late_stage else np.nan,
                "base_start_s": base_start,
                "base_end_s": base_end,
                "tourniquet_start_s": tourniquet_start,
                "tourniquet_end_s": tourniquet_end,
                "late_start_s": late_start if late_start is not None else np.nan,
                "late_end_s": late_end if late_end is not None else np.nan,
                "baseline_mean_rBFI": base_summary["mean"],
                "baseline_std_rBFI": base_summary["std"],
                "base_count": base_summary["count"],
                "tourniquet_mean_rBFI": normalized_tourniquet_mean,
                "tourniquet_std_rBFI": normalized_tourniquet_std,
                "tourniquet_count": tourniquet_summary["count"],
                "late_mean_rBFI": late_summary["mean"] if has_late_stage else np.nan,
                "late_std_rBFI": late_summary["std"] if has_late_stage else np.nan,
                "late_count": late_summary["count"] if has_late_stage else np.nan,
                "percentage_rBFI_difference": percentage_difference,
                "raw_baseline_mean_contrast": base_contrast_summary["mean"],
                "raw_baseline_std_contrast": base_contrast_summary["std"],
                "raw_tourniquet_mean_contrast": tourniquet_contrast_summary["mean"],
                "raw_tourniquet_std_contrast": tourniquet_contrast_summary["std"],
                "raw_late_mean_contrast": late_contrast_summary["mean"] if has_late_stage else np.nan,
                "raw_late_std_contrast": late_contrast_summary["std"] if has_late_stage else np.nan,
                "raw_baseline_mean_intensity": base_intensity_summary["mean"],
                "raw_baseline_std_intensity": base_intensity_summary["std"],
                "raw_tourniquet_mean_intensity": tourniquet_intensity_summary["mean"],
                "raw_tourniquet_std_intensity": tourniquet_intensity_summary["std"],
                "raw_late_mean_intensity": late_intensity_summary["mean"] if has_late_stage else np.nan,
                "raw_late_std_intensity": late_intensity_summary["std"] if has_late_stage else np.nan,
                "raw_baseline_mean_rBFI": base_rbfi_summary["mean"],
                "raw_baseline_std_rBFI": base_rbfi_summary["std"],
                "raw_tourniquet_mean_rBFI": tourniquet_rbfi_summary["mean"],
                "raw_tourniquet_std_rBFI": tourniquet_rbfi_summary["std"],
                "raw_late_mean_rBFI": late_rbfi_summary["mean"] if has_late_stage else np.nan,
                "raw_late_std_rBFI": late_rbfi_summary["std"] if has_late_stage else np.nan,
                "intensity_ratio_R": R,
                "correction_factor_rBFI": rbfi_multiplier,
                "correction_factor_contrast": contrast_multiplier,
                "metric": self.metric,
                "total_frames": int(time_vec.size),
                "baseline_mean_normalized": base_summary["mean"],
                "baseline_std_normalized": base_summary["std"],
                "tourniquet_mean_normalized": normalized_tourniquet_mean,
                "tourniquet_std_normalized": normalized_tourniquet_std,
            }
            self.results.append(row)

            print(f"  Raw baseline contrast mean/std: {base_contrast_summary['mean']:.6g} / {base_contrast_summary['std']:.6g}")
            print(f"  Normalized baseline mean/std: {base_summary['mean']:.6g} / {base_summary['std']:.6g}")
            print(f"  Normalized tourniquet mean/std: {tourniquet_summary['mean']:.6g} / {tourniquet_summary['std']:.6g}")
            print(f"  Percentage difference (tourniquet vs baseline): {percentage_difference:.4f}%")
            if has_late_stage and late_summary is not None and np.isfinite(late_summary["mean"]):
                print(f"  Normalized late (post-release) mean/std: {late_summary['mean']:.6g} / {late_summary['std']:.6g}")
            if time_gate_ps is not None:
                print(f"  Time gate difference: {time_gate_ps:.4f} ps")
            if coherence_gate_ps is not None:
                print(f"  Coherence gate: {coherence_gate_ps:.4f} ps")
            print(f"✓ Processed {folder_label}")

        except Exception as error:
            print(f"✗ Error: {error}")

    # ------------------------------------------------------------------ #
    # Top-level run + aggregation
    # ------------------------------------------------------------------ #
    def run(self):
        print("\n" + "=" * 60)
        print("SENSITIVITY TEST ANALYSIS")
        print("=" * 60)
        print(f"Stages: {self.stages}   Intensity correction: {self.correct_intensity}")

        measurement_folders = self.find_measurement_folders()
        if not measurement_folders:
            print("No SCOS_Single folders found")
            return

        if self.correct_intensity:
            # calibrate from Gated measurements before processing any folders,
            # so the model can be applied to ISCOS/PaLS-iSCOS measurements too
            self.calibrate_intensity_artifact()

        for folder in measurement_folders:
            self.process_measurement(folder, force_recalc=self.recalc_SCOS)

        if self.results:
            df = pd.DataFrame(self.results)
            preferred_columns = [
                "folder",
                "method",
                "measurement_type",
                "measurement_index",
                "stages",
                "correct_intensity",
                "time_gate_ps",
                "coherence_gate_ps",
                "gate_position_ps",
                "tpsf_peak_ps",
                "irf_peak_ps",
                "tpsf_peak_offset_ps",
                "irf_peak_offset_ps",
                "time_gate_reference_ps",
                "percentage_rBFI_difference",
                "baseline_mean_rBFI",
                "baseline_std_rBFI",
                "tourniquet_mean_rBFI",
                "tourniquet_std_rBFI",
                "late_mean_rBFI",
                "late_std_rBFI",
                "raw_baseline_mean_rBFI",
                "raw_baseline_std_rBFI",
                "raw_tourniquet_mean_rBFI",
                "raw_tourniquet_std_rBFI",
                "raw_late_mean_rBFI",
                "raw_late_std_rBFI",
                "intensity_ratio_R",
                "correction_factor_rBFI",
                "correction_factor_contrast",
                "window_seconds",
                "window_cut_transition_seconds",
                "base_start_s",
                "base_end_s",
                "tourniquet_start_s",
                "tourniquet_end_s",
                "late_start_s",
                "late_end_s",
                "base_count",
                "tourniquet_count",
                "late_count",
                "total_frames",
                "baseline_mean_normalized",
                "baseline_std_normalized",
                "tourniquet_mean_normalized",
                "tourniquet_std_normalized",
            ]
            df = df[[column for column in preferred_columns if column in df.columns]]
            df.to_csv(self.output_csv, index=False)
            print(f"\n✓ Summary saved to {self.output_csv}")
            try:
                self.aggregate_and_plot(df)
            except Exception as err:
                print(f"Warning: could not create aggregated summary/plots: {err}")
        else:
            print("\nNo results generated")

    def aggregate_and_plot(self, df):
        # Create offset column: Gated uses time_gate_ps, others use coherence_gate_ps
        df_work = df.copy()
        df_work["offset_ps"] = np.where(
            df_work.get("method", "") == "Gated",
            df_work.get("time_gate_ps"),
            df_work.get("coherence_gate_ps"),
        )
        df_work["offset_ps"] = pd.to_numeric(df_work["offset_ps"], errors="coerce")

        # If Gated rows lack `time_gate_ps`, reconstruct from `gate_position_ps`
        try:
            t_ref = self.infer_time_gate_reference_ps()
            if t_ref is not None:
                mask_fill = (df_work.get("method", "") == "Gated") & df_work["offset_ps"].isna() & df_work["gate_position_ps"].notna()
                if mask_fill.any():
                    df_work.loc[mask_fill, "offset_ps"] = pd.to_numeric(df_work.loc[mask_fill, "gate_position_ps"], errors="coerce") - float(t_ref)
        except Exception:
            pass

        output_parent = Path(self.output_csv).parent if self.output_csv else Path(self.base_path)
        summary_folder = output_parent / "sensitivity_test_results_plots"
        summary_folder.mkdir(parents=True, exist_ok=True)

        group = df_work.dropna(subset=["offset_ps", "percentage_rBFI_difference"])
        if group.empty:
            print("No offset data available for aggregated summary")
            return

        sample_buckets = {}
        for _, row in group.iterrows():
            method = row.get("method")
            offset = float(row.get("offset_ps"))
            key = (method, offset)
            sample_buckets.setdefault(key, []).append({
                "pct_mean": row.get("percentage_rBFI_difference"),
                "baseline_mean": row.get("baseline_mean_normalized"),
                "baseline_std": row.get("baseline_std_normalized"),
                "tourniquet_mean": row.get("tourniquet_mean_normalized"),
                "tourniquet_std": row.get("tourniquet_std_normalized"),
            })

        rows = []
        for (method, offset), folder_list in sample_buckets.items():
            folder_pct_means = np.array([f["pct_mean"] for f in folder_list if np.isfinite(f["pct_mean"])])
            if folder_pct_means.size == 0:
                continue

            baseline_means = np.array([f["baseline_mean"] for f in folder_list if np.isfinite(f["baseline_mean"])])
            baseline_stds = np.array([f["baseline_std"] for f in folder_list if np.isfinite(f["baseline_std"])])
            tourniquet_means = np.array([f["tourniquet_mean"] for f in folder_list if np.isfinite(f["tourniquet_mean"])])
            tourniquet_stds = np.array([f["tourniquet_std"] for f in folder_list if np.isfinite(f["tourniquet_std"])])

            rows.append({
                "method": method,
                "offset_ps": offset,
                "percentage_mean": float(np.nanmean(folder_pct_means)),
                "percentage_std": float(np.nanstd(folder_pct_means)),
                "percentage_n": int(folder_pct_means.size),
                "baseline_mean_rBFI": float(np.nanmean(baseline_means)) if baseline_means.size else np.nan,
                "baseline_std_rBFI": float(np.nanmean(baseline_stds)) if baseline_stds.size else np.nan,
                "tourniquet_mean_rBFI": float(np.nanmean(tourniquet_means)) if tourniquet_means.size else np.nan,
                "tourniquet_std_rBFI": float(np.nanmean(tourniquet_stds)) if tourniquet_stds.size else np.nan,
            })

        if not rows:
            print("No aggregated samples found")
            return

        agg = pd.DataFrame(rows)
        summary_csv = output_parent / "sensitivity_test_results_summary.csv"
        agg.to_csv(summary_csv, index=False)
        print(f"✓ Aggregated summary saved to {summary_csv}")

        y_label = "Percentage rBFI difference (%)" if self.metric == "rbfi" else "Percentage contrast difference (%)"

        # Per-method subplot
        methods = agg["method"].unique().tolist()
        n = max(1, len(methods))
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        for i, method in enumerate(methods):
            ax = axs[0, i]
            rows_m = agg[agg["method"] == method].sort_values("offset_ps")
            if rows_m.empty:
                continue
            x = rows_m["offset_ps"].astype(float).values
            y = rows_m["percentage_mean"].values
            yerr = rows_m["percentage_std"].fillna(0).values
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
            ax.set_title(method)
            ax.set_xlabel("Offset (ps)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out1 = summary_folder / "sensitivity_summary_by_method.png"
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved per-method sensitivity plot to {out1}")

        # Combined overlay plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for method in methods:
            rows_m = agg[agg["method"] == method].sort_values("offset_ps")
            if rows_m.empty:
                continue
            x = rows_m["offset_ps"].astype(float).values
            y = rows_m["percentage_mean"].values
            yerr = rows_m["percentage_std"].fillna(0).values
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method)
        ax.set_xlabel("Offset (ps)")
        ax.set_ylabel(y_label)
        ax.set_title("Sensitivity summary (all methods)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        out2 = summary_folder / "sensitivity_summary_by_method_combined.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved combined sensitivity plot to {out2}")


if __name__ == "__main__":
    import sys

    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    recalc_scos = "--recalc" in sys.argv

    window_seconds = 30.0
    if "--window" in sys.argv:
        window_index = sys.argv.index("--window")
        if window_index + 1 < len(sys.argv):
            window_seconds = float(sys.argv[window_index + 1])

    window_cut_transition_seconds = 5.0
    if "--transition" in sys.argv:
        transition_index = sys.argv.index("--transition")
        if transition_index + 1 < len(sys.argv):
            window_cut_transition_seconds = float(sys.argv[transition_index + 1])

    # metric selection: 'contrast' (default) or 'rbfi'
    metric_arg = "contrast"
    if "--metric" in sys.argv:
        mi = sys.argv.index("--metric")
        if mi + 1 < len(sys.argv):
            metric_arg = sys.argv[mi + 1].lower()

    # stages: 2 (default) or 3
    stages_arg = 2
    if "--stages" in sys.argv:
        si = sys.argv.index("--stages")
        if si + 1 < len(sys.argv):
            stages_arg = int(sys.argv[si + 1])

    # intensity correction: off by default, opt in with --correct-intensity
    correct_intensity_arg = "--correct-intensity" in sys.argv

    output_csv = str(Path(base_path) / "sensitivity_test_results.csv")
    
    print(f"Running sensitivity test analysis on base path: {base_path}")
    print(f"  Recalculate SCOS: {recalc_scos}")
    print(f"  Window seconds: {window_seconds}")
    print(f"  Window cut transition seconds: {window_cut_transition_seconds}")
    print(f"  Metric: {metric_arg}")
    print(f"  Stages: {stages_arg}")
    print(f"  Correct intensity: {correct_intensity_arg}")
    
    analyzer = SensitivityTestAnalyzer(
        base_path,
        output_csv=output_csv,
        recalc_SCOS=recalc_scos,
        window_seconds=window_seconds,
        window_cut_transition_seconds=window_cut_transition_seconds,
        metric=metric_arg,
        stages=stages_arg,
        correct_intensity=correct_intensity_arg,
    )
    analyzer.run()