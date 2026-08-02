"""
Sensitivity test analysis for SCOS_Single measurements.

This script compares the first 30 s baseline against the last 30 s
tourniquet segment using corrected rBFI, defined as 1 / K^2_corrected.
It records the time gate relative to the experiment's gate-1 reference
and the coherence gate relative to the matched IRF peak for ref/late rows.
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scos_calculation import SCOS_Calculation
except ImportError:
    def SCOS_Calculation(*args, **kwargs):
        print("Warning: scos_calculation module not found.")
        return {}


class SensitivityTestAnalyzer:
    def __init__(self, base_path, output_csv="sensitivity_test_results.csv", recalc_SCOS=False, window_seconds=30.0, metric="contrast"):
        self.base_path = Path(base_path)
        self.output_csv = output_csv
        self.recalc_SCOS = recalc_SCOS
        self.window_seconds = float(window_seconds)
        self.results = []
        self._time_gate_reference_ps = None
        self.metric = str(metric).lower() if metric is not None else "contrast"

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

    def summarize_window(self, rbfi, time_vec, start_s, end_s):
        time_vec = np.asarray(time_vec, dtype=float).flatten()
        rbfi = np.asarray(rbfi, dtype=float).flatten()

        if time_vec.size == 0 or rbfi.size == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
                "mask": np.zeros(0, dtype=bool),
            }

        mask = (time_vec >= start_s) & (time_vec <= end_s)
        window_values = rbfi[mask]
        window_values = window_values[np.isfinite(window_values)]

        return {
            "mean": float(np.nanmean(window_values)) if window_values.size else np.nan,
            "std": float(np.nanstd(window_values)) if window_values.size else np.nan,
            "count": int(window_values.size),
            "mask": mask,
        }

    def plot_results(self, time_vec, rbfi, save_path, filename, title_suffix="", base_mask=None, late_mask=None):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(time_vec, rbfi, color="navy", linewidth=1.4, label="Signal")

            if base_mask is not None and np.any(base_mask):
                ax.axvspan(time_vec[base_mask][0], time_vec[base_mask][-1], color="seagreen", alpha=0.12, label="Base window")

            if late_mask is not None and np.any(late_mask):
                ax.axvspan(time_vec[late_mask][0], time_vec[late_mask][-1], color="darkorange", alpha=0.12, label="Late window")

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

    def process_measurement(self, folder_path, force_recalc=False):
        folder_label = self.extract_folder_label(folder_path)
        measurement_type = self.extract_measurement_type(folder_path)
        method_label = self.extract_method_label(folder_path)
        measurement_index = self.extract_measurement_index(folder_path)
        print(f"\nProcessing sensitivity measurement: {folder_label}")

        try:
            metadata = self.load_metadata(folder_path)
            power_level = self.extract_power_level(folder_label) or self.extract_power_level(folder_path.parent.name) or self.extract_power_level(folder_path.name)

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
            irf_metadata_peak_ps = self.get_metadata_decon_peak_ps(irf_folder)
            tpsf_metadata_peak_ps = self.get_metadata_decon_peak_ps(tpsf_folder)
            coherence_gate_ps = None
            if method_label in {"ISCOS", "PaLS-iSCOS"} and irf_peak_ps is not None and tpsf_peak_ps is not None:
                coherence_gate_ps = round(irf_peak_ps - tpsf_peak_ps, 4)

            if force_recalc:
                image_data = self.load_npy(folder_path, "image_data.npy")
                mask = self.load_npy(folder_path, "roi_mask.npy", allow_none=True)
                background_img = self.load_npy(folder_path, "backgroundImg.npy", allow_none=True)
                dark_var_per_window = self.load_npy(folder_path, "darkVarPerWindow.npy", allow_none=True)

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
            # contrast is sqrt(K2)
            with np.errstate(invalid='ignore'):
                contrast = np.sqrt(np.maximum(k2_corrected.astype(float).flatten(), 0.0))

            if time_vec.size == 0 or contrast.size == 0:
                raise ValueError("Missing time vector or contrast data")

            start_time = float(time_vec[0])
            end_time = float(time_vec[-1])
            base_start = start_time
            base_end = min(start_time + self.window_seconds, end_time)
            late_end = end_time
            late_start = max(end_time - self.window_seconds, start_time)

            # compute raw summaries for both contrast and rBFI (rbfi = 1/contrast^2)
            base_contrast_summary = self.summarize_window(contrast, time_vec, base_start, base_end)
            late_contrast_summary = self.summarize_window(contrast, time_vec, late_start, late_end)

            rbfi_raw = self.calculate_corrected_rbfi(contrast)
            base_rbfi_summary = self.summarize_window(rbfi_raw, time_vec, base_start, base_end)
            late_rbfi_summary = self.summarize_window(rbfi_raw, time_vec, late_start, late_end)

            # choose metric for normalization and plotting
            if self.metric == "rbfi":
                baseline_mean = base_rbfi_summary["mean"]
                if not np.isfinite(baseline_mean) or baseline_mean == 0:
                    raise ValueError("Invalid baseline mean rBFI for normalization")

                normalized = self.normalize_to_baseline(rbfi_raw, baseline_mean)
                base_summary = self.summarize_window(normalized, time_vec, base_start, base_end)
                late_summary = self.summarize_window(normalized, time_vec, late_start, late_end)
                normalized_late_mean = late_summary["mean"]
                normalized_late_std = late_summary["std"]
                percentage_rbfi_difference = float((normalized_late_mean - 1.0) * 100.0) if np.isfinite(normalized_late_mean) else np.nan

                title_suffix = f"Normalized rBFI ({folder_label})"
                plot_filename = f"{folder_path.name}_sensitivity_rbfi.png"
                self.plot_results(
                    time_vec,
                    normalized,
                    folder_path,
                    plot_filename,
                    title_suffix=title_suffix,
                    base_mask=base_summary["mask"],
                    late_mask=late_summary["mask"],
                )
            else:
                # default: contrast-based sensitivity
                baseline_mean = base_contrast_summary["mean"]
                if not np.isfinite(baseline_mean) or baseline_mean == 0:
                    raise ValueError("Invalid baseline mean contrast for normalization")

                # normalize contrast by baseline mean
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = contrast / float(baseline_mean)
                normalized[~np.isfinite(normalized)] = np.nan

                base_summary = self.summarize_window(normalized, time_vec, base_start, base_end)
                late_summary = self.summarize_window(normalized, time_vec, late_start, late_end)

                normalized_late_mean = late_summary["mean"]
                normalized_late_std = late_summary["std"]
                percentage_rbfi_difference = float((normalized_late_mean - 1.0) * 100.0) if np.isfinite(normalized_late_mean) else np.nan

                title_suffix = f"Normalized contrast ({folder_label})"
                plot_filename = f"{folder_path.name}_sensitivity_contrast.png"
                self.plot_results(
                    time_vec,
                    normalized,
                    folder_path,
                    plot_filename,
                    title_suffix=title_suffix,
                    base_mask=base_summary["mask"],
                    late_mask=late_summary["mask"],
                )

            self.results.append({
                "folder": folder_label,
                "folder_path": str(folder_path),
                "method": method_label,
                "measurement_type": measurement_type,
                "measurement_index": measurement_index,
                "gate_position_ps": gate_position_ps_value if gate_position_ps_value is not None else np.nan,
                "tpsf_peak_ps": tpsf_peak_ps,
                "irf_peak_ps": irf_peak_ps,
                "tpsf_peak_offset_ps": tpsf_peak_offset_ps,
                "irf_peak_offset_ps": irf_peak_offset_ps,
                "time_gate_reference_ps": time_gate_reference_ps,
                "time_gate_ps": time_gate_ps if time_gate_ps is not None else np.nan,
                "coherence_gate_ps": coherence_gate_ps if coherence_gate_ps is not None else np.nan,
                "window_seconds": self.window_seconds,
                "base_start_s": base_start,
                "base_end_s": base_end,
                "late_start_s": late_start,
                "late_end_s": late_end,
                "baseline_mean_rBFI": base_summary["mean"],
                "baseline_std_rBFI": base_summary["std"],
                "base_count": base_summary["count"],
                "tourniquet_mean_rBFI": normalized_late_mean,
                "tourniquet_std_rBFI": normalized_late_std,
                "late_count": late_summary["count"],
                "percentage_rBFI_difference": percentage_rbfi_difference,
                "raw_baseline_mean_contrast": base_contrast_summary["mean"],
                "raw_baseline_std_contrast": base_contrast_summary["std"],
                "raw_tourniquet_mean_contrast": late_contrast_summary["mean"],
                "raw_tourniquet_std_contrast": late_contrast_summary["std"],
                "raw_baseline_mean_rBFI": base_rbfi_summary["mean"],
                "raw_baseline_std_rBFI": base_rbfi_summary["std"],
                "raw_tourniquet_mean_rBFI": late_rbfi_summary["mean"],
                "raw_tourniquet_std_rBFI": late_rbfi_summary["std"],
                "metric": self.metric,
                "total_frames": int(time_vec.size),
            })

            print(f"  Raw baseline contrast mean/std: {base_contrast_summary['mean']:.6g} / {base_contrast_summary['std']:.6g}")
            print(f"  Normalized baseline contrast mean/std: {base_summary['mean']:.6g} / {base_summary['std']:.6g}")
            print(f"  Normalized tourniquet contrast mean/std: {late_summary['mean']:.6g} / {late_summary['std']:.6g}")
            print(f"  Percentage contrast difference: {percentage_rbfi_difference:.4f}%")
            if time_gate_ps is not None:
                print(f"  Time gate difference: {time_gate_ps:.4f} ps")
            # if time_gate_reference_ps is not None:
            #     print(f"  Time gate reference: {time_gate_reference_ps:.4f} ps")
            # if irf_peak_ps is not None and irf_decon_peak_ps is not None and irf_start_ps is not None:
            #     print(f"  IRF decon peak + start: {irf_decon_peak_ps:.4f} + {irf_start_ps:.4f} = {irf_peak_ps:.4f} ps")
            # if tpsf_peak_ps is not None and tpsf_decon_peak_ps is not None and tpsf_start_ps is not None:
            #     print(f"  TPSF decon peak + start: {tpsf_decon_peak_ps:.4f} + {tpsf_start_ps:.4f} = {tpsf_peak_ps:.4f} ps")
            # if irf_metadata_peak_ps is not None and irf_decon_peak_ps is not None:
            #     irf_peak_diff = abs(irf_metadata_peak_ps - irf_decon_peak_ps)
            #     if irf_peak_diff > 1.0:
            #         print(f"  IRF metadata decon peak differs by {irf_peak_diff:.4f} ps")
            # if tpsf_metadata_peak_ps is not None and tpsf_decon_peak_ps is not None:
            #     tpsf_peak_diff = abs(tpsf_metadata_peak_ps - tpsf_decon_peak_ps)
            #     if tpsf_peak_diff > 1.0:
            #         print(f"  TPSF metadata decon peak differs by {tpsf_peak_diff:.4f} ps")
            if coherence_gate_ps is not None:
                print(f"  Coherence gate: {coherence_gate_ps:.4f} ps")
            print(f"✓ Processed {folder_label}")

        except Exception as error:
            print(f"✗ Error: {error}")

    def run(self):
        print("\n" + "=" * 60)
        print("SENSITIVITY TEST ANALYSIS")
        print("=" * 60)

        measurement_folders = self.find_measurement_folders()
        if not measurement_folders:
            print("No SCOS_Single folders found")
            return

        for folder in measurement_folders:
            self.process_measurement(folder, force_recalc=self.recalc_SCOS)

        if self.results:
            df = pd.DataFrame(self.results)
            preferred_columns = [
                "folder",
                "method",
                "measurement_type",
                "measurement_index",
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
                "raw_baseline_mean_rBFI",
                "raw_baseline_std_rBFI",
                "raw_tourniquet_mean_rBFI",
                "raw_tourniquet_std_rBFI",
                "window_seconds",
                "base_start_s",
                "base_end_s",
                "late_start_s",
                "late_end_s",
                "base_count",
                "late_count",
                "total_frames",
            ]
            df = df[[column for column in preferred_columns if column in df.columns]]
            df.to_csv(self.output_csv, index=False)
            print(f"\n✓ Summary saved to {self.output_csv}")
            try:
                # compute per-method offset and aggregated statistics and plots
                self.aggregate_and_plot(df)
            except Exception as err:
                print(f"Warning: could not create aggregated summary/plots: {err}")
        else:
            print("\nNo results generated")

    def aggregate_and_plot(self, df):
        # Create offset column: Gated uses time_gate_ps, others use coherence_gate_ps
        df_work = df.copy()
        df_work['offset_ps'] = np.where(df_work.get('method', '') == 'Gated', df_work.get('time_gate_ps'), df_work.get('coherence_gate_ps'))
        # ensure numeric offsets
        df_work['offset_ps'] = pd.to_numeric(df_work['offset_ps'], errors='coerce')

        # If Gated rows lack `time_gate_ps`, try to reconstruct from `gate_position_ps`
        # using the analyzer's inferred gate-1 reference time (if available).
        try:
            t_ref = self.infer_time_gate_reference_ps()
            if t_ref is not None:
                mask_fill = (df_work.get('method', '') == 'Gated') & df_work['offset_ps'].isna() & df_work['gate_position_ps'].notna()
                if mask_fill.any():
                    df_work.loc[mask_fill, 'offset_ps'] = pd.to_numeric(df_work.loc[mask_fill, 'gate_position_ps'], errors='coerce') - float(t_ref)
        except Exception:
            pass

        # Ensure we have a folder_path column; if missing, try to reconstruct from the `folder` text by searching base_path
        if 'folder_path' not in df_work.columns:
            df_work['folder_path'] = None
            for idx, row in df_work.iterrows():
                folder_text = row.get('folder', '')
                parts = [p.strip() for p in str(folder_text).split('/') if p.strip()]
                if not parts:
                    continue
                # expect something like 'parent / foldername'
                if len(parts) >= 2:
                    parent_name = parts[-2]
                    folder_name = parts[-1]
                else:
                    parent_name = None
                    folder_name = parts[-1]

                try:
                    candidates = list(self.base_path.rglob(folder_name))
                except Exception:
                    candidates = []

                chosen = None
                if candidates:
                    if parent_name:
                        for c in candidates:
                            if c.parent.name == parent_name:
                                chosen = c
                                break
                    if chosen is None:
                        chosen = candidates[0]

                if chosen is not None:
                    df_work.at[idx, 'folder_path'] = str(chosen)

        # place summary files next to the main output CSV so they are easy to find
        output_parent = Path(self.output_csv).parent if self.output_csv else Path(self.base_path)
        summary_folder = output_parent / "sensitivity_test_results_plots"
        summary_folder.mkdir(parents=True, exist_ok=True)

        # For each (method, offset) collect baseline and late samples across all measurements
        group = df_work.dropna(subset=['offset_ps', 'folder_path'])
        if group.empty:
            print("No offset data available for aggregated summary")
            return

        sample_buckets = {}
        for _, row in group.iterrows():
            method = row.get('method')
            offset = float(row.get('offset_ps'))
            folder_path = row.get('folder_path')
            base_start = row.get('base_start_s')
            base_end = row.get('base_end_s')
            late_start = row.get('late_start_s')
            late_end = row.get('late_end_s')
            # try to load time series
            try:
                tp = Path(folder_path)
                time_vec = np.asarray(self.load_npy(tp, 'time_vector.npy')).flatten()
                k2_corrected = np.asarray(self.load_npy(tp, 'K2_corrected.npy')).flatten()
            except Exception:
                continue

            # get raw baseline mean for normalization (from saved row or recompute)
            if self.metric == 'rbfi':
                baseline_mean = row.get('raw_baseline_mean_rBFI')
            else:
                baseline_mean = row.get('raw_baseline_mean_contrast')

            # fallback: compute from data if missing
            base_mask = (time_vec >= base_start) & (time_vec <= base_end)
            if baseline_mean is None or not np.isfinite(baseline_mean) or baseline_mean == 0:
                base_vals = k2_corrected[base_mask]
                base_vals = base_vals[np.isfinite(base_vals)]
                if base_vals.size:
                    if self.metric == 'rbfi':
                        # compute rbfi from k2: rbfi = 1 / (sqrt(K2))^2 = 1 / K2
                        # but ensure non-negative and avoid divide-by-zero
                        k2_vals = np.maximum(base_vals.astype(float), 0.0)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            rbfi_vals = 1.0 / k2_vals
                        rbfi_vals = rbfi_vals[np.isfinite(rbfi_vals)]
                        baseline_mean = float(np.nanmean(rbfi_vals)) if rbfi_vals.size else np.nan
                    else:
                        # contrast fallback
                        contrast_vals = np.sqrt(np.maximum(base_vals.astype(float), 0.0))
                        baseline_mean = float(np.nanmean(contrast_vals)) if contrast_vals.size else np.nan

            if not np.isfinite(baseline_mean) or baseline_mean == 0:
                continue

            # compute normalized series depending on metric
            contrast = np.sqrt(np.maximum(k2_corrected.astype(float), 0.0))
            if self.metric == 'rbfi':
                rbfi_raw = self.calculate_corrected_rbfi(contrast)
                normalized = self.normalize_to_baseline(rbfi_raw, baseline_mean)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = contrast / float(baseline_mean)
                normalized[~np.isfinite(normalized)] = np.nan

            # collect normalized samples
            base_mask = (time_vec >= base_start) & (time_vec <= base_end)
            late_mask = (time_vec >= late_start) & (time_vec <= late_end)
            base_samples = normalized[base_mask]
            late_samples = normalized[late_mask]
            base_samples = base_samples[np.isfinite(base_samples)]
            late_samples = late_samples[np.isfinite(late_samples)]

            # compute folder-level statistics (treat this folder as one sample)
            if base_samples.size == 0 or late_samples.size == 0:
                continue
            folder_baseline_mean = float(np.nanmean(base_samples))
            folder_baseline_std = float(np.nanstd(base_samples))
            folder_late_mean = float(np.nanmean(late_samples))
            folder_late_std = float(np.nanstd(late_samples))
            if not np.isfinite(folder_baseline_mean):
                continue
            # percentage change for this folder (mean of per-frame percent changes on normalized contrast)
            folder_pct_samples = (late_samples - 1.0) * 100.0
            folder_pct_mean = float(np.nanmean(folder_pct_samples)) if folder_pct_samples.size else np.nan

            key = (method, offset)
            sample_buckets.setdefault(key, []).append({
                'pct_mean': folder_pct_mean,
                'baseline_mean': folder_baseline_mean,
                'baseline_std': folder_baseline_std,
                'late_mean': folder_late_mean,
                'late_std': folder_late_std,
                'n_late': int(np.sum(np.isfinite(late_samples)))
            })

        # compute aggregated statistics per key
        rows = []
        for (method, offset), folder_list in sample_buckets.items():
            if not folder_list:
                continue
            # treat each folder as one sample: use folder-level pct_mean values
            folder_pct_means = np.array([f['pct_mean'] for f in folder_list if np.isfinite(f['pct_mean'])])
            if folder_pct_means.size == 0:
                continue
            pct_mean = float(np.nanmean(folder_pct_means))
            pct_std = float(np.nanstd(folder_pct_means))
            pct_n = int(folder_pct_means.size)

            baseline_means = np.array([f['baseline_mean'] for f in folder_list if np.isfinite(f['baseline_mean'])])
            baseline_stds = np.array([f['baseline_std'] for f in folder_list if np.isfinite(f['baseline_std'])])
            late_means = np.array([f['late_mean'] for f in folder_list if np.isfinite(f['late_mean'])])
            late_stds = np.array([f['late_std'] for f in folder_list if np.isfinite(f['late_std'])])

            rows.append({
                'method': method,
                'offset_ps': offset,
                'percentage_mean': pct_mean,
                'percentage_std': pct_std,
                'percentage_n': pct_n,
                'baseline_mean_rBFI': float(np.nanmean(baseline_means)) if baseline_means.size else np.nan,
                'baseline_std_rBFI': float(np.nanmean(baseline_stds)) if baseline_stds.size else np.nan,
                'tourniquet_mean_rBFI': float(np.nanmean(late_means)) if late_means.size else np.nan,
                'tourniquet_std_rBFI': float(np.nanmean(late_stds)) if late_stds.size else np.nan,
            })

        if not rows:
            print('No aggregated samples found')
            return

        agg = pd.DataFrame(rows)

        summary_csv = output_parent / "sensitivity_test_results_summary.csv"
        agg.to_csv(summary_csv, index=False)
        print(f"✓ Aggregated summary saved to {summary_csv}")

        # Per-method subplot
        methods = agg['method'].unique().tolist()
        n = max(1, len(methods))
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        for i, method in enumerate(methods):
            ax = axs[0, i]
            rows = agg[agg['method'] == method].sort_values('offset_ps')
            if rows.empty:
                continue
            x = rows['offset_ps'].astype(float).values
            y = rows['percentage_mean'].values
            yerr = rows['percentage_std'].fillna(0).values
            ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3)
            ax.set_title(method)
            ax.set_xlabel('Offset (ps)')
            ax.set_ylabel('Percentage rBFI difference (%)')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out1 = summary_folder / 'sensitivity_summary_by_method.png'
        plt.savefig(out1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-method sensitivity plot to {out1}")

        # Combined overlay plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for method in methods:
            rows = agg[agg['method'] == method].sort_values('offset_ps')
            if rows.empty:
                continue
            x = rows['offset_ps'].astype(float).values
            y = rows['percentage_mean'].values
            yerr = rows['percentage_std'].fillna(0).values
            ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3, label=method)
        ax.set_xlabel('Offset (ps)')
        ax.set_ylabel('Percentage rBFI difference (%)')
        ax.set_title('Sensitivity summary (all methods)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        out2 = summary_folder / 'sensitivity_summary_by_method_combined.png'
        plt.savefig(out2, dpi=150, bbox_inches='tight')
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

    # metric selection: 'contrast' (default) or 'rbfi'
    metric_arg = 'contrast'
    if '--metric' in sys.argv:
        mi = sys.argv.index('--metric')
        if mi + 1 < len(sys.argv):
            metric_arg = sys.argv[mi + 1].lower()

    output_csv = str(Path(base_path) / "sensitivity_test_results.csv")
    analyzer = SensitivityTestAnalyzer(
        base_path,
        output_csv=output_csv,
        recalc_SCOS=recalc_scos,
        window_seconds=window_seconds,
        metric=metric_arg,
    )
    analyzer.run()

    