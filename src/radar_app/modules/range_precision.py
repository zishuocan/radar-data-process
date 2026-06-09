from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi, sqrt

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import make_window, next_fft_size, positive_range_fft
from radar_app.modules.range_hrrp import select_frame_data


HRRP_DB_FLOOR = -80.0


@dataclass(frozen=True)
class BandwidthCase:
    key: str
    label: str
    keep_ratio: float
    window: str


@dataclass
class ComplexHrrpResult:
    ranges_m: np.ndarray
    power: np.ndarray
    magnitude_norm: np.ndarray
    hrrp_db: np.ndarray
    n_keep: int
    nfft: int
    bandwidth_hz: float
    range_resolution_m: float


@dataclass
class PrecisionResult:
    case: BandwidthCase
    target_range_m: float
    peak_index: int
    peak_power: float
    noise_power: float
    snr: float
    snr_db: float
    precision_m: float
    target_left_m: float
    target_right_m: float
    noise_bin_count: int
    hrrp: ComplexHrrpResult
    noise_mask: np.ndarray


def _chirp_label(chirp: str) -> str:
    text = chirp.strip().lower()
    if text in {"", "平均", "avg", "average", "mean", "all"}:
        return "平均"
    return str(int(float(chirp)))


def compute_complex_hrrp(
    frame_data: np.ndarray,
    params: RadarParams,
    keep_ratio: float,
    window_name: str,
    chirp_choice: str,
    zero_padding_power: int,
) -> ComplexHrrpResult:
    label = _chirp_label(chirp_choice)
    if label == "平均":
        selected = frame_data
    else:
        chirp_index = max(0, min(frame_data.shape[0] - 1, int(label) - 1))
        selected = frame_data[[chirp_index], :]

    selected = selected.astype(np.complex128 if np.iscomplexobj(selected) else np.float64)
    n_keep = max(1, int(round(params.samples_per_chirp * keep_ratio)))
    n_keep = min(params.samples_per_chirp, n_keep)
    selected = selected[:, :n_keep]
    selected = selected - np.mean(selected, axis=1, keepdims=True)
    selected = selected * make_window(window_name, n_keep).reshape(1, -1)

    nfft = next_fft_size(n_keep, zero_padding_power)
    ranges_m, spectrum = positive_range_fft(selected, params, axis=1, nfft=nfft)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    magnitude = np.sqrt(power)
    magnitude_norm = magnitude / magnitude.max() if magnitude.max(initial=0.0) > 0 else magnitude
    hrrp_db = 20.0 * np.log10(np.maximum(magnitude_norm, 10.0 ** (HRRP_DB_FLOOR / 20.0)))
    bandwidth_hz = params.slope_hz_s * n_keep / params.sample_rate_hz
    range_resolution_m = C0 / (2.0 * bandwidth_hz)
    return ComplexHrrpResult(
        ranges_m=ranges_m,
        power=power,
        magnitude_norm=magnitude_norm,
        hrrp_db=hrrp_db,
        n_keep=n_keep,
        nfft=nfft,
        bandwidth_hz=bandwidth_hz,
        range_resolution_m=range_resolution_m,
    )


def parabolic_peak_range(ranges_m: np.ndarray, values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(ranges_m[index])
    y0, y1, y2 = values[index - 1], values[index], values[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(ranges_m[index])
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(ranges_m[index] + delta * (ranges_m[1] - ranges_m[0]))


def find_target_peak(
    ranges_m: np.ndarray,
    magnitude_norm: np.ndarray,
    min_range_m: float,
    max_range_m: float,
) -> int:
    valid = np.where((ranges_m >= min_range_m) & (ranges_m <= max_range_m))[0]
    valid = valid[(valid > 0) & (valid < len(magnitude_norm) - 1)]
    if valid.size == 0:
        raise ValueError("目标检测范围内没有可用距离单元")

    valid_max = float(np.max(magnitude_norm[valid]))
    threshold = max(0.08 * valid_max, float(np.median(magnitude_norm[valid]) * 2.0))
    local_peaks = valid[
        (magnitude_norm[valid] >= magnitude_norm[valid - 1])
        & (magnitude_norm[valid] >= magnitude_norm[valid + 1])
        & (magnitude_norm[valid] >= threshold)
    ]
    if local_peaks.size == 0:
        local_peaks = valid
    return int(local_peaks[np.argmax(magnitude_norm[local_peaks])])


def find_3db_indices(power: np.ndarray, peak_index: int) -> tuple[int, int]:
    peak_power = max(float(power[peak_index]), 1e-300)
    threshold_db = 10.0 * np.log10(peak_power) - 3.0
    power_db = 10.0 * np.log10(np.maximum(power, 1e-300))

    left = peak_index
    while left > 0 and power_db[left] > threshold_db:
        left -= 1
    right = peak_index
    while right < len(power_db) - 1 and power_db[right] > threshold_db:
        right += 1
    return left, right


def estimate_precision(
    case: BandwidthCase,
    hrrp: ComplexHrrpResult,
    min_range_m: float,
    noise_max_range_m: float,
    min_guard_bins: int,
) -> PrecisionResult:
    peak_index = find_target_peak(hrrp.ranges_m, hrrp.magnitude_norm, min_range_m, noise_max_range_m)
    target_range_m = parabolic_peak_range(hrrp.ranges_m, hrrp.magnitude_norm, peak_index)

    left_3db, right_3db = find_3db_indices(hrrp.power, peak_index)
    bin_width_m = float(hrrp.ranges_m[1] - hrrp.ranges_m[0])
    guard_bins = max(min_guard_bins, int(ceil(hrrp.range_resolution_m / bin_width_m)))
    exclude_left = max(0, left_3db - guard_bins)
    exclude_right = min(len(hrrp.power) - 1, right_3db + guard_bins)

    noise_mask = (hrrp.ranges_m >= min_range_m) & (hrrp.ranges_m <= noise_max_range_m)
    noise_mask[exclude_left : exclude_right + 1] = False
    noise_bin_count = int(np.count_nonzero(noise_mask))
    if noise_bin_count <= 0:
        raise ValueError("排除目标区域后没有剩余噪声单元")

    peak_power = float(hrrp.power[peak_index])
    noise_power = float(np.mean(hrrp.power[noise_mask]))
    if noise_power <= 0:
        raise ValueError("噪声功率估计为 0")

    snr = peak_power / noise_power
    snr_db = 10.0 * np.log10(snr)
    precision_m = sqrt(3.0) / (pi * sqrt(2.0 * snr)) * hrrp.range_resolution_m

    return PrecisionResult(
        case=case,
        target_range_m=target_range_m,
        peak_index=peak_index,
        peak_power=peak_power,
        noise_power=noise_power,
        snr=snr,
        snr_db=snr_db,
        precision_m=precision_m,
        target_left_m=float(hrrp.ranges_m[exclude_left]),
        target_right_m=float(hrrp.ranges_m[exclude_right]),
        noise_bin_count=noise_bin_count,
        hrrp=hrrp,
        noise_mask=noise_mask,
    )


def analyze_precision(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int,
    antenna_number: int,
    chirp_choice: str,
    full_window: str,
    half_window: str,
    zero_padding_power: int,
    min_range_m: float,
    noise_max_range_m: float | None,
    min_guard_bins: int,
) -> list[PrecisionResult]:
    frame_data = select_frame_data(data, frame_number, antenna_number)
    cases = [
        BandwidthCase("full", "全带宽", 1.0, full_window),
        BandwidthCase("half", "半带宽", 0.5, half_window),
    ]
    results: list[PrecisionResult] = []
    for case in cases:
        hrrp = compute_complex_hrrp(
            frame_data,
            params,
            case.keep_ratio,
            case.window,
            chirp_choice,
            zero_padding_power,
        )
        results.append(
            estimate_precision(
                case,
                hrrp,
                min_range_m=min_range_m,
                noise_max_range_m=noise_max_range_m or params.max_range_m,
                min_guard_bins=min_guard_bins,
            )
        )
    return results
