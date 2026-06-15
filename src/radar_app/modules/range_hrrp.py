from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import make_window, next_fft_size, normalized_db_from_amplitude, positive_range_fft


HRRP_DB_FLOOR = -60.0


@dataclass
class MusicRangeResult:
    ranges_m: np.ndarray
    spectrum_norm: np.ndarray
    spectrum_db: np.ndarray
    source_count: int
    matrix_order: int
    snapshot_count: int
    targets: list[tuple[float, float, int]]


@dataclass
class HrrpAnalysis:
    ranges_m: np.ndarray
    magnitude_norm: np.ndarray
    hrrp_db: np.ndarray
    n_keep: int
    nfft: int
    targets: list[tuple[float, float, int]]
    widths: list[tuple[float, float, float, float]]
    music: MusicRangeResult | None = None


def select_frame_data(data: np.ndarray, frame_number: int, antenna_number: int) -> np.ndarray:
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = data[frame_index]
    if frame_data.ndim == 3:
        antenna_index = max(0, min(frame_data.shape[0] - 1, antenna_number - 1))
        return frame_data[antenna_index]
    return frame_data


def _chirp_index(chirp_choice: str, chirp_count: int) -> int | None:
    text = chirp_choice.strip().lower()
    if text in {"", "平均", "avg", "average", "mean", "all"}:
        return None
    return max(0, min(chirp_count - 1, int(float(text)) - 1))


def compute_hrrp(
    frame_data: np.ndarray,
    params: RadarParams,
    window_name: str,
    chirp_choice: str,
    keep_ratio: float = 1.0,
    zero_padding_power: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    chirp_index = _chirp_index(chirp_choice, frame_data.shape[0])
    selected = frame_data if chirp_index is None else frame_data[[chirp_index], :]
    selected = selected.astype(np.complex128 if np.iscomplexobj(selected) else np.float64)

    n_keep = max(1, int(round(params.samples_per_chirp * keep_ratio)))
    n_keep = min(params.samples_per_chirp, n_keep)
    selected = selected[:, :n_keep]
    selected = selected - np.mean(selected, axis=1, keepdims=True)
    selected = selected * make_window(window_name, n_keep).reshape(1, -1)

    nfft = next_fft_size(n_keep, zero_padding_power)
    ranges_m, spectrum = positive_range_fft(selected, params, axis=1, nfft=nfft)
    magnitude = np.mean(np.abs(spectrum), axis=0)
    if magnitude.max(initial=0.0) > 0:
        magnitude = magnitude / magnitude.max()
    return ranges_m, magnitude, n_keep, nfft


def _analytic_signal(samples: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(samples):
        return samples.astype(np.complex128, copy=False)

    values = samples.astype(np.float64, copy=False)
    sample_count = values.shape[-1]
    spectrum = np.fft.fft(values, axis=-1)
    multiplier = np.zeros(sample_count, dtype=np.float64)
    multiplier[0] = 1.0
    if sample_count % 2 == 0:
        multiplier[1 : sample_count // 2] = 2.0
        multiplier[sample_count // 2] = 1.0
    else:
        multiplier[1 : (sample_count + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier.reshape((1,) * (values.ndim - 1) + (-1,)), axis=-1)


def _music_matrix_order(sample_count: int, requested_order: int | None, source_count: int) -> int:
    if requested_order is not None and requested_order > 0:
        order = int(requested_order)
    else:
        order = min(64, max(8, sample_count // 2))
    order = max(source_count + 1, order)
    return max(2, min(sample_count - 1, order))


def compute_music_range_spectrum(
    frame_data: np.ndarray,
    params: RadarParams,
    chirp_choice: str,
    keep_ratio: float,
    ranges_m: np.ndarray,
    source_count: int = 1,
    matrix_order: int | None = None,
    target_count: int = 1,
    min_range_m: float = 0.15,
    min_gap_m: float = 0.15,
) -> MusicRangeResult:
    chirp_index = _chirp_index(chirp_choice, frame_data.shape[0])
    selected = frame_data if chirp_index is None else frame_data[[chirp_index], :]

    n_keep = max(1, int(round(params.samples_per_chirp * keep_ratio)))
    n_keep = min(params.samples_per_chirp, n_keep)
    selected = _analytic_signal(selected[:, :n_keep])
    selected = selected - np.mean(selected, axis=1, keepdims=True)

    if n_keep < 8:
        raise ValueError("MUSIC 至少需要 8 个采样点")

    requested_sources = max(1, int(source_count))
    order = _music_matrix_order(n_keep, matrix_order, requested_sources)
    actual_sources = max(1, min(requested_sources, order - 1))
    snapshot_count = 0
    covariance = np.zeros((order, order), dtype=np.complex128)

    for chirp_samples in selected:
        snapshots = np.lib.stride_tricks.sliding_window_view(chirp_samples, order).T
        if snapshots.shape[1] == 0:
            continue
        covariance += snapshots @ snapshots.conj().T
        snapshot_count += snapshots.shape[1]

    if snapshot_count <= actual_sources:
        raise ValueError("MUSIC 快拍数不足，请降低源数或增加采样点")

    covariance /= snapshot_count
    exchange = np.fliplr(np.eye(order, dtype=np.complex128))
    covariance = 0.5 * (covariance + exchange @ covariance.conj() @ exchange)
    covariance = 0.5 * (covariance + covariance.conj().T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MUSIC 协方差矩阵分解失败") from exc

    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("MUSIC 协方差矩阵包含无效数值")

    order_indices = np.argsort(eigenvalues)
    noise_subspace = eigenvectors[:, order_indices[: order - actual_sources]]
    sample_indices = np.arange(order, dtype=np.float64).reshape(-1, 1)
    beat_freqs_hz = 2.0 * params.slope_hz_s * ranges_m / C0
    steering = np.exp(1j * 2.0 * np.pi * sample_indices * beat_freqs_hz.reshape(1, -1) / params.sample_rate_hz)
    projection = noise_subspace.conj().T @ steering
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    spectrum = 1.0 / np.maximum(denominator, 1e-300)
    if spectrum.max(initial=0.0) > 0:
        spectrum = spectrum / spectrum.max()
    spectrum_db = 10.0 * np.log10(np.maximum(spectrum, 10.0 ** (HRRP_DB_FLOOR / 10.0)))
    targets = detect_targets(
        ranges_m,
        spectrum,
        target_count=max(1, int(target_count)),
        min_range_m=min_range_m,
        min_gap_m=min_gap_m,
    )
    return MusicRangeResult(
        ranges_m=ranges_m,
        spectrum_norm=spectrum,
        spectrum_db=spectrum_db,
        source_count=actual_sources,
        matrix_order=order,
        snapshot_count=int(snapshot_count),
        targets=targets,
    )


def _parabolic_peak(ranges_m: np.ndarray, values: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(values) - 1:
        return float(ranges_m[index]), float(values[index])
    y0, y1, y2 = values[index - 1], values[index], values[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(ranges_m[index]), float(y1)
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    step = ranges_m[1] - ranges_m[0]
    peak_range = ranges_m[index] + delta * step
    peak_value = y1 - 0.25 * (y0 - y2) * delta
    return float(peak_range), float(max(peak_value, y1))


def detect_targets(
    ranges_m: np.ndarray,
    magnitude: np.ndarray,
    target_count: int = 2,
    min_range_m: float = 0.15,
    min_gap_m: float = 0.15,
) -> list[tuple[float, float, int]]:
    if ranges_m.size < 3 or magnitude.size < 3:
        return []

    valid = np.where(ranges_m >= min_range_m)[0]
    valid = valid[(valid > 0) & (valid < len(magnitude) - 1)]
    if valid.size == 0:
        return []

    valid_max = float(np.max(magnitude[valid]))
    if valid_max <= 0:
        return []

    threshold = max(0.08 * valid_max, float(np.median(magnitude[valid]) * 2.0))
    local_peaks = [
        int(idx)
        for idx in valid
        if magnitude[idx] >= magnitude[idx - 1]
        and magnitude[idx] >= magnitude[idx + 1]
        and magnitude[idx] >= threshold
    ]
    if not local_peaks:
        local_peaks = [int(idx) for idx in valid]

    ranked = sorted(local_peaks, key=lambda item: magnitude[item], reverse=True)
    selected: list[tuple[float, float, int]] = []
    for idx in ranked:
        peak_range, peak_value = _parabolic_peak(ranges_m, magnitude, idx)
        if all(abs(peak_range - previous[0]) >= min_gap_m for previous in selected):
            selected.append((peak_range, peak_value, idx))
        if len(selected) >= target_count:
            break
    return sorted(selected, key=lambda item: item[0])


def _interpolate_crossing(x1: float, y1: float, x2: float, y2: float, level: float) -> float:
    if np.isclose(y1, y2):
        return float(x1)
    return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1))


def measure_3db_width(
    ranges_m: np.ndarray,
    power_db: np.ndarray,
    peak_index: int,
) -> tuple[float, float, float, float]:
    threshold_db = float(power_db[peak_index] - 3.0)

    left_idx = peak_index
    while left_idx > 0 and power_db[left_idx] > threshold_db:
        left_idx -= 1
    if left_idx == 0 and power_db[left_idx] > threshold_db:
        return np.nan, np.nan, np.nan, threshold_db
    left_3db_m = _interpolate_crossing(
        float(ranges_m[left_idx]),
        float(power_db[left_idx]),
        float(ranges_m[left_idx + 1]),
        float(power_db[left_idx + 1]),
        threshold_db,
    )

    right_idx = peak_index
    while right_idx < len(power_db) - 1 and power_db[right_idx] > threshold_db:
        right_idx += 1
    if right_idx == len(power_db) - 1 and power_db[right_idx] > threshold_db:
        return np.nan, np.nan, np.nan, threshold_db
    right_3db_m = _interpolate_crossing(
        float(ranges_m[right_idx - 1]),
        float(power_db[right_idx - 1]),
        float(ranges_m[right_idx]),
        float(power_db[right_idx]),
        threshold_db,
    )
    return left_3db_m, right_3db_m, right_3db_m - left_3db_m, threshold_db


def analyze_hrrp(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int,
    antenna_number: int,
    window_name: str,
    chirp_choice: str,
    keep_ratio: float,
    target_count: int,
    min_range_m: float,
    min_gap_m: float,
    zero_padding_power: int = 5,
    use_music: bool = False,
    music_source_count: int = 1,
    music_matrix_order: int | None = None,
) -> HrrpAnalysis:
    frame_data = select_frame_data(data, frame_number, antenna_number)
    ranges_m, magnitude, n_keep, nfft = compute_hrrp(
        frame_data,
        params,
        window_name,
        chirp_choice,
        keep_ratio=keep_ratio,
        zero_padding_power=zero_padding_power,
    )
    hrrp_db = normalized_db_from_amplitude(magnitude, HRRP_DB_FLOOR)
    targets = detect_targets(ranges_m, magnitude, target_count, min_range_m, min_gap_m)
    widths = [measure_3db_width(ranges_m, hrrp_db, index) for _range, _value, index in targets]
    music = None
    if use_music:
        music = compute_music_range_spectrum(
            frame_data,
            params,
            chirp_choice,
            keep_ratio,
            ranges_m,
            source_count=music_source_count,
            matrix_order=music_matrix_order,
            target_count=target_count,
            min_range_m=min_range_m,
            min_gap_m=min_gap_m,
        )
    return HrrpAnalysis(
        ranges_m=ranges_m,
        magnitude_norm=magnitude,
        hrrp_db=hrrp_db,
        n_keep=n_keep,
        nfft=nfft,
        targets=targets,
        widths=widths,
        music=music,
    )
