from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import EPS, make_window, next_fft_size


REQUIRED_VIRTUAL_CHANNELS = 8
HRRP_DB_FLOOR = -60.0
RANGE_ANGLE_DB_FLOOR = -55.0
ANGLE_DB_FLOOR = -50.0


@dataclass(frozen=True)
class ChannelPhaseRecord:
    serial: int
    channel: int
    amplitude: float
    phase_rad: float
    phase_deg: float
    unwrapped_phase_rad: float
    unwrapped_phase_deg: float
    delta_from_previous_rad: float | None
    delta_from_previous_deg: float | None


@dataclass(frozen=True)
class CfarDetection:
    serial: int
    range_m: float
    angle_deg: float
    relative_power_db: float
    range_index: int
    angle_index: int


@dataclass
class CalibrationResult:
    target_range_m: float
    target_index: int
    virtual_channels: tuple[int, ...]
    channel_phases_rad: np.ndarray
    channel_phases_deg: np.ndarray
    phase_correction: np.ndarray


@dataclass
class SingleTargetResult:
    frame_number: int
    chirp_label: str
    virtual_channels: tuple[int, ...]
    params: RadarParams
    antenna_spacing_m: float
    nfft_range: int
    nfft_angle: int
    range_resolution_m: float
    max_display_range_m: float
    time_us: np.ndarray
    time_signal: np.ndarray
    ranges_m: np.ndarray
    hrrp_db: np.ndarray
    hrrp_power: np.ndarray
    target_index: int
    target_range_m: float
    channel_records: list[ChannelPhaseRecord]
    mean_delta_rad: float
    mean_delta_deg: float
    phase_angle_deg: float
    phase_sin_argument: float
    angle_axis_deg: np.ndarray
    angle_spectrum_db: np.ndarray
    angle_fft_angle_deg: float


@dataclass
class MultiTargetResult:
    frame_number: int
    chirp_label: str
    virtual_channels: tuple[int, ...]
    params: RadarParams
    antenna_spacing_m: float
    nfft_range: int
    nfft_angle: int
    max_display_range_m: float
    time_us: np.ndarray
    time_signal: np.ndarray
    ranges_m: np.ndarray
    angle_axis_deg: np.ndarray
    range_angle_complex: np.ndarray
    range_angle_power: np.ndarray
    range_angle_db: np.ndarray
    detections: list[CfarDetection]


def parse_virtual_channels(
    text: str,
    antenna_count: int,
    required_count: int = REQUIRED_VIRTUAL_CHANNELS,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if not text.strip():
        raise ValueError("请输入 8 个虚拟通道号，例如 1,2,3,4,5,6,7,8")
    tokens = [item for item in re.split(r"[,，;\s]+", text.strip()) if item]
    if len(tokens) != required_count:
        raise ValueError(f"虚拟通道必须恰好输入 {required_count} 个，当前为 {len(tokens)} 个")

    channels: list[int] = []
    for token in tokens:
        value = float(token)
        if not value.is_integer():
            raise ValueError(f"虚拟通道号必须是整数: {token}")
        channel = int(value)
        if channel < 1 or channel > antenna_count:
            raise ValueError(f"虚拟通道 {channel} 超出范围，当前数据共有 {antenna_count} 个通道")
        channels.append(channel)
    if len(set(channels)) != len(channels):
        raise ValueError("虚拟通道不能重复")
    return np.array([channel - 1 for channel in channels], dtype=int), tuple(channels)


def parse_chirp_indices(choice: str, chirps_per_frame: int) -> tuple[np.ndarray, str]:
    normalized = choice.strip().lower()
    if normalized in {"", "平均", "avg", "average", "mean", "all"}:
        return np.arange(chirps_per_frame, dtype=int), "平均"
    value = float(normalized)
    if not value.is_integer():
        raise ValueError(f"Chirp 必须是整数或平均: {choice}")
    chirp = int(value)
    if chirp < 1 or chirp > chirps_per_frame:
        raise ValueError(f"Chirp {chirp} 超出范围，当前每帧共有 {chirps_per_frame} 个 chirp")
    return np.array([chirp - 1], dtype=int), str(chirp)


def ensure_antenna_cube(frame_data: np.ndarray) -> np.ndarray:
    if frame_data.ndim != 3:
        raise ValueError("测角需要多通道数据，当前帧不是 antenna/chirp/sample 结构")
    return frame_data


def compute_range_spectrum(
    virtual_cube: np.ndarray,
    params: RadarParams,
    window_name: str,
    chirp_choice: str,
    zero_padding_power: int,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    if virtual_cube.ndim != 3:
        raise ValueError("虚拟通道数据维度应为 channel/chirp/sample")
    if virtual_cube.shape[0] != REQUIRED_VIRTUAL_CHANNELS:
        raise ValueError(f"方位测角需要 {REQUIRED_VIRTUAL_CHANNELS} 个虚拟通道")

    chirp_indices, chirp_label = parse_chirp_indices(chirp_choice, virtual_cube.shape[1])
    selected = virtual_cube[:, chirp_indices, :]
    selected = selected.astype(np.complex128 if np.iscomplexobj(selected) else np.float64, copy=False)
    selected = selected - np.mean(selected, axis=-1, keepdims=True)
    selected = selected * make_window(window_name, params.samples_per_chirp).reshape(1, 1, -1)

    nfft = next_fft_size(params.samples_per_chirp, zero_padding_power)
    if np.iscomplexobj(selected):
        spectrum = np.fft.fft(selected, n=nfft, axis=-1)
        freqs = np.fft.fftfreq(nfft, d=1.0 / params.sample_rate_hz)
        positive = freqs >= 0
        spectrum = spectrum[:, :, positive]
        freqs = freqs[positive]
    else:
        spectrum = np.fft.rfft(selected, n=nfft, axis=-1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / params.sample_rate_hz)

    range_spectrum = np.mean(spectrum, axis=1)
    ranges_m = C0 * freqs / (2.0 * params.slope_hz_s)
    return ranges_m, range_spectrum, nfft, chirp_label


def compute_time_signal(virtual_cube: np.ndarray, chirp_choice: str) -> tuple[np.ndarray, str]:
    chirp_indices, chirp_label = parse_chirp_indices(chirp_choice, virtual_cube.shape[1])
    signal = np.mean(virtual_cube[0, chirp_indices, :], axis=0)
    return signal, chirp_label


def hrrp_from_range_spectrum(range_spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = np.mean(np.abs(range_spectrum) ** 2, axis=0)
    amplitude = np.sqrt(power)
    if amplitude.max(initial=0.0) > 0:
        amplitude = amplitude / amplitude.max()
    hrrp_db = 20.0 * np.log10(np.maximum(amplitude, 10.0 ** (HRRP_DB_FLOOR / 20.0)))
    return power, hrrp_db


def find_strongest_range_peak(
    ranges_m: np.ndarray,
    power: np.ndarray,
    min_range_m: float,
    max_range_m: float,
) -> int:
    valid = np.where((ranges_m >= min_range_m) & (ranges_m <= max_range_m))[0]
    valid = valid[(valid > 0) & (valid < len(power) - 1)]
    if valid.size == 0:
        raise ValueError("没有可用于目标检测的距离单元")
    valid_power = power[valid]
    valid_max = float(np.max(valid_power))
    if valid_max <= 0:
        raise ValueError("距离像功率全为 0，无法检测目标")
    threshold = max(0.05 * valid_max, float(np.median(valid_power) * 3.0))
    local_peaks = valid[
        (power[valid] >= power[valid - 1])
        & (power[valid] >= power[valid + 1])
        & (power[valid] >= threshold)
    ]
    if local_peaks.size == 0:
        local_peaks = valid
    return int(local_peaks[np.argmax(power[local_peaks])])


def wavelength_m(params: RadarParams) -> float:
    return C0 / (params.start_freq_ghz * 1e9)


def angle_axis_and_mask(nfft_angle: int, params: RadarParams, antenna_spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    spatial_freqs = np.fft.fftshift(np.fft.fftfreq(nfft_angle, d=antenna_spacing_m))
    sin_theta = wavelength_m(params) * spatial_freqs
    valid = np.abs(sin_theta) <= 1.0
    angles = np.degrees(np.arcsin(np.clip(sin_theta[valid], -1.0, 1.0)))
    return angles, valid


def compute_angle_spectrum(
    channel_vector: np.ndarray,
    params: RadarParams,
    antenna_spacing_m: float,
    angle_fft_size: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    nfft = max(int(angle_fft_size), channel_vector.size)
    channel_window = np.hamming(channel_vector.size)
    spectrum = np.fft.fftshift(np.fft.fft(channel_vector * channel_window, n=nfft))
    angle_axis_deg, valid_angle = angle_axis_and_mask(nfft, params, antenna_spacing_m)
    power = np.abs(spectrum[valid_angle]) ** 2
    if power.size == 0:
        raise ValueError("当前阵元间距和载频下没有物理有效的方位角单元")
    power_max = float(np.max(power))
    spectrum_db = 10.0 * np.log10(np.maximum(power / max(power_max, EPS), 10.0 ** (ANGLE_DB_FLOOR / 10.0)))
    peak_index = int(np.argmax(power))
    return angle_axis_deg, spectrum_db, float(angle_axis_deg[peak_index]), nfft


def compute_channel_phase_records(
    channel_vector: np.ndarray,
    virtual_channels: tuple[int, ...],
) -> tuple[list[ChannelPhaseRecord], float, float]:
    amplitudes = np.abs(channel_vector)
    phases = np.angle(channel_vector)
    unwrapped = np.unwrap(phases)
    deltas = np.diff(unwrapped)
    mean_delta_rad = float(np.mean(deltas)) if deltas.size else float("nan")
    mean_delta_deg = float(np.degrees(mean_delta_rad)) if np.isfinite(mean_delta_rad) else float("nan")

    records: list[ChannelPhaseRecord] = []
    for idx, channel in enumerate(virtual_channels):
        delta_rad = None if idx == 0 else float(deltas[idx - 1])
        delta_deg = None if delta_rad is None else float(np.degrees(delta_rad))
        records.append(
            ChannelPhaseRecord(
                serial=idx + 1,
                channel=channel,
                amplitude=float(amplitudes[idx]),
                phase_rad=float(phases[idx]),
                phase_deg=float(np.degrees(phases[idx])),
                unwrapped_phase_rad=float(unwrapped[idx]),
                unwrapped_phase_deg=float(np.degrees(unwrapped[idx])),
                delta_from_previous_rad=delta_rad,
                delta_from_previous_deg=delta_deg,
            )
        )
    return records, mean_delta_rad, mean_delta_deg


def angle_from_phase_delta(mean_delta_rad: float, params: RadarParams, antenna_spacing_m: float) -> tuple[float, float]:
    argument = wavelength_m(params) * mean_delta_rad / (2.0 * np.pi * antenna_spacing_m)
    angle_deg = float(np.degrees(np.arcsin(np.clip(argument, -1.0, 1.0))))
    return angle_deg, float(argument)


def compute_range_angle_map(
    range_spectrum: np.ndarray,
    ranges_m: np.ndarray,
    params: RadarParams,
    antenna_spacing_m: float,
    angle_fft_size: int,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    nfft = max(int(angle_fft_size), range_spectrum.shape[0])
    channel_window = np.hamming(range_spectrum.shape[0]).reshape(-1, 1)
    angle_spectrum = np.fft.fftshift(
        np.fft.fft(range_spectrum * channel_window, n=nfft, axis=0),
        axes=0,
    )
    angle_axis_deg, valid_angle = angle_axis_and_mask(nfft, params, antenna_spacing_m)
    range_mask = (ranges_m >= 0.0) & (ranges_m <= max_range_m)
    if not np.any(range_mask):
        raise ValueError("最大显示距离过小，距离-角度图没有可显示的距离单元")
    return ranges_m[range_mask], angle_axis_deg, angle_spectrum[valid_angle][:, range_mask], nfft


def normalized_power_db(power: np.ndarray, floor_db: float) -> np.ndarray:
    power_max = float(np.max(power)) if power.size else 0.0
    return 10.0 * np.log10(np.maximum(power / max(power_max, EPS), 10.0 ** (floor_db / 10.0)))


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def _box_sum(integral: np.ndarray, a0: int, a1: int, r0: int, r1: int) -> float:
    return float(integral[a1, r1] - integral[a0, r1] - integral[a1, r0] + integral[a0, r0])


def ca_cfar_2d(
    power: np.ndarray,
    ranges_m: np.ndarray,
    angles_deg: np.ndarray,
    min_range_m: float,
    train_range: int,
    guard_range: int,
    train_angle: int,
    guard_angle: int,
    threshold_db: float,
    max_detections: int,
    min_gap_m: float,
    min_gap_deg: float,
) -> list[CfarDetection]:
    if power.size == 0:
        return []

    n_angles, n_ranges = power.shape
    train_range = max(1, int(train_range))
    guard_range = max(0, int(guard_range))
    train_angle = max(1, int(train_angle))
    guard_angle = max(0, int(guard_angle))
    range_margin = train_range + guard_range
    angle_margin = train_angle + guard_angle
    if n_ranges <= 2 * range_margin or n_angles <= 2 * angle_margin:
        return []

    valid_range_indices = np.where(ranges_m >= min_range_m)[0]
    if valid_range_indices.size == 0:
        return []

    range_start = max(range_margin, int(valid_range_indices[0]))
    range_stop = n_ranges - range_margin
    angle_start = angle_margin
    angle_stop = n_angles - angle_margin
    threshold_scale = 10.0 ** (threshold_db / 10.0)
    integral = _integral_image(power)
    power_max = float(np.max(power))

    # 预筛选：跳过功率过低的像素（快速剔除噪声）
    global_median = float(np.median(power[angle_start:angle_stop, range_start:range_stop]))
    pre_threshold = max(global_median * threshold_scale * 0.5, 1e-30)
    power_roi = power[angle_start:angle_stop, range_start:range_stop]
    candidate_mask = power_roi > pre_threshold
    candidate_rows, candidate_cols = np.where(candidate_mask)
    # 只对候选像素做完整 CFAR 检测
    candidates: list[tuple[float, int, int]] = []
    for pos in range(len(candidate_rows)):
        angle_idx = angle_start + candidate_rows[pos]
        range_idx = range_start + candidate_cols[pos]
        cell_power = float(power[angle_idx, range_idx])
        if cell_power <= 0.0:
            continue
        outer_a0 = angle_idx - angle_margin
        outer_a1 = angle_idx + angle_margin + 1
        guard_a0 = angle_idx - guard_angle
        guard_a1 = angle_idx + guard_angle + 1
        outer_r0 = range_idx - range_margin
        outer_r1 = range_idx + range_margin + 1
        guard_r0 = range_idx - guard_range
        guard_r1 = range_idx + guard_range + 1

        total_sum = _box_sum(integral, outer_a0, outer_a1, outer_r0, outer_r1)
        guard_sum = _box_sum(integral, guard_a0, guard_a1, guard_r0, guard_r1)
        total_count = (outer_a1 - outer_a0) * (outer_r1 - outer_r0)
        guard_count = (guard_a1 - guard_a0) * (guard_r1 - guard_r0)
        noise_count = total_count - guard_count
        if noise_count <= 0:
            continue
        noise_power = max((total_sum - guard_sum) / noise_count, EPS)
        if cell_power < noise_power * threshold_scale:
            continue
        local = power[
            max(0, angle_idx - 1) : min(n_angles, angle_idx + 2),
            max(0, range_idx - 1) : min(n_ranges, range_idx + 2),
        ]
        if cell_power < float(np.max(local)):
            continue
        candidates.append((cell_power, angle_idx, range_idx))

    detections: list[CfarDetection] = []
    for cell_power, angle_idx, range_idx in sorted(candidates, reverse=True):
        range_m = float(ranges_m[range_idx])
        angle_deg = float(angles_deg[angle_idx])
        if any(abs(range_m - item.range_m) < min_gap_m and abs(angle_deg - item.angle_deg) < min_gap_deg for item in detections):
            continue
        detections.append(
            CfarDetection(
                serial=len(detections) + 1,
                range_m=range_m,
                angle_deg=angle_deg,
                relative_power_db=10.0 * np.log10(max(cell_power / max(power_max, EPS), EPS)),
                range_index=range_idx,
                angle_index=angle_idx,
            )
        )
        if len(detections) >= max_detections:
            break
    return detections


def analyze_single_target(
    frame_number: int,
    virtual_channels: tuple[int, ...],
    virtual_cube: np.ndarray,
    params: RadarParams,
    window_name: str,
    chirp_choice: str,
    zero_padding_power: int,
    min_range_m: float,
    max_display_range_m: float,
    antenna_spacing_m: float,
    angle_fft_size: int,
    phase_correction: np.ndarray | None = None,
) -> tuple[SingleTargetResult, np.ndarray]:
    ranges_m, range_spectrum, nfft_range, chirp_label = compute_range_spectrum(
        virtual_cube,
        params,
        window_name,
        chirp_choice,
        zero_padding_power,
    )
    if phase_correction is not None:
        if phase_correction.shape[0] != range_spectrum.shape[0]:
            raise ValueError("0 度校准通道数与当前虚拟通道数不一致")
        range_spectrum = range_spectrum * phase_correction.reshape(-1, 1)
    hrrp_power, hrrp_db = hrrp_from_range_spectrum(range_spectrum)
    target_index = find_strongest_range_peak(ranges_m, hrrp_power, min_range_m, max_display_range_m)
    channel_vector = range_spectrum[:, target_index]
    records, mean_delta_rad, mean_delta_deg = compute_channel_phase_records(channel_vector, virtual_channels)
    phase_angle_deg, phase_sin_argument = angle_from_phase_delta(mean_delta_rad, params, antenna_spacing_m)
    angle_axis_deg, angle_spectrum_db, angle_fft_angle_deg, nfft_angle = compute_angle_spectrum(
        channel_vector,
        params,
        antenna_spacing_m,
        angle_fft_size,
    )
    time_signal, time_chirp_label = compute_time_signal(virtual_cube, chirp_choice)

    return (
        SingleTargetResult(
            frame_number=frame_number,
            chirp_label=time_chirp_label if time_chirp_label == chirp_label else chirp_label,
            virtual_channels=virtual_channels,
            params=params,
            antenna_spacing_m=antenna_spacing_m,
            nfft_range=nfft_range,
            nfft_angle=nfft_angle,
            range_resolution_m=C0 / (2.0 * params.bandwidth_hz),
            max_display_range_m=max_display_range_m,
            time_us=np.arange(params.samples_per_chirp) / params.sample_rate_hz * 1e6,
            time_signal=time_signal,
            ranges_m=ranges_m,
            hrrp_db=hrrp_db,
            hrrp_power=hrrp_power,
            target_index=target_index,
            target_range_m=float(ranges_m[target_index]),
            channel_records=records,
            mean_delta_rad=mean_delta_rad,
            mean_delta_deg=mean_delta_deg,
            phase_angle_deg=phase_angle_deg,
            phase_sin_argument=phase_sin_argument,
            angle_axis_deg=angle_axis_deg,
            angle_spectrum_db=angle_spectrum_db,
            angle_fft_angle_deg=angle_fft_angle_deg,
        ),
        range_spectrum,
    )


def analyze_multi_target(
    frame_number: int,
    chirp_label: str,
    virtual_channels: tuple[int, ...],
    time_us: np.ndarray,
    time_signal: np.ndarray,
    range_spectrum: np.ndarray,
    ranges_m: np.ndarray,
    params: RadarParams,
    antenna_spacing_m: float,
    angle_fft_size: int,
    max_display_range_m: float,
    min_range_m: float,
    cfar_train_range: int,
    cfar_guard_range: int,
    cfar_train_angle: int,
    cfar_guard_angle: int,
    cfar_threshold_db: float,
    max_detections: int,
    min_gap_m: float,
    min_gap_deg: float,
) -> MultiTargetResult:
    display_ranges, angle_axis_deg, range_angle_map, nfft_angle = compute_range_angle_map(
        range_spectrum,
        ranges_m,
        params,
        antenna_spacing_m,
        angle_fft_size,
        max_display_range_m,
    )
    range_angle_power = np.abs(range_angle_map) ** 2
    range_angle_db = normalized_power_db(range_angle_power, RANGE_ANGLE_DB_FLOOR)
    detections = ca_cfar_2d(
        range_angle_power,
        display_ranges,
        angle_axis_deg,
        min_range_m=min_range_m,
        train_range=cfar_train_range,
        guard_range=cfar_guard_range,
        train_angle=cfar_train_angle,
        guard_angle=cfar_guard_angle,
        threshold_db=cfar_threshold_db,
        max_detections=max_detections,
        min_gap_m=min_gap_m,
        min_gap_deg=min_gap_deg,
    )
    return MultiTargetResult(
        frame_number=frame_number,
        chirp_label=chirp_label,
        virtual_channels=virtual_channels,
        params=params,
        antenna_spacing_m=antenna_spacing_m,
        nfft_range=0,
        nfft_angle=nfft_angle,
        max_display_range_m=max_display_range_m,
        time_us=time_us,
        time_signal=time_signal,
        ranges_m=display_ranges,
        angle_axis_deg=angle_axis_deg,
        range_angle_complex=range_angle_map,
        range_angle_power=range_angle_power,
        range_angle_db=range_angle_db,
        detections=detections,
    )


def analyze_angle(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int,
    virtual_channels_text: str,
    window_name: str,
    chirp_choice: str,
    zero_padding_power: int,
    min_range_m: float,
    max_display_range_m: float,
    antenna_spacing_m: float,
    angle_fft_size: int,
    cfar_train_range: int,
    cfar_guard_range: int,
    cfar_train_angle: int,
    cfar_guard_angle: int,
    cfar_threshold_db: float,
    max_detections: int,
    min_gap_m: float,
    min_gap_deg: float,
    phase_correction: np.ndarray | None = None,
) -> tuple[SingleTargetResult, MultiTargetResult, np.ndarray]:
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = ensure_antenna_cube(data[frame_index])
    channel_indices, channels = parse_virtual_channels(virtual_channels_text, frame_data.shape[0])
    virtual_cube = frame_data[channel_indices, :, :]
    single, range_spectrum = analyze_single_target(
        frame_number,
        channels,
        virtual_cube,
        params,
        window_name,
        chirp_choice,
        zero_padding_power,
        min_range_m,
        max_display_range_m,
        antenna_spacing_m,
        angle_fft_size,
        phase_correction,
    )
    multi = analyze_multi_target(
        frame_number,
        single.chirp_label,
        channels,
        single.time_us,
        single.time_signal,
        range_spectrum,
        single.ranges_m,
        params,
        antenna_spacing_m,
        angle_fft_size,
        max_display_range_m,
        min_range_m,
        cfar_train_range,
        cfar_guard_range,
        cfar_train_angle,
        cfar_guard_angle,
        cfar_threshold_db,
        max_detections,
        min_gap_m,
        min_gap_deg,
    )
    return single, multi, range_spectrum


def build_phase_calibration(single_result: SingleTargetResult, range_spectrum: np.ndarray) -> CalibrationResult:
    channel_vector = range_spectrum[:, single_result.target_index]
    phases = np.angle(channel_vector)
    correction = np.exp(-1j * phases)
    return CalibrationResult(
        target_range_m=single_result.target_range_m,
        target_index=single_result.target_index,
        virtual_channels=single_result.virtual_channels,
        channel_phases_rad=phases,
        channel_phases_deg=np.degrees(phases),
        phase_correction=correction,
    )
