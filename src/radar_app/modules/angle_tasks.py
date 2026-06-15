"""测角实验任务模块 — 分步演示测角全流程"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import make_window, next_fft_size, normalized_db_from_amplitude
from radar_app.modules.angle import (
    HRRP_DB_FLOOR,
    RANGE_ANGLE_DB_FLOOR,
    angle_axis_and_mask,
    ca_cfar_2d,
    compute_range_spectrum,
    compute_time_signal,
    find_strongest_range_peak,
    hrrp_from_range_spectrum,
    parse_virtual_channels,
    wavelength_m,
)


@dataclass
class Task1Result:
    """任务1: 一维距离像"""
    ranges_m: np.ndarray
    hrrp_db: np.ndarray
    target_index: int
    target_range_m: float
    channel_amplitudes: list[float]
    channel_phases_deg: list[float]


@dataclass
class Task2Result:
    """任务2: 相位解缠绕与相位差"""
    channel_records: list[dict]
    mean_delta_deg: float


@dataclass
class Task3Result:
    """任务3: 方位 FFT"""
    angle_axis_deg: np.ndarray
    angle_spectrum_db: np.ndarray
    peak_angle_deg: float


@dataclass
class Task4Result:
    """任务4: 二维 FFT → 距离-角度复数图像"""
    ranges_m: np.ndarray
    angle_axis_deg: np.ndarray
    range_angle_db: np.ndarray
    complex_image: np.ndarray


@dataclass
class Task5Result:
    """任务5: CFAR 检测"""
    ranges_m: np.ndarray
    angle_axis_deg: np.ndarray
    range_angle_db: np.ndarray
    detections: list[dict]


def run_task1(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int = 1,
    virtual_channels_text: str = "1,2,3,4,5,6,7,8",
    window_name: str = "hamming",
    chirp_choice: str = "平均",
    zero_padding_power: int = 5,
    min_range_m: float = 0.0,
    max_display_range_m: float = 6.0,
) -> Task1Result:
    """任务1: 一维距离像 + 目标距离单元通道幅度/相位"""
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = data[frame_index]
    if frame_data.ndim != 3:
        raise ValueError("需要多通道数据 (antenna/chirp/sample)")
    channel_indices, channels = parse_virtual_channels(virtual_channels_text, frame_data.shape[0])
    virtual_cube = frame_data[channel_indices, :, :]

    ranges_m, range_spectrum, _, _, _ = compute_range_spectrum(
        virtual_cube, params, window_name, chirp_choice, zero_padding_power,
    )
    hrrp_power, hrrp_db = hrrp_from_range_spectrum(range_spectrum)
    target_index = find_strongest_range_peak(ranges_m, hrrp_power, min_range_m, max_display_range_m)
    target_range_m = float(ranges_m[target_index])

    channel_vector = range_spectrum[:, target_index]
    amplitudes = [float(abs(v)) for v in channel_vector]
    phases_deg = [float(np.degrees(np.angle(v))) for v in channel_vector]

    return Task1Result(
        ranges_m=ranges_m,
        hrrp_db=hrrp_db,
        target_index=target_index,
        target_range_m=target_range_m,
        channel_amplitudes=amplitudes,
        channel_phases_deg=phases_deg,
    )


def run_task2(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int = 1,
    virtual_channels_text: str = "1,2,3,4,5,6,7,8",
    window_name: str = "hamming",
    chirp_choice: str = "平均",
    zero_padding_power: int = 5,
    min_range_m: float = 0.0,
    max_display_range_m: float = 6.0,
) -> Task2Result:
    """任务2: 相位解缠绕 + 相邻通道相位差分"""
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = data[frame_index]
    if frame_data.ndim != 3:
        raise ValueError("需要多通道数据")
    channel_indices, channels = parse_virtual_channels(virtual_channels_text, frame_data.shape[0])
    virtual_cube = frame_data[channel_indices, :, :]

    ranges_m, range_spectrum, _, _, _ = compute_range_spectrum(
        virtual_cube, params, window_name, chirp_choice, zero_padding_power,
    )
    hrrp_power, hrrp_db = hrrp_from_range_spectrum(range_spectrum)
    target_index = find_strongest_range_peak(ranges_m, hrrp_power, min_range_m, max_display_range_m)

    channel_vector = range_spectrum[:, target_index]
    phases = np.angle(channel_vector)
    unwrapped = np.unwrap(phases)
    deltas = np.diff(unwrapped)
    mean_delta_deg = float(np.degrees(np.mean(deltas))) if deltas.size else 0.0

    records: list[dict] = []
    for idx, ch in enumerate(channels):
        rec = {
            "channel": ch,
            "phase_deg": float(np.degrees(phases[idx])),
            "unwrapped_deg": float(np.degrees(unwrapped[idx])),
            "delta_deg": float(np.degrees(deltas[idx - 1])) if idx > 0 else None,
        }
        records.append(rec)

    return Task2Result(
        channel_records=records,
        mean_delta_deg=mean_delta_deg,
    )


def run_task3(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int = 1,
    virtual_channels_text: str = "1,2,3,4,5,6,7,8",
    window_name: str = "hamming",
    chirp_choice: str = "平均",
    zero_padding_power: int = 5,
    min_range_m: float = 0.0,
    max_display_range_m: float = 6.0,
    angle_fft_size: int = 256,
    antenna_spacing_m: float = 0.002,
) -> Task3Result:
    """任务3: 目标距离单元方位 FFT → 方位角"""
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = data[frame_index]
    if frame_data.ndim != 3:
        raise ValueError("需要多通道数据")
    channel_indices, channels = parse_virtual_channels(virtual_channels_text, frame_data.shape[0])
    virtual_cube = frame_data[channel_indices, :, :]

    ranges_m, range_spectrum, _, _, _ = compute_range_spectrum(
        virtual_cube, params, window_name, chirp_choice, zero_padding_power,
    )
    hrrp_power, _ = hrrp_from_range_spectrum(range_spectrum)
    target_index = find_strongest_range_peak(ranges_m, hrrp_power, min_range_m, max_display_range_m)
    channel_vector = range_spectrum[:, target_index]

    # 方位 FFT（加 hamming 窗）
    nfft = max(angle_fft_size, len(channel_vector))
    window = np.hamming(len(channel_vector))
    spectrum = np.fft.fftshift(np.fft.fft(channel_vector * window, n=nfft))
    angle_axis_deg, valid_mask = angle_axis_and_mask(nfft, params, antenna_spacing_m)
    valid_angles = angle_axis_deg

    spectrum_power = np.abs(spectrum) ** 2
    # 只取有效角度范围
    if not np.all(valid_mask):
        spectrum_power = spectrum_power[valid_mask]
    else:
        spectrum_power = spectrum_power
    spectrum_db = normalized_db_from_amplitude(np.sqrt(spectrum_power), -50.0)
    peak_idx = int(np.argmax(spectrum_power))
    peak_angle = float(valid_angles[peak_idx]) if peak_idx < len(valid_angles) else 0.0

    return Task3Result(
        angle_axis_deg=valid_angles,
        angle_spectrum_db=spectrum_db,
        peak_angle_deg=peak_angle,
    )


def run_task4(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int = 1,
    virtual_channels_text: str = "1,2,3,4,5,6,7,8",
    window_name: str = "hamming",
    chirp_choice: str = "平均",
    zero_padding_power: int = 5,
    angle_fft_size: int = 256,
    antenna_spacing_m: float = 0.002,
    max_range_m: float = 6.0,
) -> Task4Result:
    """任务4: 二维 FFT → 距离-角度复数图像"""
    frame_index = max(0, min(data.shape[0] - 1, frame_number - 1))
    frame_data = data[frame_index]
    if frame_data.ndim != 3:
        raise ValueError("需要多通道数据")
    channel_indices, channels = parse_virtual_channels(virtual_channels_text, frame_data.shape[0])
    virtual_cube = frame_data[channel_indices, :, :]

    ranges_m, range_spectrum, _, _, _ = compute_range_spectrum(
        virtual_cube, params, window_name, chirp_choice, zero_padding_power,
    )

    # 方位维 FFT（沿通道维）
    nfft_angle = max(angle_fft_size, range_spectrum.shape[0])
    channel_window = np.hamming(range_spectrum.shape[0]).reshape(-1, 1)
    angle_complex = np.fft.fftshift(
        np.fft.fft(range_spectrum * channel_window, n=nfft_angle, axis=0),
        axes=0,
    )

    angle_axis_deg, valid_mask = angle_axis_and_mask(nfft_angle, params, antenna_spacing_m)

    # 截取有效角度和距离范围
    range_mask = (ranges_m >= 0.0) & (ranges_m <= max_range_m)
    display_ranges = ranges_m[range_mask]

    if not np.all(valid_mask):
        angle_complex = angle_complex[valid_mask]
        display_angles = angle_axis_deg
    else:
        display_angles = angle_axis_deg

    angle_complex = angle_complex[:, range_mask]
    power = np.abs(angle_complex) ** 2
    power_db = 10.0 * np.log10(np.maximum(power / max(np.max(power), 1e-30), 1e-15))
    power_db = np.maximum(power_db, -55.0)

    return Task4Result(
        ranges_m=display_ranges,
        angle_axis_deg=display_angles,
        range_angle_db=power_db,
        complex_image=angle_complex,
    )


def run_task5(
    data: np.ndarray,
    params: RadarParams,
    frame_number: int = 1,
    virtual_channels_text: str = "1,2,3,4,5,6,7,8",
    window_name: str = "hamming",
    chirp_choice: str = "平均",
    zero_padding_power: int = 5,
    angle_fft_size: int = 256,
    antenna_spacing_m: float = 0.002,
    max_range_m: float = 6.0,
    min_range_m: float = 0.0,
    cfar_train_range: int = 8,
    cfar_guard_range: int = 2,
    cfar_train_angle: int = 6,
    cfar_guard_angle: int = 2,
    cfar_threshold_db: float = 0.0,
    max_detections: int = 5,
    min_gap_m: float = 0.3,
    min_gap_deg: float = 5.0,
) -> Task5Result:
    """任务5: 距离-角度 CFAR 检测"""
    task4 = run_task4(
        data, params, frame_number, virtual_channels_text,
        window_name, chirp_choice, zero_padding_power,
        angle_fft_size, antenna_spacing_m, max_range_m,
    )

    power = 10.0 ** (task4.range_angle_db / 10.0)

    detections_raw = ca_cfar_2d(
        power,
        task4.ranges_m,
        task4.angle_axis_deg,
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

    detections = [
        {
            "serial": d.serial,
            "range_m": d.range_m,
            "angle_deg": d.angle_deg,
            "relative_power_db": d.relative_power_db,
        }
        for d in detections_raw
    ]

    return Task5Result(
        ranges_m=task4.ranges_m,
        angle_axis_deg=task4.angle_axis_deg,
        range_angle_db=task4.range_angle_db,
        detections=detections,
    )
