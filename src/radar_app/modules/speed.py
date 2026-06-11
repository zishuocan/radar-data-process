from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import EPS, make_window, next_fft_size


HEATMAP_DB_FLOOR = -45.0
RANGE_DOPPLER_DB_FLOOR = -65.0
SLOW_TIME_OPTIONS = ["天线xChirp", "Chirp"]


@dataclass
class SpeedAnalysisResult:
    times_s: np.ndarray
    ranges_m: np.ndarray
    smoothed_ranges_m: np.ndarray
    speeds_mps: np.ndarray
    valid_mask: np.ndarray
    confidence_db: np.ndarray
    display_ranges_m: np.ndarray
    heatmap_db: np.ndarray
    track_indices: np.ndarray
    start_range_m: float
    end_range_m: float
    average_speed_mps: float
    fitted_speed_mps: float
    median_speed_mps: float
    max_speed_mps: float
    min_speed_mps: float
    valid_frame_count: int
    range_resolution_m: float
    max_range_m: float
    nfft: int


@dataclass
class RangeDopplerResult:
    velocities_mps: np.ndarray
    ranges_m: np.ndarray
    map_db: np.ndarray
    max_velocity_mps: float
    max_range_m: float
    max_db: float
    frame_number: int
    velocity_resolution_mps: float
    max_unambiguous_velocity_mps: float


def ensure_frame_antenna_chirp_sample(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        return data
    if data.ndim == 3:
        return data[:, np.newaxis, :, :]
    raise ValueError("雷达数据维度应为 frame/chirp/sample 或 frame/antenna/chirp/sample")


def parse_average_or_index(text: str, upper_bound: int) -> int | None:
    normalized = text.strip().lower()
    if normalized in {"", "平均", "avg", "average", "mean", "all"}:
        return None
    index = int(float(normalized)) - 1
    return max(0, min(upper_bound - 1, index))


def select_channels(data: np.ndarray, antenna_choice: str, chirp_choice: str) -> np.ndarray:
    cube = ensure_frame_antenna_chirp_sample(data)
    antenna_index = parse_average_or_index(antenna_choice, cube.shape[1])
    chirp_index = parse_average_or_index(chirp_choice, cube.shape[2])
    if antenna_index is not None:
        cube = cube[:, antenna_index : antenna_index + 1, :, :]
    if chirp_index is not None:
        cube = cube[:, :, chirp_index : chirp_index + 1, :]
    return cube.reshape(cube.shape[0], cube.shape[1] * cube.shape[2], cube.shape[3])


def compute_range_power(
    data: np.ndarray,
    params: RadarParams,
    antenna_choice: str,
    chirp_choice: str,
    window_name: str,
    zero_padding_power: int,
    suppress_static: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected = select_channels(data, antenna_choice, chirp_choice)
    selected = selected.astype(np.complex128 if np.iscomplexobj(selected) else np.float64)
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

    if suppress_static and spectrum.shape[0] > 1:
        spectrum = spectrum - np.mean(spectrum, axis=0, keepdims=True)

    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    ranges_m = C0 * freqs / (2.0 * params.slope_hz_s)
    return ranges_m, power, nfft


def top_peak_candidates(
    ranges_m: np.ndarray,
    power_row: np.ndarray,
    valid_indices: np.ndarray,
    max_candidates: int,
    min_margin_db: float,
    min_gap_m: float,
) -> np.ndarray:
    local_power = power_row[valid_indices]
    if local_power.size == 0:
        return np.array([], dtype=int)

    db = 10.0 * np.log10(np.maximum(local_power, EPS))
    median_db = float(np.median(db))
    left = np.r_[local_power[0], local_power[:-1]]
    right = np.r_[local_power[1:], local_power[-1]]
    local_maxima = np.where((local_power >= left) & (local_power >= right))[0]
    strong = local_maxima[db[local_maxima] >= median_db + min_margin_db]
    if strong.size == 0:
        strong = local_maxima
    if strong.size == 0:
        strong = np.arange(local_power.size)

    ranked = list(strong[np.argsort(local_power[strong])[::-1]])
    if len(ranked) < max_candidates:
        for item in np.argsort(local_power)[::-1]:
            if int(item) not in ranked:
                ranked.append(int(item))

    selected: list[int] = []
    for local_index in ranked:
        absolute_index = int(valid_indices[local_index])
        candidate_range = float(ranges_m[absolute_index])
        if all(abs(candidate_range - float(ranges_m[previous])) >= min_gap_m for previous in selected):
            selected.append(absolute_index)
        if len(selected) >= max_candidates:
            break
    if not selected:
        selected.append(int(valid_indices[int(np.argmax(local_power))]))
    return np.array(selected, dtype=int)


def track_target_range(
    ranges_m: np.ndarray,
    power: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    max_jump_m: float,
    min_margin_db: float,
    prefer_approaching: bool,
    max_candidates: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.where((ranges_m >= min_range_m) & (ranges_m <= max_range_m))[0]
    if valid_indices.size == 0:
        raise ValueError("搜索距离范围内没有可用距离单元")

    candidate_gap_m = max(0.18, 1.5 * float(np.median(np.diff(ranges_m))))
    candidate_lists = [
        top_peak_candidates(ranges_m, row, valid_indices, max_candidates, min_margin_db, candidate_gap_m)
        for row in power
    ]
    if any(candidates.size == 0 for candidates in candidate_lists):
        raise ValueError("未检测到可跟踪的运动目标峰值")

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    first_candidates = candidate_lists[0]
    first_db = 10.0 * np.log10(np.maximum(power[0, first_candidates], EPS))
    costs.append((np.max(first_db) - first_db) / 12.0)
    parents.append(np.full(first_candidates.size, -1, dtype=int))

    jump_scale = max(0.05, max_jump_m)
    for frame_index in range(1, len(candidate_lists)):
        prev_candidates = candidate_lists[frame_index - 1]
        current_candidates = candidate_lists[frame_index]
        current_db = 10.0 * np.log10(np.maximum(power[frame_index, current_candidates], EPS))
        emission_cost = (np.max(current_db) - current_db) / 12.0

        frame_costs = np.empty(current_candidates.size, dtype=np.float64)
        frame_parents = np.empty(current_candidates.size, dtype=int)
        prev_ranges = ranges_m[prev_candidates]
        current_ranges = ranges_m[current_candidates]
        for current_pos, current_range in enumerate(current_ranges):
            delta = current_range - prev_ranges
            transition_cost = 0.32 * (np.abs(delta) / jump_scale) ** 2
            if prefer_approaching:
                transition_cost += 1.8 * (np.maximum(delta, 0.0) / jump_scale) ** 2
            total_cost = costs[-1] + transition_cost
            best_parent = int(np.argmin(total_cost))
            frame_costs[current_pos] = total_cost[best_parent] + emission_cost[current_pos]
            frame_parents[current_pos] = best_parent
        costs.append(frame_costs)
        parents.append(frame_parents)

    path_positions = np.empty(len(candidate_lists), dtype=int)
    path_positions[-1] = int(np.argmin(costs[-1]))
    for frame_index in range(len(candidate_lists) - 1, 0, -1):
        path_positions[frame_index - 1] = parents[frame_index][path_positions[frame_index]]

    track_indices = np.array(
        [candidate_lists[i][path_positions[i]] for i in range(len(candidate_lists))],
        dtype=int,
    )
    tracked_power = power[np.arange(power.shape[0]), track_indices]
    noise_floor_db = np.median(10.0 * np.log10(np.maximum(power[:, valid_indices], EPS)), axis=1)
    track_db = 10.0 * np.log10(np.maximum(tracked_power, EPS))
    confidence_db = track_db - noise_floor_db
    valid_mask = confidence_db >= min_margin_db
    return track_indices, confidence_db, valid_mask


def fill_invalid_ranges(ranges_m: np.ndarray, times_s: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(valid_mask) >= 2:
        return np.interp(times_s, times_s[valid_mask], ranges_m[valid_mask])
    return ranges_m.copy()


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size < 3:
        return values.copy()
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, values.size if values.size % 2 else values.size - 1)
    if window_size <= 1:
        return values.copy()
    pad = window_size // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(padded, kernel, mode="valid")


def robust_speed_summary(
    times_s: np.ndarray,
    ranges_m: np.ndarray,
    speeds_mps: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    if np.count_nonzero(valid_mask) >= 2:
        valid_times = times_s[valid_mask]
        valid_ranges = ranges_m[valid_mask]
    else:
        valid_times = times_s
        valid_ranges = ranges_m

    start_range = float(valid_ranges[0])
    end_range = float(valid_ranges[-1])
    duration = max(float(valid_times[-1] - valid_times[0]), EPS)
    average_speed = (end_range - start_range) / duration
    fitted_speed = float(np.polyfit(valid_times, valid_ranges, deg=1)[0]) if valid_times.size >= 2 else float("nan")
    valid_speeds = speeds_mps[valid_mask] if np.count_nonzero(valid_mask) > 0 else speeds_mps
    return (
        start_range,
        end_range,
        average_speed,
        fitted_speed,
        float(np.median(valid_speeds)),
        float(np.max(valid_speeds)),
        float(np.min(valid_speeds)),
    )


def kalman_track_target_range(
    ranges_m: np.ndarray,
    power: np.ndarray,
    times_s: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    max_jump_m: float,
    min_margin_db: float,
    q_range: float = 0.5,
    q_velocity: float = 0.1,
    r_range: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """卡尔曼滤波目标跟踪

    状态: [range, velocity]^T
    观测: range（距离FFT峰值）

    Args:
        ranges_m: 距离轴
        power: (n_frames, n_ranges) 距离功率
        times_s: 每帧时间戳
        搜索范围/阈值参数同 track_target_range
        q_range/q_velocity: 过程噪声（越大越机动）
        r_range: 观测噪声（越大越信任预测）

    Returns:
        (track_indices, confidence_db, valid_mask) 同 track_target_range
    """
    valid_indices = np.where((ranges_m >= min_range_m) & (ranges_m <= max_range_m))[0]
    if valid_indices.size == 0:
        raise ValueError("搜索距离范围内没有可用距离单元")

    n_frames = power.shape[0]
    dt = times_s[1] - times_s[0] if n_frames > 1 else 0.1

    # 状态转移矩阵 F = [[1, dt], [0, 1]]
    # 观测矩阵 H = [[1, 0]]
    # 过程噪声协方差 Q
    # 观测噪声协方差 R
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = np.array([[q_range, 0.0], [0.0, q_velocity]], dtype=np.float64)
    R = np.array([[r_range]], dtype=np.float64)

    track_indices = np.zeros(n_frames, dtype=int)
    confidence_db = np.zeros(n_frames, dtype=np.float64)
    valid_mask = np.zeros(n_frames, dtype=bool)

    # 初始状态: 用第一帧的最强峰
    first_valid = valid_indices[np.argmax(power[0, valid_indices])]
    x = np.array([float(ranges_m[first_valid]), 0.0], dtype=np.float64)
    P = np.eye(2, dtype=np.float64) * 0.1
    track_indices[0] = first_valid
    noise_floor_db = np.median(10.0 * np.log10(np.maximum(power[0, valid_indices], EPS)))
    peak_db = 10.0 * np.log10(max(power[0, first_valid], EPS))
    confidence_db[0] = peak_db - noise_floor_db
    valid_mask[0] = confidence_db[0] >= min_margin_db

    # 逐帧跟踪
    for frame_idx in range(1, n_frames):
        # --- 预测 ---
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        predicted_range = x_pred[0]

        # --- 在预测位置附近搜索最佳候选 ---
        search_window = max_jump_m * 2.0
        candidate_indices = valid_indices[
            np.abs(ranges_m[valid_indices] - predicted_range) <= search_window
        ]

        if candidate_indices.size > 0:
            # 找功率最大的候选
            best_local = candidate_indices[np.argmax(power[frame_idx, candidate_indices])]
            z = float(ranges_m[best_local])

            # 计算该候选的信噪比
            nf_db = np.median(10.0 * np.log10(np.maximum(power[frame_idx, valid_indices], EPS)))
            p_db = 10.0 * np.log10(max(power[frame_idx, best_local], EPS))
            conf = p_db - nf_db

            if conf >= min_margin_db:
                # --- 更新（有效观测） ---
                y = z - H @ x_pred  # 创新
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x = x_pred + (K @ y).flatten()
                P = (np.eye(2) - K @ H) @ P_pred
                track_indices[frame_idx] = best_local
                confidence_db[frame_idx] = conf
                valid_mask[frame_idx] = True
            else:
                # 观测 SNR 不足 → 用预测值（coast）
                x = x_pred
                P = P_pred
                # 找最近的索引
                track_indices[frame_idx] = valid_indices[np.argmin(np.abs(ranges_m[valid_indices] - predicted_range))]
                confidence_db[frame_idx] = conf
                valid_mask[frame_idx] = False
        else:
            # 搜索窗口内无候选 → 用预测值
            x = x_pred
            P = P_pred
            track_indices[frame_idx] = valid_indices[np.argmin(np.abs(ranges_m[valid_indices] - predicted_range))]
            confidence_db[frame_idx] = -999.0
            valid_mask[frame_idx] = False

    return track_indices, confidence_db, valid_mask


def analyze_speed(
    data: np.ndarray,
    params: RadarParams,
    antenna_choice: str,
    chirp_choice: str,
    window_name: str,
    zero_padding_power: int,
    suppress_static: bool,
    min_range_m: float,
    max_range_m: float,
    max_jump_m: float,
    min_margin_db: float,
    prefer_approaching: bool,
    smooth_frames: int,
    track_method: str = "viterbi",
) -> SpeedAnalysisResult:
    ranges_m, power, nfft = compute_range_power(
        data,
        params,
        antenna_choice,
        chirp_choice,
        window_name,
        zero_padding_power,
        suppress_static,
    )
    max_range_m = min(max_range_m, float(ranges_m[-1]))
    if min_range_m >= max_range_m:
        raise ValueError("最小搜索距离必须小于最大搜索距离")

    display_mask = (ranges_m >= min_range_m) & (ranges_m <= max_range_m)
    display_ranges = ranges_m[display_mask]
    display_power = power[:, display_mask]
    heatmap_db = 10.0 * np.log10(np.maximum(display_power, EPS))
    heatmap_db = np.maximum(heatmap_db - float(np.max(heatmap_db)), HEATMAP_DB_FLOOR)

    track_indices, confidence_db, valid_mask = track_target_range(
        ranges_m,
        power,
        min_range_m,
        max_range_m,
        max_jump_m,
        min_margin_db,
        prefer_approaching,
    )

    frame_count = power.shape[0]
    times_s = np.arange(frame_count, dtype=np.float64) * (params.frame_period_ms / 1000.0)

    if track_method == "kalman":
        track_indices, confidence_db, valid_mask = kalman_track_target_range(
            ranges_m, power, times_s,
            min_range_m, max_range_m, max_jump_m, min_margin_db,
        )
    tracked_ranges = ranges_m[track_indices]
    filled_ranges = fill_invalid_ranges(tracked_ranges, times_s, valid_mask)
    smoothed_ranges = moving_average(filled_ranges, smooth_frames)
    speeds_mps = np.gradient(smoothed_ranges, times_s) if frame_count >= 2 else np.zeros_like(smoothed_ranges)

    start_range, end_range, avg_speed, fit_speed, median_speed, max_speed, min_speed = robust_speed_summary(
        times_s,
        smoothed_ranges,
        speeds_mps,
        valid_mask,
    )
    return SpeedAnalysisResult(
        times_s=times_s,
        ranges_m=tracked_ranges,
        smoothed_ranges_m=smoothed_ranges,
        speeds_mps=speeds_mps,
        valid_mask=valid_mask,
        confidence_db=confidence_db,
        display_ranges_m=display_ranges,
        heatmap_db=heatmap_db,
        track_indices=track_indices,
        start_range_m=start_range,
        end_range_m=end_range,
        average_speed_mps=avg_speed,
        fitted_speed_mps=fit_speed,
        median_speed_mps=median_speed,
        max_speed_mps=max_speed,
        min_speed_mps=min_speed,
        valid_frame_count=int(np.count_nonzero(valid_mask)),
        range_resolution_m=params.range_resolution_m,
        max_range_m=params.max_range_m,
        nfft=nfft,
    )


def select_antenna_cube(data: np.ndarray, antenna_choice: str) -> np.ndarray:
    cube = ensure_frame_antenna_chirp_sample(data)
    antenna_index = parse_average_or_index(antenna_choice, cube.shape[1])
    if antenna_index is None:
        return cube
    return cube[:, antenna_index : antenna_index + 1, :, :]


def range_fft_cube(
    cube: np.ndarray,
    params: RadarParams,
    window_name: str,
    zero_padding_power: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = cube.astype(np.complex128 if np.iscomplexobj(cube) else np.float64)
    selected = selected - np.mean(selected, axis=-1, keepdims=True)
    selected = selected * make_window(window_name, params.samples_per_chirp).reshape(1, 1, 1, -1)
    nfft = next_fft_size(params.samples_per_chirp, zero_padding_power)
    if np.iscomplexobj(selected):
        spectrum = np.fft.fft(selected, n=nfft, axis=-1)
        freqs = np.fft.fftfreq(nfft, d=1.0 / params.sample_rate_hz)
        positive = freqs >= 0
        return C0 * freqs[positive] / (2.0 * params.slope_hz_s), spectrum[..., positive]
    spectrum = np.fft.rfft(selected, n=nfft, axis=-1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / params.sample_rate_hz)
    return C0 * freqs / (2.0 * params.slope_hz_s), spectrum


def build_slow_time_matrix(frame_spectrum: np.ndarray, slow_time_mode: str) -> np.ndarray:
    if slow_time_mode == "Chirp":
        return np.mean(frame_spectrum, axis=0).T
    return frame_spectrum.reshape(frame_spectrum.shape[0] * frame_spectrum.shape[1], frame_spectrum.shape[2]).T


def compute_range_doppler(
    data: np.ndarray,
    params: RadarParams,
    frame_index: int,
    antenna_choice: str,
    window_name: str,
    zero_padding_power: int,
    chirp_period_us: float,
    doppler_fft_size: int,
    speed_limit_mps: float,
    min_range_m: float,
    max_range_m: float,
    suppress_static: bool,
    slow_time_mode: str,
) -> RangeDopplerResult:
    if chirp_period_us <= 0:
        raise ValueError("Chirp 周期必须为正数")
    if doppler_fft_size < 8:
        raise ValueError("Doppler FFT 点数至少为 8")
    if speed_limit_mps <= 0:
        raise ValueError("速度显示范围必须为正数")

    cube = select_antenna_cube(data, antenna_choice)
    frame_index = max(0, min(cube.shape[0] - 1, frame_index))
    ranges_m, spectrum = range_fft_cube(cube, params, window_name, zero_padding_power)
    if suppress_static and spectrum.shape[0] > 1:
        spectrum = spectrum - np.mean(spectrum, axis=0, keepdims=True)

    range_mask = (ranges_m >= min_range_m) & (ranges_m <= max_range_m)
    if not np.any(range_mask):
        raise ValueError("距离-多普勒图的距离范围内没有可用距离单元")

    slow_time = build_slow_time_matrix(spectrum[frame_index], slow_time_mode)
    slow_time = slow_time[range_mask]
    display_ranges = ranges_m[range_mask]

    slow_count = slow_time.shape[1]
    doppler_fft_size = max(int(doppler_fft_size), slow_count)
    doppler_window = np.hamming(slow_count).reshape(1, -1)
    slow_time = slow_time - np.mean(slow_time, axis=1, keepdims=True)
    doppler_spectrum = np.fft.fftshift(
        np.fft.fft(slow_time * doppler_window, n=doppler_fft_size, axis=1),
        axes=1,
    )

    wavelength_m = C0 / (params.start_freq_ghz * 1e9)
    chirp_period_s = chirp_period_us * 1e-6
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, d=chirp_period_s))
    velocities = doppler_freqs * wavelength_m / 2.0
    velocity_mask = (velocities >= -speed_limit_mps) & (velocities <= speed_limit_mps)
    if not np.any(velocity_mask):
        raise ValueError("速度显示范围内没有可用单元")

    display_velocities = velocities[velocity_mask]
    velocity_order = np.argsort(display_velocities)
    display_velocities = display_velocities[velocity_order]
    power = np.abs(doppler_spectrum[:, velocity_mask]) ** 2
    power = power[:, velocity_order]
    map_db_absolute = 10.0 * np.log10(np.maximum(power, EPS))
    max_db = float(np.max(map_db_absolute))
    map_db = np.maximum(map_db_absolute - max_db, RANGE_DOPPLER_DB_FLOOR)
    max_pos = np.unravel_index(int(np.argmax(map_db)), map_db.shape)
    velocity_resolution = wavelength_m / (2.0 * slow_count * chirp_period_s)
    max_unambiguous_velocity = wavelength_m / (4.0 * chirp_period_s)
    return RangeDopplerResult(
        velocities_mps=display_velocities,
        ranges_m=display_ranges,
        map_db=map_db,
        max_velocity_mps=float(display_velocities[max_pos[1]]),
        max_range_m=float(display_ranges[max_pos[0]]),
        max_db=max_db,
        frame_number=frame_index + 1,
        velocity_resolution_mps=float(velocity_resolution),
        max_unambiguous_velocity_mps=float(max_unambiguous_velocity),
    )
