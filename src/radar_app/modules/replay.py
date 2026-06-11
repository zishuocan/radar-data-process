"""多帧数据回放与动画模块"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radar_app.core.params import C0, RadarParams
from radar_app.core.signal import make_window, next_fft_size


@dataclass
class ReplayData:
    times_s: np.ndarray          # (n_frames,) 每帧的时间戳
    ranges_m: np.ndarray         # (n_ranges,) 距离轴
    waterfall_db: np.ndarray     # (n_frames, n_ranges) 距离像瀑布图（dB）
    nfft: int
    antenna_label: str
    chirp_label: str


def compute_replay_data(
    data: np.ndarray,
    params: RadarParams,
    antenna_choice: str,
    chirp_choice: str,
    window_name: str = "hamming",
    zero_padding_power: int = 5,
) -> ReplayData:
    """对所有帧逐帧计算距离像，生成瀑布图数据

    Args:
        data: (n_frames, [n_antennas,] n_chirps, n_samples) 雷达数据
        params: 雷达参数
        antenna_choice: 天线选择（"1", "2", ..., "平均"）
        chirp_choice: Chirp 选择（"1", "2", ..., "平均"）
        window_name: 窗函数名
        zero_padding_power: 补零幂

    Returns:
        ReplayData 对象
    """
    # 确保数据为 (n_frames, n_antennas, n_chirps, n_samples)
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]

    n_frames, n_antennas, n_chirps, n_samples = data.shape

    # 解析天线选择
    antenna_indices = _parse_choice(antenna_choice, n_antennas)
    chirp_indices = _parse_choice(chirp_choice, n_chirps)

    nfft = next_fft_size(params.samples_per_chirp, zero_padding_power)
    if np.iscomplexobj(data):
        freqs = np.fft.fftfreq(nfft, d=1.0 / params.sample_rate_hz)
        positive = freqs >= 0
        freqs = freqs[positive]
        n_ranges = len(freqs)
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / params.sample_rate_hz)
        n_ranges = len(freqs)

    ranges_m = C0 * freqs / (2.0 * params.slope_hz_s)

    # 逐帧处理（避免一次性加载过大内存）
    waterfall = np.zeros((n_frames, n_ranges), dtype=np.float64)
    window = make_window(window_name, n_samples).reshape(1, -1)

    for frame_idx in range(n_frames):
        # 取指定帧、天线、chirp 的数据
        selected = data[frame_idx, antenna_indices, :, :][:, chirp_indices, :]
        # selected shape: (n_selected_ant, n_selected_chirp, n_samples)
        # 合并天线和 chirp 维
        n_combined = selected.shape[0] * selected.shape[1]
        selected = selected.reshape(n_combined, n_samples)
        selected = selected.astype(np.complex128 if np.iscomplexobj(selected) else np.float64)

        # 去直流
        selected = selected - np.mean(selected, axis=-1, keepdims=True)
        # 加窗
        selected = selected * window

        # FFT
        if np.iscomplexobj(selected):
            spectrum = np.fft.fft(selected, n=nfft, axis=-1)[:, positive]
        else:
            spectrum = np.fft.rfft(selected, n=nfft, axis=-1)

        # 幅度平均后归一化
        magnitude = np.mean(np.abs(spectrum), axis=0)
        mag_max = float(np.max(magnitude))
        if mag_max > 0:
            magnitude = magnitude / mag_max

        # 转 dB
        waterfall[frame_idx] = 20.0 * np.log10(np.maximum(magnitude, 1e-12))

    # 归一化 dB 到合理范围
    wf_max = float(np.max(waterfall))
    waterfall = np.maximum(waterfall - wf_max, -60.0)

    times_s = np.arange(n_frames, dtype=np.float64) * (params.frame_period_ms / 1000.0)

    ant_label = _choice_label(antenna_choice, n_antennas, "天线")
    chirp_label = _choice_label(chirp_choice, n_chirps, "Chirp")

    return ReplayData(
        times_s=times_s,
        ranges_m=ranges_m,
        waterfall_db=waterfall,
        nfft=nfft,
        antenna_label=ant_label,
        chirp_label=chirp_label,
    )


def _parse_choice(text: str, upper_bound: int) -> list[int]:
    """解析"1"/"2"/"平均"等为索引列表"""
    normalized = text.strip().lower()
    if normalized in {"", "平均", "avg", "average", "mean", "all"}:
        return list(range(upper_bound))
    idx = max(0, min(upper_bound - 1, int(float(normalized)) - 1))
    return [idx]


def _choice_label(text: str, upper_bound: int, name: str) -> str:
    normalized = text.strip().lower()
    if normalized in {"", "平均", "avg", "average", "mean", "all"}:
        return f"平均{name}"
    return f"{name}{text}"
