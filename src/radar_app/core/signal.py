from __future__ import annotations

from math import ceil

import numpy as np

from .params import C0, RadarParams


EPS = 1e-300
WINDOW_OPTIONS = ["rect", "hann", "hamming", "blackman", "bartlett", "kaiser", "flattop"]


def normalize_choice(value: str) -> str:
    return value.strip().lower().replace("-", "").replace("_", "").replace("窗", "")


def make_window(name: str, n: int) -> np.ndarray:
    normalized = normalize_choice(name)
    if normalized in {"rect", "rectangle", "rectangular", "boxcar", "矩形"}:
        return np.ones(n)
    if normalized in {"hann", "hanning"}:
        return np.hanning(n)
    if normalized == "hamming":
        return np.hamming(n)
    if normalized == "blackman":
        return np.blackman(n)
    if normalized == "bartlett":
        return np.bartlett(n)
    if normalized == "kaiser":
        return np.kaiser(n, beta=8.6)
    if normalized == "flattop":
        idx = np.arange(n, dtype=np.float64)
        phase = 2.0 * np.pi * idx / max(1, n - 1)
        return (
            0.21557895
            - 0.41663158 * np.cos(phase)
            + 0.277263158 * np.cos(2.0 * phase)
            - 0.083578947 * np.cos(3.0 * phase)
            + 0.006947368 * np.cos(4.0 * phase)
        )
    raise ValueError(f"未知窗函数: {name}")


def next_fft_size(sample_count: int, zero_padding_power: int = 0) -> int:
    return 1 << (int(ceil(np.log2(max(1, sample_count)))) + max(0, int(zero_padding_power)))


def positive_range_fft(
    samples: np.ndarray,
    params: RadarParams,
    axis: int = -1,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if nfft is None:
        nfft = next_fft_size(params.samples_per_chirp)

    if np.iscomplexobj(samples):
        spectrum = np.fft.fft(samples, n=nfft, axis=axis)
        freqs = np.fft.fftfreq(nfft, d=1.0 / params.sample_rate_hz)
        positive = freqs >= 0
        spectrum = np.take(spectrum, np.where(positive)[0], axis=axis)
        freqs = freqs[positive]
    else:
        spectrum = np.fft.rfft(samples, n=nfft, axis=axis)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / params.sample_rate_hz)

    ranges_m = C0 * freqs / (2.0 * params.slope_hz_s)
    return ranges_m, spectrum


def normalized_db_from_amplitude(amplitude: np.ndarray, floor_db: float) -> np.ndarray:
    values = np.asarray(amplitude, dtype=np.float64)
    if values.max(initial=0.0) > 0:
        values = values / values.max()
    return 20.0 * np.log10(np.maximum(values, 10.0 ** (floor_db / 20.0)))


def normalized_db_from_power(power: np.ndarray, floor_db: float) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    power_max = float(values.max(initial=0.0))
    return 10.0 * np.log10(np.maximum(values / max(power_max, EPS), 10.0 ** (floor_db / 10.0)))
