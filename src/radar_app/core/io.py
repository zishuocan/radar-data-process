from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .params import RadarParams


AUTO_FORMAT_LABEL = "自动"


@dataclass(frozen=True)
class DataFormat:
    key: str
    label: str
    dtype: np.dtype
    bytes_per_point: int
    is_complex: bool = False
    antennas: int = 1
    per_frame_header_bytes: int = 0
    auto_priority: int = 1


DATA_FORMATS = {
    "int8_ant16": DataFormat(
        "int8_ant16",
        "int8 16通道原始帧",
        np.dtype("i1"),
        1,
        antennas=16,
        per_frame_header_bytes=8,
        auto_priority=4,
    ),
    "iq_int16": DataFormat("iq_int16", "int16 I/Q交织复数", np.dtype("<i2"), 4, True, auto_priority=3),
    "int16_real": DataFormat("int16_real", "int16 实数", np.dtype("<i2"), 2, auto_priority=2),
    "uint16_real": DataFormat("uint16_real", "uint16 实数", np.dtype("<u2"), 2, auto_priority=1),
}

FORMAT_OPTIONS = [AUTO_FORMAT_LABEL] + [fmt.label for fmt in DATA_FORMATS.values()]
FORMAT_LABEL_TO_KEY = {fmt.label: key for key, fmt in DATA_FORMATS.items()}
FORMAT_LABEL_TO_KEY.update({key: key for key in DATA_FORMATS})


@dataclass
class LoadedRadarData:
    data: np.ndarray
    header_bytes: int
    per_frame_header_bytes: int
    format_key: str
    format_label: str
    dropped_tail_bytes: int

    @property
    def frame_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def antenna_count(self) -> int:
        if self.data.ndim == 4:
            return int(self.data.shape[1])
        return 1


def frame_bytes(params: RadarParams, fmt: DataFormat) -> int:
    payload_bytes = (
        params.samples_per_chirp
        * params.chirps_per_frame
        * fmt.antennas
        * fmt.bytes_per_point
    )
    return payload_bytes + fmt.per_frame_header_bytes


def _known_header_candidates(file_size: int) -> list[int]:
    common = [248, 0, 128, 240, 256, 512, 1024, 2048, 4096]
    return [item for item in common if 0 <= item < file_size]


def infer_header_size(file_size: int, params: RadarParams, fmt: DataFormat) -> int:
    if fmt.per_frame_header_bytes:
        return 0

    bytes_per_frame = frame_bytes(params, fmt)
    best_header = 0
    best_score: tuple[int, int, int, int] | None = None
    for header in _known_header_candidates(file_size):
        remaining = file_size - header
        if remaining <= 0:
            continue
        frames = remaining // bytes_per_frame
        remainder = remaining % bytes_per_frame
        exact = 1 if remainder == 0 and frames > 0 else 0
        prefer_248 = 1 if header == 248 else 0
        prefer_common = 1 if header in (248, 0, 256, 512) else 0
        score = (exact, prefer_248, prefer_common, frames)
        if best_score is None or score > best_score:
            best_header = header
            best_score = score
    return best_header


def parse_header_text(header_text: str, file_size: int, params: RadarParams, fmt: DataFormat) -> int:
    text = header_text.strip().lower()
    if text in {"", "auto", "自动"}:
        return infer_header_size(file_size, params, fmt)

    header = int(float(text))
    if header < 0 or header >= file_size:
        raise ValueError("文件头长度必须大于等于 0 且小于文件大小")
    return header


def choose_auto_format(file_size: int, params: RadarParams, header_text: str) -> tuple[str, int]:
    best: tuple[int, int, int, int, str, int] | None = None
    for key, fmt in DATA_FORMATS.items():
        header = parse_header_text(header_text, file_size, params, fmt)
        bytes_per_frame = frame_bytes(params, fmt)
        remaining = max(0, file_size - header)
        frames = remaining // bytes_per_frame
        if frames <= 0:
            continue
        remainder = remaining % bytes_per_frame
        exact = 1 if remainder == 0 else 0
        prefer_248 = 1 if header == 248 else 0
        score = (fmt.auto_priority, exact, prefer_248, frames, key, header)
        if best is None or score[:4] > best[:4]:
            best = score

    if best is None:
        return "int16_real", 0
    return best[4], best[5]


def _format_key(format_label: str) -> str | None:
    if format_label.strip().lower() in {"", "auto", "自动"}:
        return None
    try:
        return FORMAT_LABEL_TO_KEY[format_label]
    except KeyError as exc:
        raise ValueError(f"未知数据格式: {format_label}") from exc


def load_radar_data(
    file_path: str | Path,
    params: RadarParams,
    format_label: str = AUTO_FORMAT_LABEL,
    header_text: str = AUTO_FORMAT_LABEL,
) -> LoadedRadarData:
    params.validate()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError("文件为空")

    key = _format_key(format_label)
    if key is None:
        format_key, header_bytes = choose_auto_format(file_size, params, header_text)
    else:
        format_key = key
        header_bytes = parse_header_text(header_text, file_size, params, DATA_FORMATS[format_key])

    fmt = DATA_FORMATS[format_key]
    frame_points = params.samples_per_chirp * params.chirps_per_frame

    if fmt.antennas > 1 or fmt.per_frame_header_bytes:
        return _load_multichannel(path, params, fmt, header_bytes, file_size, frame_points)
    return _load_single_channel(path, params, fmt, header_bytes, file_size, frame_points)


def _load_multichannel(
    path: Path,
    params: RadarParams,
    fmt: DataFormat,
    header_bytes: int,
    file_size: int,
    frame_points: int,
) -> LoadedRadarData:
    frame_payload_values = fmt.antennas * frame_points
    bytes_per_frame = frame_bytes(params, fmt)
    header_values = fmt.per_frame_header_bytes // fmt.dtype.itemsize
    frame_values = bytes_per_frame // fmt.dtype.itemsize
    if bytes_per_frame % fmt.dtype.itemsize != 0:
        raise ValueError("每帧头长度必须能被数据类型字节数整除")

    with path.open("rb") as file:
        file.seek(header_bytes)
        raw = np.fromfile(file, dtype=fmt.dtype)

    frames = (raw.size * fmt.dtype.itemsize) // bytes_per_frame
    if frames <= 0:
        raise ValueError("文件内容不足一帧，请检查数据格式或文件头长度")

    usable_values = frames * frame_values
    raw_frames = raw[:usable_values].reshape(frames, frame_values)
    payload = raw_frames[:, header_values : header_values + frame_payload_values]
    data = payload.astype(np.float64, copy=False).reshape(
        frames, fmt.antennas, params.chirps_per_frame, params.samples_per_chirp
    )
    used_bytes = frames * bytes_per_frame
    return LoadedRadarData(
        data=data,
        header_bytes=header_bytes,
        per_frame_header_bytes=fmt.per_frame_header_bytes,
        format_key=fmt.key,
        format_label=fmt.label,
        dropped_tail_bytes=max(0, file_size - header_bytes - used_bytes),
    )


def _load_single_channel(
    path: Path,
    params: RadarParams,
    fmt: DataFormat,
    header_bytes: int,
    file_size: int,
    frame_points: int,
) -> LoadedRadarData:
    with path.open("rb") as file:
        file.seek(header_bytes)
        raw = np.fromfile(file, dtype=fmt.dtype)

    if fmt.is_complex:
        usable_iq_values = (raw.size // (frame_points * 2)) * frame_points * 2
        if usable_iq_values <= 0:
            raise ValueError("文件内容不足一帧，请检查数据格式或文件头长度")
        raw = raw[:usable_iq_values].astype(np.float64, copy=False)
        complex_values = raw[0::2] + 1j * raw[1::2]
        frames = complex_values.size // frame_points
        data = complex_values.reshape(frames, params.chirps_per_frame, params.samples_per_chirp)
        used_bytes = usable_iq_values * fmt.dtype.itemsize
    else:
        usable_values = (raw.size // frame_points) * frame_points
        if usable_values <= 0:
            raise ValueError("文件内容不足一帧，请检查数据格式或文件头长度")
        raw = raw[:usable_values].astype(np.float64, copy=False)
        frames = raw.size // frame_points
        data = raw.reshape(frames, params.chirps_per_frame, params.samples_per_chirp)
        used_bytes = usable_values * fmt.dtype.itemsize

    return LoadedRadarData(
        data=data,
        header_bytes=header_bytes,
        per_frame_header_bytes=fmt.per_frame_header_bytes,
        format_key=fmt.key,
        format_label=fmt.label,
        dropped_tail_bytes=max(0, file_size - header_bytes - used_bytes),
    )
