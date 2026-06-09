from __future__ import annotations

from dataclasses import dataclass


C0 = 299_792_458.0


@dataclass(frozen=True)
class RadarParams:
    start_freq_ghz: float = 75.0
    slope_mhz_us: float = 30.0
    sample_rate_ksps: float = 3000.0
    samples_per_chirp: int = 256
    chirps_per_frame: int = 8
    frame_period_ms: float = 500.0

    @property
    def sample_rate_hz(self) -> float:
        return self.sample_rate_ksps * 1e3

    @property
    def slope_hz_s(self) -> float:
        return self.slope_mhz_us * 1e12

    @property
    def bandwidth_hz(self) -> float:
        return self.slope_hz_s * self.samples_per_chirp / self.sample_rate_hz

    @property
    def range_resolution_m(self) -> float:
        return C0 / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        return C0 * (self.sample_rate_hz / 2.0) / (2.0 * self.slope_hz_s)

    def validate(self) -> None:
        if self.start_freq_ghz <= 0:
            raise ValueError("起始频率必须为正数")
        if self.slope_mhz_us <= 0:
            raise ValueError("调频斜率必须为正数")
        if self.sample_rate_ksps <= 0:
            raise ValueError("采样率必须为正数")
        if self.samples_per_chirp <= 0:
            raise ValueError("每个 chirp 的采样点数必须为正整数")
        if self.chirps_per_frame <= 0:
            raise ValueError("每帧 chirp 数必须为正整数")
        if self.frame_period_ms <= 0:
            raise ValueError("帧周期必须为正数")
