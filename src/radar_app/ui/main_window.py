from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from radar_app.core import FORMAT_OPTIONS, WINDOW_OPTIONS, LoadedRadarData, RadarParams, load_radar_data
from radar_app.modules import angle, range_hrrp, range_precision, speed


class ToolTip:
    """鼠标悬停浮窗提示，400ms 延迟，浅黄半透明，智能定位"""

    _active_tip: ToolTip | None = None

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip_window: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<Button-1>", self._hide, add=True)
        widget.bind("<Destroy>", self._on_destroy, add=True)

    def _schedule(self, event: tk.Event | None = None) -> None:
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        if self._active_tip is not None and self._active_tip is not self:
            self._active_tip._hide()
        ToolTip._active_tip = self

        wx = self.widget.winfo_rootx()
        wy = self.widget.winfo_rooty()
        wh = self.widget.winfo_height()

        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_attributes("-topmost", True)

        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify=tk.LEFT,
            wraplength=420,
            font=("Microsoft YaHei", 9),
            bg="#ffffe0",
            fg="#333333",
            padx=8,
            pady=5,
        )
        label.pack()

        self.tip_window.update_idletasks()
        tw = self.tip_window.winfo_reqwidth()
        th = self.tip_window.winfo_reqheight()
        sw = self.widget.winfo_screenwidth()
        sh = self.widget.winfo_screenheight()

        x = wx
        y = wy + wh + 2
        if x + tw > sw:
            x = sw - tw - 4
        if y + th > sh:
            y = wy - th - 2
        self.tip_window.wm_geometry(f"+{max(0, x)}+{max(0, y)}")

    def _on_leave(self, event: tk.Event | None = None) -> None:
        self._hide()

    def _hide(self) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None
        if ToolTip._active_tip is self:
            ToolTip._active_tip = None

    def _on_destroy(self, event: tk.Event | None = None) -> None:
        self._hide()


class ResultPane:
    def __init__(self, parent: ttk.Frame, figsize: tuple[float, float] = (9.8, 5.8)) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        self.table = ttk.Treeview(parent, columns=("name", "value"), show="headings", height=7)
        self.table.heading("name", text="项目")
        self.table.heading("value", text="值")
        self.table.column("name", width=190, stretch=False)
        self.table.column("value", width=760, stretch=True)
        self.table.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))

        self.figure = Figure(figsize=figsize, dpi=100)
        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=1, column=0, sticky=tk.NSEW)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky=tk.EW)

    def set_rows(self, rows: list[tuple[str, object]]) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        for name, value in rows:
            self.table.insert("", tk.END, values=(name, self._format(value)))

    @staticmethod
    def _format(value: object) -> str:
        if isinstance(value, float):
            if np.isfinite(value):
                return f"{value:.6g}"
            return str(value)
        return str(value)


class RadarApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("雷达 bin 文件统一处理程序")
        self.root.geometry("1360x900")
        self.root.minsize(1100, 760)

        self.loaded: LoadedRadarData | None = None
        self.current_result: object | None = None
        self.current_module = ""
        self.current_rows: list[tuple[str, object]] = []
        self.calibration: angle.CalibrationResult | None = None

        self.file_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="请选择 bin 文件")
        self.start_freq_var = tk.StringVar(value="75")
        self.slope_var = tk.StringVar(value="30")
        self.sample_rate_var = tk.StringVar(value="3000")
        self.samples_var = tk.StringVar(value="256")
        self.chirps_var = tk.StringVar(value="8")
        self.frame_period_var = tk.StringVar(value="500")
        self.format_var = tk.StringVar(value="自动")
        self.header_var = tk.StringVar(value="自动")
        self.window_var = tk.StringVar(value="hamming")
        self.zero_padding_var = tk.StringVar(value="5")
        self.frame_var = tk.StringVar(value="1")
        self.antenna_var = tk.StringVar(value="1")
        self.chirp_var = tk.StringVar(value="1")

        self.hrrp_keep_var = tk.StringVar(value="1.0")
        self.hrrp_target_count_var = tk.StringVar(value="2")
        self.hrrp_min_range_var = tk.StringVar(value="0.15")
        self.hrrp_min_gap_var = tk.StringVar(value="0.15")

        self.precision_min_range_var = tk.StringVar(value="0.15")
        self.precision_noise_max_var = tk.StringVar(value="")
        self.precision_guard_bins_var = tk.StringVar(value="8")
        self.precision_full_window_var = tk.StringVar(value="hamming")
        self.precision_half_window_var = tk.StringVar(value="hamming")

        self.speed_min_range_var = tk.StringVar(value="0.1")
        self.speed_max_range_var = tk.StringVar(value="12.0")
        self.speed_max_jump_var = tk.StringVar(value="1.0")
        self.speed_margin_var = tk.StringVar(value="6.0")
        self.speed_smooth_var = tk.StringVar(value="3")
        self.speed_suppress_var = tk.BooleanVar(value=True)
        self.speed_prefer_approach_var = tk.BooleanVar(value=False)

        self.rd_frame_var = tk.StringVar(value="1")
        self.rd_min_range_var = tk.StringVar(value="0.1")
        self.rd_max_range_var = tk.StringVar(value="12.0")
        self.rd_chirp_period_var = tk.StringVar(value="100")
        self.rd_fft_var = tk.StringVar(value="128")
        self.rd_speed_limit_var = tk.StringVar(value="2.5")
        self.rd_slow_time_var = tk.StringVar(value="天线xChirp")
        self.rd_suppress_var = tk.BooleanVar(value=True)

        self.angle_channels_var = tk.StringVar(value=",".join(map(str, angle.DEFAULT_AZIMUTH_CHANNELS)))
        self.angle_spacing_mm_var = tk.StringVar(value="2.0")
        self.angle_fft_var = tk.StringVar(value="256")
        self.angle_min_range_var = tk.StringVar(value="0.15")
        self.angle_max_range_var = tk.StringVar(value="4.0")
        self.angle_gap_m_var = tk.StringVar(value="0.15")
        self.angle_gap_deg_var = tk.StringVar(value="5.0")
        self.angle_max_detections_var = tk.StringVar(value="1")
        self.angle_train_range_var = tk.StringVar(value="8")
        self.angle_guard_range_var = tk.StringVar(value="2")
        self.angle_train_angle_var = tk.StringVar(value="6")
        self.angle_guard_angle_var = tk.StringVar(value="2")
        self.angle_threshold_var = tk.StringVar(value="0.0")

        self._build_ui()
        self._draw_empty()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.rowconfigure(2, weight=1)
        outer.columnconfigure(0, weight=1)

        common = ttk.LabelFrame(outer, text="数据与公共参数")
        common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        for col in range(10):
            common.columnconfigure(col, weight=1 if col in {1, 3, 5, 7, 9} else 0)

        ttk.Label(common, text="bin 文件").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        _e = ttk.Entry(common, textvariable=self.file_var)
        _e.grid(row=0, column=1, columnspan=7, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_e, "待处理的雷达原始数据文件（.bin 格式）\n点击右侧“选择”按钮选取文件")
        _b = ttk.Button(common, text="选择", command=self.choose_file)
        _b.grid(row=0, column=8, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_b, "浏览并选择要处理的 .bin 文件")
        _b = ttk.Button(common, text="读取", command=self.load_file)
        _b.grid(row=0, column=9, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_b, "读取当前路径的 bin 文件并解析数据")

        fields = [
            ("起始频率(GHz)", self.start_freq_var, "FMCW 起始频率（75 GHz 典型值）\n对应波长 λ ≈ 4 mm"),
            ("斜率(MHz/us)", self.slope_var, "Chirp 调频斜率（30 MHz/μs 典型值）\n影响带宽和距离分辨率"),
            ("采样率(ksps)", self.sample_rate_var, "ADC 采样率（3000 ksps 典型值）\n决定最大无模糊距离"),
            ("采样点", self.samples_var, "每 Chirp 的采样点数（256 典型值）\n与距离分辨率直接相关"),
            ("Chirp/帧", self.chirps_var, "每帧包含的 Chirp 数量（8 典型值）\n影响多普勒处理能力"),
            ("帧周期(ms)", self.frame_period_var, "相邻两帧的时间间隔（500 ms 典型值）\n影响测速的时间分辨率"),
        ]
        for idx, (label, var, tip) in enumerate(fields):
            row = 1 + idx // 3
            col = (idx % 3) * 2
            ttk.Label(common, text=label).grid(row=row, column=col, sticky=tk.W, padx=6, pady=4)
            _e = ttk.Entry(common, textvariable=var, width=12)
            _e.grid(row=row, column=col + 1, sticky=tk.EW, padx=6, pady=4)
            ToolTip(_e, tip)

        ttk.Label(common, text="数据格式").grid(row=1, column=6, sticky=tk.W, padx=6, pady=4)
        self.format_box = ttk.Combobox(common, textvariable=self.format_var, values=FORMAT_OPTIONS, state="readonly", width=18)
        self.format_box.grid(row=1, column=7, sticky=tk.EW, padx=6, pady=4)
        ToolTip(self.format_box, "bin 文件中雷达数据的存储格式\n选择“自动”可智能识别")
        ttk.Label(common, text="文件头(bytes)").grid(row=1, column=8, sticky=tk.W, padx=6, pady=4)
        _e = ttk.Entry(common, textvariable=self.header_var, width=12)
        _e.grid(row=1, column=9, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_e, "文件头字节数\n选择“自动”可自动检测\n常见值：0、248")

        ttk.Label(common, text="窗函数").grid(row=2, column=6, sticky=tk.W, padx=6, pady=4)
        _c = ttk.Combobox(common, textvariable=self.window_var, values=WINDOW_OPTIONS, state="readonly", width=12)
        _c.grid(row=2, column=7, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_c, "FFT 前加窗类型\n• rect：矩形（无加窗）\n• hamming：汉明窗（默认）\n• hann：汉宁窗\n• blackman：布莱克曼窗\n• flattop：平顶窗")
        ttk.Label(common, text="补零幂").grid(row=2, column=8, sticky=tk.W, padx=6, pady=4)
        _e = ttk.Entry(common, textvariable=self.zero_padding_var, width=12)
        _e.grid(row=2, column=9, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_e, "FFT 补零倍数控制\nFFT 点数=2^(ceil(log2(N)) + 补零幂)\n补零幂=5 时 256 点→8192 点\n越大频谱越平滑，计算量越大")

        selectors = ttk.Frame(outer)
        selectors.grid(row=1, column=0, sticky=tk.EW, pady=(0, 8))
        selectors.columnconfigure(11, weight=1)
        ttk.Label(selectors, text="帧").grid(row=0, column=0, sticky=tk.W)
        self.frame_spin = ttk.Spinbox(selectors, textvariable=self.frame_var, from_=1, to=1, width=8)
        self.frame_spin.grid(row=0, column=1, padx=(4, 14))
        ToolTip(self.frame_spin, "选择要处理第几帧数据\n读取文件后根据实际帧数自动调整上限")
        ttk.Label(selectors, text="天线").grid(row=0, column=2, sticky=tk.W)
        self.antenna_box = ttk.Combobox(selectors, textvariable=self.antenna_var, values=["1"], state="readonly", width=10)
        self.antenna_box.grid(row=0, column=3, padx=(4, 14))
        ToolTip(self.antenna_box, "选择天线通道\n可选“平均”或指定某一天线")
        ttk.Label(selectors, text="Chirp").grid(row=0, column=4, sticky=tk.W)
        self.chirp_box = ttk.Combobox(selectors, textvariable=self.chirp_var, values=["1"], state="readonly", width=10)
        self.chirp_box.grid(row=0, column=5, padx=(4, 14))
        ToolTip(self.chirp_box, "选择 Chirp 序号\n可选“平均”或指定某个 Chirp")
        _b = ttk.Button(selectors, text="处理当前模块", command=self.process_current_tab)
        _b.grid(row=0, column=6, padx=(0, 8))
        ToolTip(_b, "使用当前参数运行选中标签页的算法")
        _b = ttk.Button(selectors, text="保存当前结果", command=self.save_current_result)
        _b.grid(row=0, column=7, padx=(0, 8))
        ToolTip(_b, "将当前模块的结果导出为 PNG 图片和 CSV 数据")
        _b = ttk.Button(selectors, text="设为测角校准", command=self.build_angle_calibration)
        _b.grid(row=0, column=8, padx=(0, 8))
        ToolTip(_b, "将当前测角结果保存为 0° 相位校准数据\n后续测角时自动补偿通道间相位差")
        ttk.Label(selectors, textvariable=self.status_var).grid(row=0, column=9, columnspan=3, sticky=tk.EW)

        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=2, column=0, sticky=tk.NSEW)
        self.panes: dict[str, ResultPane] = {}
        self.tab_keys: dict[str, str] = {}

        self._add_hrrp_tab()
        self._add_precision_tab()
        self._add_speed_tab()
        self._add_range_doppler_tab()
        self._add_angle_tab()

    def _make_tab(self, key: str, title: str) -> tuple[ttk.Frame, ttk.Frame, ResultPane]:
        tab = ttk.Frame(self.tabs, padding=8)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.tabs.add(tab, text=title)
        self.tab_keys[str(tab)] = key

        controls = ttk.LabelFrame(tab, text="模块参数")
        controls.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        for col in range(12):
            controls.columnconfigure(col, weight=1 if col % 2 else 0)
        body = ttk.Frame(tab)
        body.grid(row=1, column=0, sticky=tk.NSEW)
        pane = ResultPane(body)
        self.panes[key] = pane
        return tab, controls, pane

    def _field(self, parent: ttk.Frame, row: int, col: int, label: str, var: tk.Variable, width: int = 10, tip: str | None = None) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=6, pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=col + 1, sticky=tk.EW, padx=6, pady=4)
        if tip:
            ToolTip(entry, tip)
        return entry

    def _add_hrrp_tab(self) -> None:
        _tab, controls, _pane = self._make_tab("hrrp", "测距 / HRRP")
        self._field(controls, 0, 0, "截取比例", self.hrrp_keep_var,
                    tip="截取 chirp 前段的比例（0~1）\n1.0 = 使用全部采样点\n0.5 = 只使用前一半")
        self._field(controls, 0, 2, "目标数", self.hrrp_target_count_var,
                    tip="最多检测的目标数量\n按幅度从大到小排序")
        self._field(controls, 0, 4, "忽略近端(m)", self.hrrp_min_range_var,
                    tip="忽略该距离以内的峰值\n用于消除近场强杂波的干扰")
        self._field(controls, 0, 6, "目标间隔(m)", self.hrrp_min_gap_var,
                    tip="两个不同目标之间的最小距离间隔\n小于此间隔的峰值视为同一目标")

    def _add_precision_tab(self) -> None:
        _tab, controls, _pane = self._make_tab("precision", "测距精度")
        self._field(controls, 0, 0, "忽略近端(m)", self.precision_min_range_var,
                    tip="目标搜索起始距离\n忽略该距离以内的峰值")
        self._field(controls, 0, 2, "噪声最大距离(m)", self.precision_noise_max_var,
                    tip="噪声功率估计的距离上限\n留空自动使用最大距离")
        self._field(controls, 0, 4, "保护单元", self.precision_guard_bins_var,
                    tip="目标 3dB 主瓣外的保护单元数\n防止目标能量泄漏到噪声估计中")
        ttk.Label(controls, text="全带宽窗").grid(row=0, column=6, sticky=tk.W, padx=6, pady=4)
        _c = ttk.Combobox(controls, textvariable=self.precision_full_window_var, values=WINDOW_OPTIONS, state="readonly", width=10)
        _c.grid(row=0, column=7, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_c, "全带宽配置（keep_ratio=1.0）使用的窗函数")
        ttk.Label(controls, text="半带宽窗").grid(row=0, column=8, sticky=tk.W, padx=6, pady=4)
        _c = ttk.Combobox(controls, textvariable=self.precision_half_window_var, values=WINDOW_OPTIONS, state="readonly", width=10)
        _c.grid(row=0, column=9, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_c, "半带宽配置（keep_ratio=0.5）使用的窗函数\n与全带宽对比展示带宽对精度的影响")

    def _add_speed_tab(self) -> None:
        _tab, controls, _pane = self._make_tab("speed", "测速")
        self._field(controls, 0, 0, "最小距离(m)", self.speed_min_range_var,
                    tip="目标搜索的起始距离\n排除近场杂波")
        self._field(controls, 0, 2, "最大距离(m)", self.speed_max_range_var,
                    tip="目标搜索的最大距离\n应覆盖目标运动范围")
        self._field(controls, 0, 4, "最大跳变(m/帧)", self.speed_max_jump_var,
                    tip="相邻帧间目标最大移动距离\n用于约束轨迹跟踪的搜索范围")
        self._field(controls, 0, 6, "置信阈值(dB)", self.speed_margin_var,
                    tip="峰值相对于噪声基底的最小信噪比\n低于此阈值的帧将被标记为无效")
        self._field(controls, 0, 8, "平滑帧数", self.speed_smooth_var,
                    tip="轨迹平滑的滑动窗口大小\n越大曲线越平滑，但响应越迟钝")
        _ck = ttk.Checkbutton(controls, text="静态抑制", variable=self.speed_suppress_var)
        _ck.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=6)
        ToolTip(_ck, "启用后减去所有帧的均值\n可抑制静止物体的杂波\n运动目标会保留")
        _ck = ttk.Checkbutton(controls, text="优先接近目标", variable=self.speed_prefer_approach_var)
        _ck.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=6)
        ToolTip(_ck, "启用后在轨迹跟踪中偏向距离减小的方向\n适合追踪正在靠近的目标")

    def _add_range_doppler_tab(self) -> None:
        _tab, controls, _pane = self._make_tab("range_doppler", "距离-多普勒")
        self._field(controls, 0, 0, "RD 帧", self.rd_frame_var,
                    tip="选取第几帧数据做距离-多普勒分析")
        self._field(controls, 0, 2, "最小距离(m)", self.rd_min_range_var,
                    tip="距离-多普勒图显示的最小距离")
        self._field(controls, 0, 4, "最大距离(m)", self.rd_max_range_var,
                    tip="距离-多普勒图显示的最大距离")
        self._field(controls, 0, 6, "Chirp 周期(us)", self.rd_chirp_period_var,
                    tip="单个 Chirp 的持续时间（微秒）\n用于计算多普勒频率和速度")
        self._field(controls, 0, 8, "Doppler FFT", self.rd_fft_var,
                    tip="多普勒维 FFT 点数\n点数越大速度分辨率越高")
        self._field(controls, 1, 0, "速度范围(m/s)", self.rd_speed_limit_var,
                    tip="速度显示范围 ±X m/s\n限制显示范围可突出低速目标")
        ttk.Label(controls, text="慢时间").grid(row=1, column=2, sticky=tk.W, padx=6, pady=4)
        _c = ttk.Combobox(controls, textvariable=self.rd_slow_time_var, values=speed.SLOW_TIME_OPTIONS, state="readonly", width=14)
        _c.grid(row=1, column=3, sticky=tk.EW, padx=6, pady=4)
        ToolTip(_c, "慢时间维度的组织方式\n• 天线xChirp：将天线和 chirp 都作为慢时间\n• Chirp：只将 chirp 作为慢时间")
        _ck = ttk.Checkbutton(controls, text="静态抑制", variable=self.rd_suppress_var)
        _ck.grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=6)
        ToolTip(_ck, "启用后减去慢时间均值\n可抑制静止物体的多普勒信号")

    def _add_angle_tab(self) -> None:
        _tab, controls, _pane = self._make_tab("angle", "测角")
        self._field(controls, 0, 0, "虚拟通道", self.angle_channels_var, width=22,
                    tip="8 个虚拟通道编号，用逗号分隔\n例：1,2,3,4,5,6,7,8")
        self._field(controls, 0, 2, "阵元间距(mm)", self.angle_spacing_mm_var,
                    tip="相邻虚拟天线阵元的物理间距（毫米）\n影响角度测量范围和分辨率")
        self._field(controls, 0, 4, "方位 FFT", self.angle_fft_var,
                    tip="方位角 FFT 点数\n越大角度分辨率越高")
        self._field(controls, 0, 6, "忽略近端(m)", self.angle_min_range_var,
                    tip="角度测量忽略该距离以内的目标\n用于排除近场干扰")
        self._field(controls, 0, 8, "显示最大距离(m)", self.angle_max_range_var,
                    tip="距离-角度图的最大显示距离")
        self._field(controls, 1, 0, "CFAR 距离间隔(m)", self.angle_gap_m_var,
                    tip="CFAR 多目标检测中\n两个目标间的最小距离间隔")
        self._field(controls, 1, 2, "CFAR 角度间隔(度)", self.angle_gap_deg_var,
                    tip="CFAR 多目标检测中\n两个目标间的最小角度间隔")
        self._field(controls, 1, 4, "最大目标数", self.angle_max_detections_var,
                    tip="CFAR 最多检测的目标数量")
        self._field(controls, 1, 6, "训练距/保护距", self.angle_train_range_var,
                    tip="CFAR 距离维训练单元数\n用于估计局部噪声功率")
        self._field(controls, 1, 8, "训练角/保护角", self.angle_train_angle_var,
                    tip="CFAR 角度维训练单元数")
        self._field(controls, 2, 0, "保护距", self.angle_guard_range_var,
                    tip="CFAR 距离维保护单元数\n防止目标能量泄漏到噪声估计中")
        self._field(controls, 2, 2, "保护角", self.angle_guard_angle_var,
                    tip="CFAR 角度维保护单元数")
        self._field(controls, 2, 4, "阈值(dB)", self.angle_threshold_var,
                    tip="CFAR 检测门限（dB）\n越低检测灵敏度越高，虚警也越多\n0 表示自适应门限")

    def _draw_empty(self) -> None:
        for pane in self.panes.values():
            pane.figure.clear()
            ax = pane.figure.add_subplot(111)
            ax.text(0.5, 0.5, "选择 bin 文件后点击处理", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            pane.canvas.draw_idle()

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(title="选择 bin 文件", filetypes=[("bin files", "*.bin"), ("all files", "*.*")])
        if path:
            self.file_var.set(path)
            self.load_file()

    def _params(self) -> RadarParams:
        params = RadarParams(
            start_freq_ghz=float(self.start_freq_var.get()),
            slope_mhz_us=float(self.slope_var.get()),
            sample_rate_ksps=float(self.sample_rate_var.get()),
            samples_per_chirp=int(float(self.samples_var.get())),
            chirps_per_frame=int(float(self.chirps_var.get())),
            frame_period_ms=float(self.frame_period_var.get()),
        )
        params.validate()
        return params

    def load_file(self) -> None:
        try:
            path = Path(self.file_var.get())
            params = self._params()
            loaded = load_radar_data(path, params, self.format_var.get(), self.header_var.get())
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            self.status_var.set(f"读取失败: {exc}")
            return

        self.loaded = loaded
        self.current_result = None
        self.current_module = ""
        self.current_rows = []
        self._update_selectors()
        self.status_var.set(
            f"已读取 {path.name}: {loaded.frame_count} 帧, {loaded.antenna_count} 通道, "
            f"{loaded.format_label}, 文件头 {loaded.header_bytes} B, 尾部丢弃 {loaded.dropped_tail_bytes} B"
        )

    def _update_selectors(self) -> None:
        if self.loaded is None:
            return
        frame_count = self.loaded.frame_count
        self.frame_spin.configure(to=max(1, frame_count))
        self.frame_var.set("1")
        self.rd_frame_var.set("1")
        antenna_values = ["平均"] + [str(i) for i in range(1, self.loaded.antenna_count + 1)]
        if self.loaded.antenna_count == 1:
            antenna_values = ["1"]
        self.antenna_box.configure(values=antenna_values)
        self.antenna_var.set("1" if "1" in antenna_values else antenna_values[0])
        chirp_count = int(self.chirps_var.get())
        chirp_values = ["平均"] + [str(i) for i in range(1, chirp_count + 1)]
        self.chirp_box.configure(values=chirp_values)
        self.chirp_var.set("1")
        if self.loaded.antenna_count >= 16:
            self.angle_channels_var.set(",".join(map(str, angle.DEFAULT_AZIMUTH_CHANNELS)))
        elif self.loaded.antenna_count >= 8:
            self.angle_channels_var.set(",".join(str(i) for i in range(1, 9)))

    def _ensure_loaded(self) -> LoadedRadarData:
        if self.loaded is None:
            self.load_file()
        if self.loaded is None:
            raise RuntimeError("请先读取 bin 文件")
        return self.loaded

    def _selected_key(self) -> str:
        return self.tab_keys[self.tabs.select()]

    def process_current_tab(self) -> None:
        key = self._selected_key()
        try:
            if key == "hrrp":
                self._process_hrrp()
            elif key == "precision":
                self._process_precision()
            elif key == "speed":
                self._process_speed()
            elif key == "range_doppler":
                self._process_range_doppler()
            elif key == "angle":
                self._process_angle()
        except Exception as exc:
            messagebox.showerror("处理失败", str(exc))
            self.status_var.set(f"处理失败: {exc}")

    def _antenna_number(self) -> int:
        text = self.antenna_var.get().strip()
        if text in {"", "平均"}:
            return 1
        return int(float(text))

    def _process_hrrp(self) -> None:
        loaded = self._ensure_loaded()
        params = self._params()
        result = range_hrrp.analyze_hrrp(
            loaded.data,
            params,
            frame_number=int(float(self.frame_var.get())),
            antenna_number=self._antenna_number(),
            window_name=self.window_var.get(),
            chirp_choice=self.chirp_var.get(),
            keep_ratio=float(self.hrrp_keep_var.get()),
            target_count=int(float(self.hrrp_target_count_var.get())),
            min_range_m=float(self.hrrp_min_range_var.get()),
            min_gap_m=float(self.hrrp_min_gap_var.get()),
            zero_padding_power=int(float(self.zero_padding_var.get())),
        )
        rows = [
            ("FFT 点数", result.nfft),
            ("截取点数", result.n_keep),
            ("检测目标数", len(result.targets)),
        ]
        for idx, ((target_range, value, _peak), width) in enumerate(zip(result.targets, result.widths), start=1):
            rows.extend(
                [
                    (f"目标 {idx} 距离(m)", target_range),
                    (f"目标 {idx} 幅度", value),
                    (f"目标 {idx} 3dB 宽度(m)", width[2]),
                ]
            )
        pane = self.panes["hrrp"]
        pane.set_rows(rows)
        self._plot_hrrp(pane.figure, result)
        pane.canvas.draw_idle()
        self._set_current("hrrp", result, rows)

    def _plot_hrrp(self, figure: Figure, result: range_hrrp.HrrpAnalysis) -> None:
        figure.clear()
        ax = figure.add_subplot(111)
        ax.plot(result.ranges_m, result.hrrp_db, linewidth=1.2, color="#1565c0")
        for idx, ((target_range, _value, peak_idx), width) in enumerate(zip(result.targets, result.widths), start=1):
            ax.axvline(target_range, color="#b3261e", linestyle="--", linewidth=1.0)
            ax.plot(target_range, result.hrrp_db[peak_idx], "v", color="#b3261e")
            if np.isfinite(width[2]):
                ax.axvspan(width[0], width[1], color="#ef6c00", alpha=0.14)
            ax.annotate(f"T{idx}: {target_range:.3f} m", (target_range, result.hrrp_db[peak_idx]), xytext=(5, 8), textcoords="offset points")
        ax.set_title("测距 / HRRP")
        ax.set_xlabel("距离 (m)")
        ax.set_ylabel("归一化幅度 (dB)")
        ax.set_ylim(range_hrrp.HRRP_DB_FLOOR, 4)
        ax.grid(True, linestyle="--", alpha=0.35)
        figure.tight_layout()

    def _process_precision(self) -> None:
        loaded = self._ensure_loaded()
        params = self._params()
        noise_text = self.precision_noise_max_var.get().strip()
        results = range_precision.analyze_precision(
            loaded.data,
            params,
            frame_number=int(float(self.frame_var.get())),
            antenna_number=self._antenna_number(),
            chirp_choice=self.chirp_var.get(),
            full_window=self.precision_full_window_var.get(),
            half_window=self.precision_half_window_var.get(),
            zero_padding_power=int(float(self.zero_padding_var.get())),
            min_range_m=float(self.precision_min_range_var.get()),
            noise_max_range_m=float(noise_text) if noise_text else None,
            min_guard_bins=int(float(self.precision_guard_bins_var.get())),
        )
        rows: list[tuple[str, object]] = []
        for result in results:
            rows.extend(
                [
                    (f"{result.case.label} 目标距离(m)", result.target_range_m),
                    (f"{result.case.label} 带宽(GHz)", result.hrrp.bandwidth_hz / 1e9),
                    (f"{result.case.label} 距离分辨率(m)", result.hrrp.range_resolution_m),
                    (f"{result.case.label} SNR(dB)", result.snr_db),
                    (f"{result.case.label} 测距精度(m)", result.precision_m),
                ]
            )
        pane = self.panes["precision"]
        pane.set_rows(rows)
        self._plot_precision(pane.figure, results)
        pane.canvas.draw_idle()
        self._set_current("precision", results, rows)

    def _plot_precision(self, figure: Figure, results: list[range_precision.PrecisionResult]) -> None:
        figure.clear()
        axes = figure.subplots(len(results), 1, squeeze=False)
        for ax, result in zip(axes.ravel(), results):
            hrrp = result.hrrp
            ax.plot(hrrp.ranges_m, hrrp.hrrp_db, color="#1565c0", linewidth=1.1)
            ax.axvspan(result.target_left_m, result.target_right_m, color="#ef6c00", alpha=0.16)
            ax.axvline(result.target_range_m, color="#b3261e", linestyle="--", linewidth=1.0)
            ax.plot(result.target_range_m, hrrp.hrrp_db[result.peak_index], "v", color="#b3261e")
            ax.set_title(f"{result.case.label}: R={result.target_range_m:.3f} m, SNR={result.snr_db:.2f} dB")
            ax.set_xlabel("距离 (m)")
            ax.set_ylabel("幅度 (dB)")
            ax.set_ylim(range_precision.HRRP_DB_FLOOR, 4)
            ax.grid(True, linestyle="--", alpha=0.35)
        figure.tight_layout()

    def _process_speed(self) -> None:
        loaded = self._ensure_loaded()
        params = self._params()
        result = speed.analyze_speed(
            loaded.data,
            params,
            antenna_choice=self.antenna_var.get(),
            chirp_choice=self.chirp_var.get(),
            window_name=self.window_var.get(),
            zero_padding_power=int(float(self.zero_padding_var.get())),
            suppress_static=self.speed_suppress_var.get(),
            min_range_m=float(self.speed_min_range_var.get()),
            max_range_m=float(self.speed_max_range_var.get()),
            max_jump_m=float(self.speed_max_jump_var.get()),
            min_margin_db=float(self.speed_margin_var.get()),
            prefer_approaching=self.speed_prefer_approach_var.get(),
            smooth_frames=int(float(self.speed_smooth_var.get())),
        )
        rows = [
            ("有效帧数", f"{result.valid_frame_count} / {len(result.times_s)}"),
            ("起始距离(m)", result.start_range_m),
            ("结束距离(m)", result.end_range_m),
            ("平均速度(m/s)", result.average_speed_mps),
            ("拟合速度(m/s)", result.fitted_speed_mps),
            ("中位速度(m/s)", result.median_speed_mps),
            ("速度范围(m/s)", f"{result.min_speed_mps:.6g} ~ {result.max_speed_mps:.6g}"),
        ]
        pane = self.panes["speed"]
        pane.set_rows(rows)
        self._plot_speed(pane.figure, result)
        pane.canvas.draw_idle()
        self._set_current("speed", result, rows)

    def _plot_speed(self, figure: Figure, result: speed.SpeedAnalysisResult) -> None:
        figure.clear()
        ax1, ax2 = figure.subplots(2, 1)
        extent = [
            float(result.display_ranges_m[0]),
            float(result.display_ranges_m[-1]),
            float(result.times_s[-1]) if len(result.times_s) else 0.0,
            0.0,
        ]
        ax1.imshow(result.heatmap_db, aspect="auto", extent=extent, cmap="viridis", vmin=speed.HEATMAP_DB_FLOOR, vmax=0)
        ax1.plot(result.ranges_m, result.times_s, color="white", linewidth=0.9, alpha=0.75, label="跟踪距离")
        ax1.plot(result.smoothed_ranges_m, result.times_s, color="#ffca28", linewidth=1.2, label="平滑轨迹")
        ax1.set_title("距离-时间热图与目标轨迹")
        ax1.set_xlabel("距离 (m)")
        ax1.set_ylabel("时间 (s)")
        ax1.legend(loc="best")
        ax2.plot(result.times_s, result.speeds_mps, color="#1565c0", linewidth=1.2)
        ax2.axhline(result.fitted_speed_mps, color="#b3261e", linestyle="--", linewidth=1.0, label="拟合速度")
        ax2.set_title("速度估计")
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("速度 (m/s)")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="best")
        figure.tight_layout()

    def _process_range_doppler(self) -> None:
        loaded = self._ensure_loaded()
        params = self._params()
        result = speed.compute_range_doppler(
            loaded.data,
            params,
            frame_index=int(float(self.rd_frame_var.get())) - 1,
            antenna_choice=self.antenna_var.get(),
            window_name=self.window_var.get(),
            zero_padding_power=int(float(self.zero_padding_var.get())),
            chirp_period_us=float(self.rd_chirp_period_var.get()),
            doppler_fft_size=int(float(self.rd_fft_var.get())),
            speed_limit_mps=float(self.rd_speed_limit_var.get()),
            min_range_m=float(self.rd_min_range_var.get()),
            max_range_m=float(self.rd_max_range_var.get()),
            suppress_static=self.rd_suppress_var.get(),
            slow_time_mode=self.rd_slow_time_var.get(),
        )
        rows = [
            ("帧号", result.frame_number),
            ("峰值距离(m)", result.max_range_m),
            ("峰值速度(m/s)", result.max_velocity_mps),
            ("速度分辨率(m/s)", result.velocity_resolution_mps),
            ("最大无模糊速度(m/s)", result.max_unambiguous_velocity_mps),
            ("峰值绝对功率(dB)", result.max_db),
        ]
        pane = self.panes["range_doppler"]
        pane.set_rows(rows)
        self._plot_range_doppler(pane.figure, result)
        pane.canvas.draw_idle()
        self._set_current("range_doppler", result, rows)

    def _plot_range_doppler(self, figure: Figure, result: speed.RangeDopplerResult) -> None:
        figure.clear()
        ax = figure.add_subplot(111)
        extent = [
            float(result.velocities_mps[0]),
            float(result.velocities_mps[-1]),
            float(result.ranges_m[0]),
            float(result.ranges_m[-1]),
        ]
        image = ax.imshow(result.map_db, aspect="auto", origin="lower", extent=extent, cmap="turbo", vmin=speed.RANGE_DOPPLER_DB_FLOOR, vmax=0)
        ax.plot(result.max_velocity_mps, result.max_range_m, "x", color="white", markersize=8)
        ax.set_title("距离-多普勒图")
        ax.set_xlabel("速度 (m/s)")
        ax.set_ylabel("距离 (m)")
        figure.colorbar(image, ax=ax, label="相对功率 (dB)")
        figure.tight_layout()

    def _analyze_current_angle(self, phase_correction: np.ndarray | None) -> tuple[angle.SingleTargetResult, angle.MultiTargetResult, np.ndarray]:
        loaded = self._ensure_loaded()
        params = self._params()
        return angle.analyze_angle(
            loaded.data,
            params,
            frame_number=int(float(self.frame_var.get())),
            virtual_channels_text=self.angle_channels_var.get(),
            window_name=self.window_var.get(),
            chirp_choice=self.chirp_var.get(),
            zero_padding_power=int(float(self.zero_padding_var.get())),
            min_range_m=float(self.angle_min_range_var.get()),
            max_display_range_m=float(self.angle_max_range_var.get()),
            antenna_spacing_m=float(self.angle_spacing_mm_var.get()) / 1000.0,
            angle_fft_size=int(float(self.angle_fft_var.get())),
            cfar_train_range=int(float(self.angle_train_range_var.get())),
            cfar_guard_range=int(float(self.angle_guard_range_var.get())),
            cfar_train_angle=int(float(self.angle_train_angle_var.get())),
            cfar_guard_angle=int(float(self.angle_guard_angle_var.get())),
            cfar_threshold_db=float(self.angle_threshold_var.get()),
            max_detections=int(float(self.angle_max_detections_var.get())),
            min_gap_m=float(self.angle_gap_m_var.get()),
            min_gap_deg=float(self.angle_gap_deg_var.get()),
            phase_correction=phase_correction,
        )

    def _process_angle(self) -> None:
        single, multi, range_spectrum = self._analyze_current_angle(
            self.calibration.phase_correction if self.calibration else None
        )
        rows = [
            ("单目标距离(m)", single.target_range_m),
            ("相位差均值(deg)", single.mean_delta_deg),
            ("相位法角度(deg)", single.phase_angle_deg),
            ("FFT 角度(deg)", single.angle_fft_angle_deg),
            ("检测目标数", len(multi.detections)),
            ("虚拟通道", ",".join(map(str, single.virtual_channels))),
        ]
        for detection in multi.detections:
            rows.append((f"CFAR {detection.serial}", f"R={detection.range_m:.4f} m, angle={detection.angle_deg:.3f} deg, {detection.relative_power_db:.2f} dB"))
        pane = self.panes["angle"]
        pane.set_rows(rows)
        self._plot_angle(pane.figure, single, multi)
        pane.canvas.draw_idle()
        self._set_current("angle", (single, multi, range_spectrum), rows)

    def _plot_angle(self, figure: Figure, single: angle.SingleTargetResult, multi: angle.MultiTargetResult) -> None:
        figure.clear()
        axes = figure.subplots(2, 2)
        ax_time, ax_hrrp, ax_angle, ax_map = axes.ravel()
        ax_time.plot(single.time_us, np.real(single.time_signal), color="#1565c0", linewidth=1.0)
        if np.iscomplexobj(single.time_signal):
            ax_time.plot(single.time_us, np.imag(single.time_signal), color="#ef6c00", linewidth=1.0, alpha=0.8)
        ax_time.set_title("时域信号")
        ax_time.set_xlabel("时间 (us)")
        ax_time.grid(True, linestyle="--", alpha=0.3)

        ax_hrrp.plot(single.ranges_m, single.hrrp_db, color="#1565c0", linewidth=1.1)
        ax_hrrp.axvline(single.target_range_m, color="#b3261e", linestyle="--", linewidth=1.0)
        ax_hrrp.set_xlim(0, single.max_display_range_m)
        ax_hrrp.set_ylim(angle.HRRP_DB_FLOOR, 4)
        ax_hrrp.set_title("测角 HRRP")
        ax_hrrp.set_xlabel("距离 (m)")
        ax_hrrp.grid(True, linestyle="--", alpha=0.3)
        ax_hrrp.text(0.97, 0.95, f"R={single.target_range_m:.4f} m",
                     transform=ax_hrrp.transAxes, ha="right", va="top",
                     fontsize=10, color="#b3261e",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b3261e", lw=0.8, alpha=0.9))

        ax_angle.plot(single.angle_axis_deg, single.angle_spectrum_db, color="#1565c0", linewidth=1.1)
        ax_angle.axvline(single.angle_fft_angle_deg, color="#b3261e", linestyle="--", linewidth=1.0)
        ax_angle.set_title("角谱")
        ax_angle.set_xlabel("角度 (deg)")
        ax_angle.set_ylabel("相对功率 (dB)")
        ax_angle.grid(True, linestyle="--", alpha=0.3)
        ax_angle.text(0.97, 0.95, f"θ={single.angle_fft_angle_deg:.3f}°",
                      transform=ax_angle.transAxes, ha="right", va="top",
                      fontsize=10, color="#b3261e",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b3261e", lw=0.8, alpha=0.9))
        phase_angle_display = f"相位法θ={single.phase_angle_deg:.3f}°" if np.isfinite(single.phase_angle_deg) else ""
        if phase_angle_display:
            ax_angle.text(0.97, 0.86, phase_angle_display,
                          transform=ax_angle.transAxes, ha="right", va="top",
                          fontsize=9, color="#1565c0",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565c0", lw=0.8, alpha=0.9))

        extent = [
            float(multi.ranges_m[0]),
            float(multi.ranges_m[-1]),
            float(multi.angle_axis_deg[0]),
            float(multi.angle_axis_deg[-1]),
        ]
        image = ax_map.imshow(multi.range_angle_db, aspect="auto", origin="lower", extent=extent, cmap="turbo", vmin=angle.RANGE_ANGLE_DB_FLOOR, vmax=0)
        for detection in multi.detections:
            ax_map.plot(detection.range_m, detection.angle_deg, "x", color="white", markersize=7)
            ax_map.annotate(f"R={detection.range_m:.3f}m\nθ={detection.angle_deg:.2f}°",
                            (detection.range_m, detection.angle_deg),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=8, color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#333", ec="white", lw=0.6, alpha=0.8))
        ax_map.set_title("距离-角度图")
        ax_map.set_xlabel("距离 (m)")
        ax_map.set_ylabel("角度 (deg)")
        figure.colorbar(image, ax=ax_map, label="相对功率 (dB)")
        figure.tight_layout()
        figure.canvas.mpl_connect("button_press_event", self._on_angle_popup)

    def _on_angle_popup(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """点击测角子图弹出独立放大窗口"""
        if event.inaxes is None or self.current_result is None:
            return
        if not isinstance(self.current_result, tuple) or len(self.current_result) < 2:
            return
        single, multi, _ = self.current_result
        title = event.inaxes.get_title()

        popup = tk.Toplevel(self.root)
        popup.title(f"测角 — {title}")
        popup.geometry("900x650")
        popup.minsize(500, 380)

        fig = Figure(figsize=(9, 5.8), dpi=100)
        ax = fig.add_subplot(111)

        if title == "时域信号":
            ax.plot(single.time_us, np.real(single.time_signal), color="#1565c0", linewidth=1.0)
            if np.iscomplexobj(single.time_signal):
                ax.plot(single.time_us, np.imag(single.time_signal), color="#ef6c00", linewidth=1.0, alpha=0.8)
            ax.set_title("时域信号")
            ax.set_xlabel("时间 (us)")
            ax.grid(True, linestyle="--", alpha=0.3)
        elif title == "测角 HRRP":
            ax.plot(single.ranges_m, single.hrrp_db, color="#1565c0", linewidth=1.1)
            ax.axvline(single.target_range_m, color="#b3261e", linestyle="--", linewidth=1.0)
            ax.set_xlim(0, single.max_display_range_m)
            ax.set_ylim(angle.HRRP_DB_FLOOR, 4)
            ax.set_title("测角 HRRP")
            ax.set_xlabel("距离 (m)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.97, 0.95, f"R={single.target_range_m:.4f} m",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color="#b3261e",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b3261e", lw=0.8, alpha=0.9))
        elif title == "角谱":
            ax.plot(single.angle_axis_deg, single.angle_spectrum_db, color="#1565c0", linewidth=1.1)
            ax.axvline(single.angle_fft_angle_deg, color="#b3261e", linestyle="--", linewidth=1.0)
            ax.set_title("角谱")
            ax.set_xlabel("角度 (deg)")
            ax.set_ylabel("相对功率 (dB)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.97, 0.95, f"FFT θ={single.angle_fft_angle_deg:.3f}°",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color="#b3261e",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b3261e", lw=0.8, alpha=0.9))
            if np.isfinite(single.phase_angle_deg):
                ax.text(0.97, 0.85, f"相位法θ={single.phase_angle_deg:.3f}°",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=10, color="#1565c0",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565c0", lw=0.8, alpha=0.9))
        elif title == "距离-角度图":
            extent = [
                float(multi.ranges_m[0]), float(multi.ranges_m[-1]),
                float(multi.angle_axis_deg[0]), float(multi.angle_axis_deg[-1]),
            ]
            image = ax.imshow(multi.range_angle_db, aspect="auto", origin="lower",
                              extent=extent, cmap="turbo",
                              vmin=angle.RANGE_ANGLE_DB_FLOOR, vmax=0)
            for detection in multi.detections:
                ax.plot(detection.range_m, detection.angle_deg, "x", color="white", markersize=7)
                ax.annotate(f"R={detection.range_m:.3f}m\nθ={detection.angle_deg:.2f}°",
                            (detection.range_m, detection.angle_deg),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=9, color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#333", ec="white", lw=0.6, alpha=0.8))
            ax.set_title("距离-角度图")
            ax.set_xlabel("距离 (m)")
            ax.set_ylabel("角度 (deg)")
            fig.colorbar(image, ax=ax, label="相对功率 (dB)")
        else:
            popup.destroy()
            return

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, popup, pack_toolbar=True).update()

    def build_angle_calibration(self) -> None:
        if self.loaded is None:
            messagebox.showinfo("校准", "请先读取一个 0 度校准文件")
            return
        try:
            single, _multi, range_spectrum = self._analyze_current_angle(None)
        except Exception as exc:
            messagebox.showerror("校准失败", str(exc))
            self.status_var.set(f"校准失败: {exc}")
            return
        self.calibration = angle.build_phase_calibration(single, range_spectrum)
        self._process_angle()
        self.status_var.set(f"已重新建立并应用测角校准: R={self.calibration.target_range_m:.4f} m")

    def _set_current(self, module: str, result: object, rows: list[tuple[str, object]]) -> None:
        self.current_module = module
        self.current_result = result
        self.current_rows = rows
        self.status_var.set(f"{self.tabs.tab(self.tabs.select(), 'text')} 处理完成")

    def save_current_result(self) -> None:
        if not self.current_module or self.current_result is None:
            messagebox.showinfo("保存", "请先处理一个模块")
            return
        out_dir = filedialog.askdirectory(title="选择保存目录")
        if not out_dir:
            return
        out_path = Path(out_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self.current_module}_{timestamp}"
        pane = self.panes[self.current_module]
        figure_path = out_path / f"{stem}.png"
        summary_path = out_path / f"{stem}_summary.csv"
        params_path = out_path / f"{stem}_params.txt"

        pane.figure.savefig(figure_path, dpi=180)
        self._write_summary(summary_path)
        self._write_detail_csv(out_path / f"{stem}_data.csv")
        self._write_params(params_path)
        self.status_var.set(f"已保存: {figure_path.name}, {summary_path.name}, {params_path.name}")

    def _write_summary(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["项目", "值"])
            for row in self.current_rows:
                writer.writerow(row)

    def _write_params(self, path: Path) -> None:
        lines = [
            f"module={self.current_module}",
            f"file={self.file_var.get()}",
            f"format={self.format_var.get()}",
            f"header={self.header_var.get()}",
            f"start_freq_ghz={self.start_freq_var.get()}",
            f"slope_mhz_us={self.slope_var.get()}",
            f"sample_rate_ksps={self.sample_rate_var.get()}",
            f"samples_per_chirp={self.samples_var.get()}",
            f"chirps_per_frame={self.chirps_var.get()}",
            f"frame_period_ms={self.frame_period_var.get()}",
            f"frame={self.frame_var.get()}",
            f"antenna={self.antenna_var.get()}",
            f"chirp={self.chirp_var.get()}",
            f"window={self.window_var.get()}",
            f"zero_padding_power={self.zero_padding_var.get()}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_detail_csv(self, path: Path) -> None:
        result = self.current_result
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if self.current_module == "hrrp" and isinstance(result, range_hrrp.HrrpAnalysis):
                writer.writerow(["range_m", "magnitude_norm", "hrrp_db"])
                writer.writerows(zip(result.ranges_m, result.magnitude_norm, result.hrrp_db))
            elif self.current_module == "precision" and isinstance(result, list):
                writer.writerow(["case", "range_m", "hrrp_db", "noise_mask"])
                for item in result:
                    for range_m, hrrp_db, noise in zip(item.hrrp.ranges_m, item.hrrp.hrrp_db, item.noise_mask):
                        writer.writerow([item.case.label, range_m, hrrp_db, int(noise)])
            elif self.current_module == "speed" and isinstance(result, speed.SpeedAnalysisResult):
                writer.writerow(["time_s", "tracked_range_m", "smoothed_range_m", "speed_mps", "valid", "confidence_db"])
                writer.writerows(
                    zip(
                        result.times_s,
                        result.ranges_m,
                        result.smoothed_ranges_m,
                        result.speeds_mps,
                        result.valid_mask.astype(int),
                        result.confidence_db,
                    )
                )
            elif self.current_module == "range_doppler" and isinstance(result, speed.RangeDopplerResult):
                writer.writerow(["range_m", "velocity_mps", "relative_power_db"])
                for ridx, range_m in enumerate(result.ranges_m):
                    for vidx, velocity in enumerate(result.velocities_mps):
                        writer.writerow([range_m, velocity, result.map_db[ridx, vidx]])
            elif self.current_module == "angle" and isinstance(result, tuple):
                single, multi, _range_spectrum = result
                writer.writerow(["type", "range_m", "angle_deg", "value"])
                writer.writerow(["single_phase", single.target_range_m, single.phase_angle_deg, single.mean_delta_deg])
                writer.writerow(["single_fft", single.target_range_m, single.angle_fft_angle_deg, ""])
                for detection in multi.detections:
                    writer.writerow(["cfar", detection.range_m, detection.angle_deg, detection.relative_power_db])
            else:
                writer.writerow(["message"])
                writer.writerow(["当前模块没有可导出的明细数据"])
