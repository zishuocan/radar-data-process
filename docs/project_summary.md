# 雷达 bin 文件统一处理程序 — 项目总结

## 概述

北航（BUAA）第二周 FMCW 雷达实验的 `.bin` 文件桌面处理程序。使用 Python + tkinter 构建原生 GUI，提供完整的雷达信号处理链路：数据读取、测距/HRRP、测距精度分析、测速、距离-多普勒图、测角等功能。

## 技术栈

| 层次      | 技术                                           |
| -------- | ---------------------------------------------- |
| 语言      | Python ≥ 3.10                                  |
| GUI      | tkinter / ttk（标准库）                          |
| 数值计算  | NumPy                                          |
| 科学计算  | FFT 运算、CFAR 检测、抛物线插值、维纳滤波（自实现） |
| 绘图      | Matplotlib（tkagg 后端嵌入 GUI）                 |
| 打包/项目 | pyproject.toml + setuptools                     |

## 项目结构

```
radar-data-process-main/
├── pyproject.toml              # 项目元数据与依赖
├── environment.yml             # Conda 环境定义（含 tk 和 pip 安装）
├── .gitignore
├── README.md                   # 使用说明（中文）
├── docs/
│   └── project_summary.md      # 本文档
└── src/
    └── radar_app/
        ├── __init__.py
        ├── __main__.py          # python -m radar_app 入口
        ├── main.py              # 应用入口 main()
        ├── core/
        │   ├── __init__.py      # 导出核心 API
        │   ├── params.py        # 雷达参数模型（起始频率、斜率、采样率等）
        │   ├── io.py            # .bin 文件读取，多格式/多通道自适应
        │   └── signal.py        # 信号处理基元（窗函数、FFT、dB 归一化）
        ├── modules/
        │   ├── __init__.py
        │   ├── range_hrrp.py    # 测距 / HRRP
        │   ├── range_precision.py  # 测距精度（全带宽/半带宽对比）
        │   ├── speed.py         # 测速 + 距离-多普勒图
        │   └── angle.py         # 测角（相位法 + FFT 法 + CFAR 多目标）
        └── ui/
            ├── __init__.py
            └── main_window.py   # 主窗口（1360×900，Notebook 多标签页）
```

## 核心模块详解

### 1. core/params.py — 雷达参数

使用 `dataclass` 定义 FMCW 雷达参数模型：

- **起始频率** `start_freq_ghz`（默认 75 GHz）
- **调频斜率** `slope_mhz_us`（默认 30 MHz/μs）
- **采样率** `sample_rate_ksps`（默认 3000 ksps）
- **每 Chirp 采样点** `samples_per_chirp`（默认 256）
- **每帧 Chirp 数** `chirps_per_frame`（默认 8）
- **帧周期** `frame_period_ms`（默认 500 ms）

自动计算：带宽、距离分辨率、最大无模糊距离。

### 2. core/io.py — 数据 I/O

支持多种 `.bin` 数据格式的自动识别与加载：

| 格式           | 描述                       | 通道 |
| -------------- | -------------------------- | ---- |
| `int8_ant16`   | int8 16 通道原始帧（带帧头） | 16   |
| `iq_int16`     | int16 I/Q 交织复数          | 1    |
| `int16_real`   | int16 实数                  | 1    |
| `uint16_real`  | uint16 实数                 | 1    |

**自适应算法**：遍历所有格式和候选文件头大小（248/0/128/240/256…），选择帧对齐最完整、优先级最高的匹配项。

### 3. core/signal.py — 信号处理基元

- **窗函数**：rect / hann / hamming / blackman / bartlett / kaiser(β=8.6) / flattop，支持中英文命名
- **FFT**：复数信号用 `fft` + 取正半边，实数信号用 `rfft`，自动映射到距离轴
- **补零**：通过 `zero_padding_power` 控制，目标 FFT 点数 = 2^(ceil(log2(N)) + power)
- **归一化**：幅度归一化 dB、功率归一化 dB

### 4. modules/range_hrrp.py — 测距 / HRRP

- 从帧数据中选择天线和 chirp（支持平均模式）
- 去直流 → 加窗 → FFT → 幅度归一化
- **抛物线插值**：对峰值及其左右两点做二次插值，提高距离估计精度
- **多目标检测**：基于局部峰值 + 阈值筛选 + 最小间隔约束，按幅度排序
- **3dB 宽度测量**：从峰值向两侧搜索功率下降 3dB 的边界，含线性插值

### 5. modules/range_precision.py — 测距精度

对比 **全带宽（keep_ratio=1.0）** 和 **半带宽（keep_ratio=0.5）** 两种配置下的测距精度：

- 计算复 HRRP → 功率 → 检测目标峰值
- 3dB 旁瓣保护 → 排除目标区域后估计噪声功率
- SNR = 峰值功率 / 均值噪声功率
- 测距精度公式（克拉美罗下界近似）：
  $$ \sigma_R = \frac{\sqrt{3}}{\pi \sqrt{2 \cdot SNR}} \cdot \Delta R $$

### 6. modules/speed.py — 测速 + 距离-多普勒图

**测速流程**：
1. 逐帧做距离 FFT（去直流 → 加窗 → 复数 FFT / RFFT）
2. 可选静态杂波抑制（减均值）
3. **维特比动态规划跟踪**：逐帧在多候选峰值间寻找代价最低的路径
   - 发射代价：基于峰值相对功率
   - 转移代价：距离跳变（L2 范数）+ 可选接近目标偏置
4. 无效帧插值填充 + 滑动平均平滑
5. 速度计算：平滑后距离对时间的梯度
6. 速度汇总：平均值、拟合速度（polyfit 1 阶）、中位值、极值

**距离-多普勒图**：
- 选取单帧，沿慢时间轴（天线×Chirp 或仅 Chirp）做 Doppler FFT
- 速度分辨率、最大无模糊速度自动计算
- 速度范围可限幅，静态抑制可选

### 7. modules/angle.py — 测角

**单目标（相位法 + FFT 法）**：
- 用户指定 8 个虚拟通道序号
- 距离 FFT → 峰值处提取通道向量
- **相位法**：对相邻通道相位差求均值 → angle = arcsin(λ · Δφ / (2π · d))
- **FFT 法**：通道向量补零加窗做 FFT → 角谱峰值对应角度
- 可加载 0° 校准数据消除通道间固定相位差

**多目标（2D CFAR）**：
- 构建距离-角度二维功率图
- **积分图像加速**：O(1) 计算任意矩形区域的和
- CA-CFAR 检测器：保护单元 + 训练单元，逐像素比较
- 峰值筛选：3×3 局部极大值 + 最小间隔去重

## GUI 界面

1360×900 主窗口，使用 `ttk.Notebook` 多标签页布局：

| 标签页       | 功能         | 关键参数                                            |
| ------------ | ------------ | --------------------------------------------------- |
| 测距/HRRP   | 一维距离像   | 截取比例、目标数、忽略近端、最小目标间隔                   |
| 测距精度     | 精度分析     | 全带宽/半带宽窗、保护单元、噪声最大距离                    |
| 测速         | 速度估计     | 距离搜索范围、最大跳变、置信阈值、平滑帧数、静态抑制          |
| 距离-多普勒  | RD 热图      | Chirp 周期、Doppler FFT 点数、速度范围、慢时间模式         |
| 测角         | 角度估计     | 虚拟通道选择、阵元间距、方位 FFT 点数、CFAR 参数           |

通用操作：选择文件 → 自动识别格式 → 读取 → 切换标签页 → 处理 → 保存结果（PNG + CSV + 参数）。

## 依赖安装

### 使用 pip

```bash
pip install numpy matplotlib
pip install -e .
```

### 使用 conda（推荐）

```bash
conda env create -f environment.yml
conda activate radar-app
```

### 运行

```bash
cd src
PYTHONPATH="." python -m radar_app
```

## 算法亮点

- **自适应数据格式识别**：穷举格式×文件头组合，无需用户手动指定
- **抛物线插值**：突破 FFT 离散栅格限制，提高距离/速度估计精度
- **维特比轨迹跟踪**：多帧联合优化，鲁棒处理低 SNR 和杂波
- **2D CA-CFAR**：积分图像加速，实时性好的恒虚警检测
- **全带宽/半带宽对比**：直观展示带宽对测距精度的影响（课堂教学设计）

## license

该项目无明确 license 声明，为北航雷达课程实验用程序。
