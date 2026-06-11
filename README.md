# Radar App

统一的 FMCW 雷达 `.bin` 文件桌面处理程序,读取、测距、测距精度、测速、距离-多普勒和测角功能。

详细参数说明见 [docs/parameters.md](docs/parameters.md)。

## 环境配置

### 方法一：conda（推荐）

```bash
# 创建并激活 conda 环境（一次性）
conda env create -f environment.yml
conda activate radar-app

# 之后每次只需激活环境即可
conda activate radar-app
```

### 方法二：pip（在已有 Python ≥ 3.10 环境中）

```bash
pip install -r requirements.txt
pip install -e .
```

> **Linux 用户注意**：tkinter 需要系统级依赖。
> Ubuntu/Debian：`sudo apt install python3-tk`
> Fedora/RHEL：`sudo dnf install python3-tkinter`

## 运行

```powershell
python -m radar_app
```
