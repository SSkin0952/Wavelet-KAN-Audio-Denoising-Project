# 🎧 Wavelet KAN Audio Denoising Project

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

### 📌 Overview

This project explores audio denoising using the **Stationary Wavelet Transform (SWT)** and **KAN-style nonlinear shrinkage functions**. All scripts operate on a clean `.wav` file and generate denoised outputs and evaluation results.

The audio used in this project is derived from the Internet Archive recording:

> “Ravel – Piano Concerto in G Major & Piano Concerto for the Left Hand (RAW WAV),” accessed December 2025.

[View the source recording on Internet Archive](https://archive.org/details/ravel-piano-concerto-i-n-g-m-83-piano-concerto-for-the-left-hand-raw-wav)

For computational efficiency, the original recording was trimmed to approximately 10 minutes during experimentation.

Due to repository size constraints, the audio file is not included in this repository. The results in `outputs_example/` were generated using the audio source above.

### ⭐ Final Model Selection

The final implementation used in this project is:

```text
wavelet_muti_level_kan_denoising_project.py
```

Reasons for selecting this model:

- applies **one learned shrinkage function per wavelet level**
- better captures multi-scale noise structure
- produces **significantly improved denoising performance**

The other two versions, `wavelet_kan_denoising_project.py` and `wavelet_kan_residual_denoising_project.py`, were tested but showed inferior performance. They are retained for comparison and ablation analysis.

### 📂 File Structure

```text
project/
├── wavelet_kan_denoising_project.py
├── wavelet_kan_residual_denoising_project.py
├── wavelet_muti_level_kan_denoising_project.py
├── make_audio_figures.py
├── outputs_example/
├── images/
└── README.md
```

### ⚙️ Installation

#### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install numpy soundfile pywavelets matplotlib torch
```

### ▶️ How to Run Denoising

All three denoising scripts follow the same interface.

General command:

```bash
python wavelet_XXX.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

Replace `wavelet_XXX.py` with one of the following:

- `wavelet_kan_denoising_project.py`
- `wavelet_kan_residual_denoising_project.py`
- `wavelet_muti_level_kan_denoising_project.py` — recommended

Example using the final model:

```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

### 📊 Output

After execution, an `outputs/` folder will be created containing:

| Output | Description |
|---|---|
| `learned_shrinkage_model.pt` | Trained shrinkage model |
| `results.csv` | Evaluation metrics, including SNR and MSE |
| `training_history.png` | Training curve |
| `learned_shrinkage_curve.png` | Visualization of the learned function |
| `demo_*.wav` | Denoised audio samples |

### 📈 Visualization

To generate comparison figures such as PSD and spectrograms, run:

```bash
python make_audio_figures.py
```

The script reads the generated outputs and produces:

- PSD comparison
- spectrogram comparison
- quiet-section spectrum

### 🔬 Method Summary

- uses **SWT (Stationary Wavelet Transform)**
- learns a **KAN-style nonlinear shrinkage function**
- compares three denoising strategies:
  - soft thresholding
  - hard thresholding
  - learned shrinkage

The multi-level approach learns an independent shrinkage function for each wavelet level, enabling more effective modeling of frequency-dependent noise characteristics.

### 💡 Notes

- The input must be a clean `.wav` file.
- File paths containing spaces must be enclosed in quotation marks.
- Runtime can be reduced using:

```bash
--max_minutes 5 --epochs 3
```

### 🚀 Recommended Usage

For the best results, run:

```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --epochs 8
```

---

<a id="中文"></a>

## 中文

### 📌 项目概述

本项目使用**平稳小波变换（Stationary Wavelet Transform, SWT）**与 **KAN 风格的非线性收缩函数**进行音频降噪。所有脚本均以干净的 `.wav` 音频为输入，生成降噪结果及相应的评估指标。

本项目使用的音频来源于 Internet Archive：

> “Ravel – Piano Concerto in G Major & Piano Concerto for the Left Hand (RAW WAV)”，访问时间为 2025 年 12 月。

[在 Internet Archive 查看原始录音](https://archive.org/details/ravel-piano-concerto-i-n-g-m-83-piano-concerto-for-the-left-hand-raw-wav)

为提高计算效率，实验过程中将原始录音裁剪为约 10 分钟。

受仓库文件大小限制，项目未包含原始音频文件。`outputs_example/` 中的结果由上述音频生成。

### ⭐ 最终模型选择

本项目最终采用的实现为：

```text
wavelet_muti_level_kan_denoising_project.py
```

选择该模型的原因：

- 为**每个小波分解层级分别学习一个收缩函数**
- 能够更好地捕捉多尺度噪声结构
- 可获得**明显更优的降噪性能**

另外两个版本 `wavelet_kan_denoising_project.py` 和 `wavelet_kan_residual_denoising_project.py` 也完成了测试，但性能相对较低，因此保留用于模型对比和消融分析。

### 📂 文件结构

```text
project/
├── wavelet_kan_denoising_project.py
├── wavelet_kan_residual_denoising_project.py
├── wavelet_muti_level_kan_denoising_project.py
├── make_audio_figures.py
├── outputs_example/
├── images/
└── README.md
```

### ⚙️ 环境安装

#### 1. 创建虚拟环境

```bash
python -m venv .venv
```

在 Windows 中激活虚拟环境：

```powershell
.venv\Scripts\activate
```

#### 2. 安装依赖

```bash
pip install numpy soundfile pywavelets matplotlib torch
```

### ▶️ 运行降噪程序

三个降噪脚本采用相同的命令行接口。

通用命令：

```bash
python wavelet_XXX.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

将 `wavelet_XXX.py` 替换为以下任一脚本：

- `wavelet_kan_denoising_project.py`
- `wavelet_kan_residual_denoising_project.py`
- `wavelet_muti_level_kan_denoising_project.py` — 推荐

使用最终模型的示例：

```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

### 📊 输出文件

运行后将创建 `outputs/` 文件夹，其中包括：

| 输出文件 | 说明 |
|---|---|
| `learned_shrinkage_model.pt` | 训练完成的收缩模型 |
| `results.csv` | SNR、MSE 等评估指标 |
| `training_history.png` | 训练过程曲线 |
| `learned_shrinkage_curve.png` | 学习得到的收缩函数曲线 |
| `demo_*.wav` | 降噪后的音频示例 |

### 📈 可视化

运行以下命令可生成 PSD、声谱图等对比图：

```bash
python make_audio_figures.py
```

该脚本读取已生成的输出，并绘制：

- 功率谱密度（PSD）对比图
- 声谱图对比
- 安静片段频谱图

### 🔬 方法概述

- 使用**平稳小波变换（SWT）**
- 学习 **KAN 风格的非线性收缩函数**
- 对比三种降噪策略：
  - 软阈值
  - 硬阈值
  - 学习型收缩函数

多层级方法为每个小波层级学习独立的收缩函数，因此能够更有效地建模随频率变化的噪声特征。

### 💡 注意事项

- 输入必须为干净的 `.wav` 文件。
- 包含空格的文件路径必须使用引号包裹。
- 可以使用以下参数缩短运行时间：

```bash
--max_minutes 5 --epochs 3
```

### 🚀 推荐运行方式

为获得最佳效果，建议运行：

```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --epochs 8
```
