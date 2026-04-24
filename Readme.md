# 🎧 Wavelet KAN Audio Denoising Project

## 📌 Overview

All scripts operate on a clean `.wav` file and generate denoised outputs and evaluation results.

The audio used in this project is derived from the Internet Archive recording:
“Ravel – Piano Concerto in G Major & Piano Concerto for the Left Hand (RAW WAV),”
accessed Dec. 2025. [Online]. Available:  
https://archive.org/details/ravel-piano-concerto-i-n-g-m-83-piano-concerto-for-the-left-hand-raw-wav

For computational efficiency, the original recording was trimmed to approximately 10 minutes during experimentation.

Due to repository size constraints, the audio file is not included in this repository.  
However, the results provided in `outputs_example/` were generated using the above audio source.

---

## ⭐ Final Model Selection

The final implementation used in this project is:
```bash
wavelet_muti_level_kan_denoising_project.py
```

**Reason:**

- Applies **one learned shrinkage function per wavelet level**
- Better captures multi-scale noise structure
- Produces **significantly improved denoising performance**

The other two versions (`wavelet_kan_denoising_project.py` and `wavelet_kan_residual_denoising_project.py`) were tested but showed **inferior performance** and are included for comparison and ablation study.

---

## 📂 File Structure
```
project/
│
├── wavelet_kan_denoising_project.py
├── wavelet_kan_residual_denoising_project.py
├── wavelet_muti_level_kan_denoising_project.py
├── make_audio_figures.py
├── outputs_example/
├── images/
└── Readme.md
```
---

## ⚙️ Installation

### 1. Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install numpy soundfile pywavelets matplotlib torch
```

## ▶️ How to Run Denoising

All three denoising scripts follow the same interface.

Run command:
```bash
python wavelet_XXX.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

Replace wavelet_XXX.py with:
- wavelet_kan_denoising_project.py
- wavelet_kan_residual_denoising_project.py
- wavelet_muti_level_kan_denoising_project.py ✅ (recommended)

Example (final model)
```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --segment_seconds 1 --hop_seconds 1 --epochs 8 --batch_size 6
```

## 📊 Output

After running, an outputs/ folder will be created containing:
- learned_shrinkage_model.pt → trained model
- results.csv → evaluation metrics (SNR, MSE)
- training_history.png → training curve
- learned_shrinkage_curve.png → learned function
- demo_*.wav → denoised audio samples

## 📈 Visualization

To generate comparison figures (PSD, spectrogram, etc.):

```bash
python make_audio_figures.py
```

This script reads generated outputs and produces:
- PSD comparison
- Spectrogram comparison
- Quiet-section spectrum

## 🔬 Method Summary
- Uses **SWT (Stationary Wavelet Transform)**
- Learns a **KAN-style nonlinear shrinkage function**
- Compares:
  - Soft threshold
  - Hard threshold
  - Learned shrinkage

The multi-level approach further improves performance by learning independent shrinkage functions for each wavelet level, enabling better modeling of frequency-dependent noise characteristics.

## 💡 Notes
- Input must be a clean `.wav` file
- File paths with spaces must be quoted
Runtime can be reduced using: 
```bash
--max_minutes 5 --epochs 3
```

## 🚀 Recommended Usage
For best results:
```bash
python wavelet_muti_level_kan_denoising_project.py --wav_path "path/to/your_audio.wav" --max_minutes 10 --epochs 8
```