import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ============================================================
# Wavelet Denoising with Learned Shrinkage (KAN-style)
# Single-file project pipeline for audio denoising
# ------------------------------------------------------------
# Why this version:
# 1) Uses Python end-to-end for audio, wavelets, training, plots.
# 2) Avoids niche KAN package dependency by implementing a
#    learnable 1D shrinkage function with radial basis expansion.
# 3) Uses SWT (stationary wavelet transform) from PyWavelets as a
#    practical shift-invariant substitute for MODWT.
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    wav_path: str
    out_dir: str = "outputs"
    sample_rate: int = 16000
    mono: bool = True
    max_minutes: float = 12.0          # use a subset first for manageable runtime
    segment_seconds: float = 2.0
    hop_seconds: float = 1.0
    wavelet: str = "db4"
    levels: int = 4
    snr_train: Tuple[int, ...] = (0, 5, 10)
    snr_eval: Tuple[int, ...] = (0, 5, 10)
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-6
    max_coeff_samples: int = 100000
    hidden_basis: int = 65             # number of basis functions in KAN-style shrinker
    basis_scale: float = 0.35
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -----------------------------
# Audio utilities
# -----------------------------
def read_audio(path: str, target_sr: int, mono: bool = True, max_minutes: float = None) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)

    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)

    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)

    # simple resampling without librosa/scipy dependency
    if sr != target_sr:
        duration = len(audio) / sr
        old_t = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_len = int(round(duration * target_sr))
        new_t = np.linspace(0.0, duration, num=new_len, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
        sr = target_sr

    if max_minutes is not None:
        max_samples = int(max_minutes * 60 * sr)
        audio = audio[:max_samples]

    # peak normalize to [-1, 1]
    peak = np.max(np.abs(audio)) + 1e-12
    audio = (audio / peak).astype(np.float32)
    return audio, sr


def split_segments(audio: np.ndarray, sr: int, segment_seconds: float, hop_seconds: float) -> np.ndarray:
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(audio) < seg_len:
        raise ValueError("Audio is shorter than one segment. Reduce segment_seconds.")

    segments = []
    for start in range(0, len(audio) - seg_len + 1, hop_len):
        segments.append(audio[start:start + seg_len].copy())
    return np.stack(segments, axis=0)


# -----------------------------
# Noise and metrics
# -----------------------------
def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(x ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    num = np.sum(clean ** 2) + 1e-12
    den = np.sum((clean - estimate) ** 2) + 1e-12
    return float(10.0 * np.log10(num / den))


# -----------------------------
# Wavelet utilities
# SWT is used as a practical substitute for MODWT
# -----------------------------
def swt_decompose(x: np.ndarray, wavelet: str, level: int):
    # trim/pad so length is divisible by 2**level
    block = 2 ** level
    n = len(x)
    target = int(math.ceil(n / block) * block)
    if target != n:
        x_pad = np.pad(x, (0, target - n), mode="reflect")
    else:
        x_pad = x
    coeffs = pywt.swt(x_pad, wavelet, level=level, norm=True)
    return coeffs, n


def swt_reconstruct(coeffs, wavelet: str, original_len: int) -> np.ndarray:
    x_hat = pywt.iswt(coeffs, wavelet, norm=True)
    return np.asarray(x_hat[:original_len], dtype=np.float32)


def universal_threshold(detail: np.ndarray) -> float:
    sigma = np.median(np.abs(detail)) / 0.6745 + 1e-12
    n = detail.size
    return float(sigma * math.sqrt(2.0 * math.log(n + 1.0)))


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def hard_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    out = x.copy()
    out[np.abs(out) < lam] = 0.0
    return out


def baseline_denoise(noisy: np.ndarray, wavelet: str, level: int, mode: str = "soft") -> np.ndarray:
    coeffs, original_len = swt_decompose(noisy, wavelet, level)
    new_coeffs = []
    for cA, cD in coeffs:
        lam = universal_threshold(cD)
        if mode == "soft":
            cD_hat = soft_threshold(cD, lam)
        elif mode == "hard":
            cD_hat = hard_threshold(cD, lam)
        else:
            raise ValueError("mode must be 'soft' or 'hard'")
        new_coeffs.append((cA, cD_hat))
    return swt_reconstruct(new_coeffs, wavelet, original_len)


# -----------------------------
# Dataset for coefficient pairs
# Each sample is one coefficient value:
# noisy detail coeff -> clean detail coeff
# -----------------------------
class WaveletCoeffDataset(Dataset):
    def __init__(self, segments: np.ndarray, wavelet: str, levels: int,
                 snr_choices: Tuple[int, ...], max_coeff_samples: int = 200000):
        xs = []
        ys = []

        print(f"Preparing coefficient dataset from {len(segments)} segments...")

        for seg_idx, clean_seg in enumerate(segments):
            noisy_seg = add_awgn(clean_seg, random.choice(snr_choices))
            clean_coeffs, _ = swt_decompose(clean_seg, wavelet, levels)
            noisy_coeffs, _ = swt_decompose(noisy_seg, wavelet, levels)

            for (_, clean_d), (_, noisy_d) in zip(clean_coeffs, noisy_coeffs):
                xs.append(noisy_d.reshape(-1, 1))
                ys.append(clean_d.reshape(-1, 1))

            if seg_idx % 20 == 0:
                print(f"Processed segments: {seg_idx + 1}/{len(segments)}")

        x_all = np.concatenate(xs, axis=0).astype(np.float32)
        y_all = np.concatenate(ys, axis=0).astype(np.float32)

        print("Total coefficient samples before subsampling:", len(x_all))

        if len(x_all) > max_coeff_samples:
            idx = np.random.choice(len(x_all), size=max_coeff_samples, replace=False)
            x_all = x_all[idx]
            y_all = y_all[idx]
            print("Subsampled coefficient samples:", len(x_all))

        self.x = torch.from_numpy(x_all)
        self.y = torch.from_numpy(y_all)

        self.mu = self.x.mean(dim=0)
        self.std = self.x.std(dim=0) + 1e-6
        self.x = (self.x - self.mu) / self.std
        self.y = (self.y - self.mu) / self.std

        print("Final dataset size:", self.x.shape)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# -----------------------------
# KAN-style 1D shrinkage model
# y = linear(x) + sum_i a_i * phi((x - c_i)/s_i)
# This is still a learnable univariate nonlinear mapping,
# which matches the project's learned shrinkage idea.
# -----------------------------
class KANShrinkage1D(nn.Module):
    """
    Residual KAN-style shrinkage:
        y = x + alpha * g(x)
    """
    def __init__(self, n_basis: int = 49, init_scale: float = 0.35):
        super().__init__()

        centers = torch.linspace(-3.0, 3.0, n_basis).view(1, n_basis)
        self.centers = nn.Parameter(centers)

        self.log_scales = nn.Parameter(
            torch.full((1, n_basis), math.log(init_scale))
        )

        self.weights = nn.Parameter(torch.zeros(1, n_basis))

        self.res_w = nn.Parameter(torch.zeros(1, 1))
        self.res_b = nn.Parameter(torch.zeros(1, 1))

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1]
        z = x - self.centers
        scales = torch.exp(self.log_scales).clamp_min(1e-4)

        # Gaussian basis
        phi = torch.exp(-0.5 * (z / scales) ** 2)      # [B, n_basis]

        nonlinear = phi @ self.weights.t()             # [B, 1]
        residual = nonlinear + x @ self.res_w + self.res_b

        # residual parameterization
        y = x + self.alpha * residual
        return y


# -----------------------------
# Training utilities
# -----------------------------
def make_splits(segments: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(segments)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return segments[train_idx], segments[val_idx], segments[test_idx]


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                cfg: Config):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    model.to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = total / max(count, 1)

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)
                count += xb.size(0)
        val_loss = total / max(count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# -----------------------------
# Inference with learned shrinkage
# -----------------------------
def learned_denoise(noisy: np.ndarray,
                    model: nn.Module,
                    norm_mu: torch.Tensor,
                    norm_std: torch.Tensor,
                    wavelet: str,
                    levels: int,
                    device: str) -> np.ndarray:
    coeffs, original_len = swt_decompose(noisy, wavelet, levels)
    new_coeffs = []

    model.eval()
    with torch.no_grad():
        for cA, cD in coeffs:
            x = torch.from_numpy(cD.reshape(-1, 1).astype(np.float32))
            x_norm = (x - norm_mu) / norm_std
            y_norm = model(x_norm.to(device)).cpu()
            y = y_norm * norm_std + norm_mu
            cD_hat = y.numpy().reshape(-1).astype(np.float32)
            new_coeffs.append((cA, cD_hat))

    return swt_reconstruct(new_coeffs, wavelet, original_len)


# -----------------------------
# Evaluation and plotting
# -----------------------------
def evaluate_method(segments: np.ndarray,
                    cfg: Config,
                    method_name: str,
                    model: nn.Module = None,
                    norm_mu: torch.Tensor = None,
                    norm_std: torch.Tensor = None) -> Dict[int, Dict[str, float]]:
    results = {}

    for snr in cfg.snr_eval:
        in_snr_list = []
        out_snr_list = []
        mse_list = []

        for clean_seg in segments:
            noisy_seg = add_awgn(clean_seg, snr)

            if method_name == "soft":
                est = baseline_denoise(noisy_seg, cfg.wavelet, cfg.levels, mode="soft")
            elif method_name == "hard":
                est = baseline_denoise(noisy_seg, cfg.wavelet, cfg.levels, mode="hard")
            elif method_name == "learned":
                est = learned_denoise(noisy_seg, model, norm_mu, norm_std, cfg.wavelet, cfg.levels, cfg.device)
            else:
                raise ValueError("Unknown method")

            in_snr_list.append(snr_db(clean_seg, noisy_seg))
            out_snr_list.append(snr_db(clean_seg, est))
            mse_list.append(mse(clean_seg, est))

        results[snr] = {
            "input_snr_mean": float(np.mean(in_snr_list)),
            "output_snr_mean": float(np.mean(out_snr_list)),
            "snr_improvement_mean": float(np.mean(np.array(out_snr_list) - np.array(in_snr_list))),
            "mse_mean": float(np.mean(mse_list)),
        }
    return results


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shrinkage_curve(model: nn.Module, out_path: str, device: str) -> None:
    x = torch.linspace(-3, 3, 800).view(-1, 1).to(device)
    with torch.no_grad():
        y = model(x).cpu().numpy().reshape(-1)
    x = x.cpu().numpy().reshape(-1)

    plt.figure(figsize=(6, 4))
    plt.plot(x, x, label="identity")
    plt.plot(x, y, label="learned shrinkage")
    plt.xlabel("Input coefficient (normalized)")
    plt.ylabel("Output coefficient (normalized)")
    plt.title("Learned shrinkage function")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_waveforms(clean: np.ndarray, noisy: np.ndarray, est: np.ndarray, sr: int, out_path: str) -> None:
    t = np.arange(len(clean)) / sr
    plt.figure(figsize=(10, 5))
    plt.plot(t, clean, label="clean", alpha=0.9)
    plt.plot(t, noisy, label="noisy", alpha=0.6)
    plt.plot(t, est, label="denoised", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_table(results_all: Dict[str, Dict[int, Dict[str, float]]], out_path: str) -> None:
    lines = ["method,snr_in_target,input_snr_mean,output_snr_mean,snr_improvement_mean,mse_mean"]
    for method_name, by_snr in results_all.items():
        for snr, vals in by_snr.items():
            lines.append(
                f"{method_name},{snr},{vals['input_snr_mean']:.6f},{vals['output_snr_mean']:.6f},"
                f"{vals['snr_improvement_mean']:.6f},{vals['mse_mean']:.8f}"
            )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main pipeline
# -----------------------------
def run(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Loading audio...")
    audio, sr = read_audio(cfg.wav_path, cfg.sample_rate, cfg.mono, cfg.max_minutes)
    print(f"Audio length used: {len(audio)/sr/60:.2f} min | sr={sr}")

    print("Segmenting audio...")
    segments = split_segments(audio, sr, cfg.segment_seconds, cfg.hop_seconds)
    print(f"Total segments: {len(segments)}")

    train_seg, val_seg, test_seg = make_splits(segments, cfg.train_ratio, cfg.val_ratio)
    print(f"Train/Val/Test: {len(train_seg)}/{len(val_seg)}/{len(test_seg)}")

    print("Building coefficient datasets...")
    train_ds = WaveletCoeffDataset(
        train_seg, cfg.wavelet, cfg.levels, cfg.snr_train, cfg.max_coeff_samples
    )
    val_ds = WaveletCoeffDataset(
        val_seg, cfg.wavelet, cfg.levels, cfg.snr_train, min(10000, cfg.max_coeff_samples)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print("Training learned shrinkage model...")
    model = KANShrinkage1D(n_basis=cfg.hidden_basis, init_scale=cfg.basis_scale)
    model, history = train_model(model, train_loader, val_loader, cfg)

    # save model and normalizer from training dataset
    ckpt = {
        "model_state": model.state_dict(),
        "norm_mu": train_ds.mu,
        "norm_std": train_ds.std,
        "config": cfg.__dict__,
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, "learned_shrinkage_model.pt"))

    print("Evaluating baselines and learned method...")
    soft_results = evaluate_method(test_seg, cfg, method_name="soft")
    hard_results = evaluate_method(test_seg, cfg, method_name="hard")
    learned_results = evaluate_method(
        test_seg,
        cfg,
        method_name="learned",
        model=model,
        norm_mu=train_ds.mu,
        norm_std=train_ds.std,
    )

    results_all = {
        "soft": soft_results,
        "hard": hard_results,
        "learned": learned_results,
    }
    save_results_table(results_all, os.path.join(cfg.out_dir, "results.csv"))

    print("Saving plots and demo audio...")
    plot_history(history, os.path.join(cfg.out_dir, "training_history.png"))
    plot_shrinkage_curve(model, os.path.join(cfg.out_dir, "learned_shrinkage_curve.png"), cfg.device)

    demo_clean = test_seg[0]

    for snr_demo in [0, 5, 10]:
        print(f"Generating demo for SNR = {snr_demo} dB")

        # add noises
        demo_noisy = add_awgn(demo_clean, snr_demo)

        # 3 methods
        demo_soft = baseline_denoise(demo_noisy, cfg.wavelet, cfg.levels, mode="soft")
        demo_hard = baseline_denoise(demo_noisy, cfg.wavelet, cfg.levels, mode="hard")
        demo_learned = learned_denoise(
            demo_noisy,
            model,
            train_ds.mu,
            train_ds.std,
            cfg.wavelet,
            cfg.levels,
            cfg.device,
        )

        # save audios
        sf.write(os.path.join(cfg.out_dir, f"demo_clean.wav"), demo_clean, sr)
        sf.write(os.path.join(cfg.out_dir, f"demo_noisy_{snr_demo}dB.wav"), demo_noisy, sr)
        sf.write(os.path.join(cfg.out_dir, f"demo_soft_{snr_demo}dB.wav"), demo_soft, sr)
        sf.write(os.path.join(cfg.out_dir, f"demo_hard_{snr_demo}dB.wav"), demo_hard, sr)
        sf.write(os.path.join(cfg.out_dir, f"demo_learned_{snr_demo}dB.wav"), demo_learned, sr)

    print("\n=== Test Results ===")
    for method_name, by_snr in results_all.items():
        print(f"\n[{method_name}]")
        for snr, vals in by_snr.items():
            print(
                f"Target SNR {snr:>2} dB | "
                f"Input {vals['input_snr_mean']:.3f} dB | "
                f"Output {vals['output_snr_mean']:.3f} dB | "
                f"Gain {vals['snr_improvement_mean']:.3f} dB | "
                f"MSE {vals['mse_mean']:.8f}"
            )

    print(f"\nDone. Outputs saved to: {cfg.out_dir}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet denoising with learned shrinkage (KAN-style)")
    parser.add_argument("--wav_path", type=str, required=True, help="Path to clean WAV file")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_minutes", type=float, default=12.0)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--hop_seconds", type=float, default=1.0)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        wav_path=args.wav_path,
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
        max_minutes=args.max_minutes,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        wavelet=args.wavelet,
        levels=args.levels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    run(cfg)


if __name__ == "__main__":
    main()
