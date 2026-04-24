import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "outputs"

CLEAN_PATH = os.path.join(OUTPUT_DIR, "demo_clean.wav")
NOISY_PATH = os.path.join(OUTPUT_DIR, "demo_noisy_5dB.wav")
HARD_PATH = os.path.join(OUTPUT_DIR, "demo_hard_5dB.wav")
LEARNED_PATH = os.path.join(OUTPUT_DIR, "demo_learned_5dB.wav")

FIG_PSD_PATH = os.path.join(OUTPUT_DIR, "fig_psd_comparison_5dB.png")
FIG_SPEC_PATH = os.path.join(OUTPUT_DIR, "fig_spectrogram_comparison_5dB.png")
FIG_QUIET_PATH = os.path.join(OUTPUT_DIR, "fig_quiet_section_average_spectrum_5dB.png")


# ============================================================
# Utilities
# ============================================================
def load_audio(path: str):
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def ensure_same_length(signals):
    min_len = min(len(x) for x in signals)
    return [x[:min_len] for x in signals]


def stft_mag_db(x: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256):
    """
    Simple STFT magnitude in dB using numpy only.
    Returns:
        S_db: [freq_bins, time_frames]
        freqs: [freq_bins]
        times: [time_frames]
    """
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    window = np.hanning(n_fft)
    frames = []
    times = []

    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        frames.append(mag)
        times.append((start + n_fft / 2) / sr)

    S = np.stack(frames, axis=1)  # [freq_bins, time_frames]
    S_db = 20 * np.log10(S + 1e-10)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.array(times)
    return S_db, freqs, times


def welch_psd(x: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 1024):
    """
    Simple Welch PSD estimate using numpy only.
    Returns:
        freqs, psd_db
    """
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    window = np.hanning(n_fft)
    U = np.sum(window ** 2)

    psd_accum = []
    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft] * window
        X = np.fft.rfft(frame)
        Pxx = (np.abs(X) ** 2) / (U * sr)
        psd_accum.append(Pxx)

    Pxx_mean = np.mean(np.stack(psd_accum, axis=0), axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    psd_db = 10 * np.log10(Pxx_mean + 1e-20)
    return freqs, psd_db


def frame_rms(x: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    vals = []
    centers = []
    for start in range(0, len(x) - frame_length + 1, hop_length):
        frame = x[start:start + frame_length]
        vals.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
        centers.append(start + frame_length // 2)
    return np.array(vals), np.array(centers)


def select_quiet_samples(x: np.ndarray,
                         frame_length: int = 2048,
                         hop_length: int = 512,
                         quiet_percentile: float = 20.0):
    """
    Select quiet frames based on frame RMS.
    Returns concatenated quiet-frame samples.
    """
    rms_vals, centers = frame_rms(x, frame_length=frame_length, hop_length=hop_length)
    if len(rms_vals) == 0:
        return x.copy()

    thr = np.percentile(rms_vals, quiet_percentile)

    quiet_segments = []
    idx = 0
    for start in range(0, len(x) - frame_length + 1, hop_length):
        if rms_vals[idx] <= thr:
            quiet_segments.append(x[start:start + frame_length])
        idx += 1

    if len(quiet_segments) == 0:
        return x.copy()

    return np.concatenate(quiet_segments, axis=0)


# ============================================================
# Plot 1: PSD comparison
# ============================================================
def plot_psd_comparison(clean, noisy, hard, learned, sr, save_path):
    plt.figure(figsize=(9, 6))

    for sig, label in [
        (clean, "Clean"),
        (noisy, "Noisy (5 dB)"),
        (hard, "Hard threshold"),
        (learned, "Learned multilevel"),
    ]:
        f, p = welch_psd(sig, sr, n_fft=2048, hop_length=1024)
        plt.semilogx(f[1:], p[1:], label=label)  # skip DC for cleaner log plot

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.title("PSD Comparison at 5 dB Input SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# Plot 2: Spectrogram comparison
# ============================================================
def plot_spectrogram_comparison(clean, noisy, hard, learned, sr, save_path):
    signals = [
        (clean, "Clean"),
        (noisy, "Noisy (5 dB)"),
        (hard, "Hard threshold"),
        (learned, "Learned multilevel"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    vmin = None
    vmax = None
    specs = []

    # First pass: compute all spectrograms and shared color range
    for sig, _ in signals:
        S_db, freqs, times = stft_mag_db(sig, sr, n_fft=1024, hop_length=256)
        specs.append((S_db, freqs, times))
        this_vmin = np.percentile(S_db, 5)
        this_vmax = np.percentile(S_db, 99)
        vmin = this_vmin if vmin is None else min(vmin, this_vmin)
        vmax = this_vmax if vmax is None else max(vmax, this_vmax)

    # Second pass: plot
    for ax, (sig, title), (S_db, freqs, times) in zip(axes, signals, specs):
        im = ax.imshow(
            S_db,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")

    fig.suptitle("Spectrogram Comparison at 5 dB Input SNR", fontsize=14)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.92, label="Magnitude (dB)")
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# Plot 3: Quiet-section average spectrum
# ============================================================
def average_spectrum(x: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 1024):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))

    window = np.hanning(n_fft)
    mags = []
    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft] * window
        X = np.fft.rfft(frame)
        mag = np.abs(X)
        mags.append(mag)

    avg_mag = np.mean(np.stack(mags, axis=0), axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    avg_db = 20 * np.log10(avg_mag + 1e-10)
    return freqs, avg_db


def plot_quiet_section_average_spectrum(clean, noisy, hard, learned, sr, save_path):
    # Select quiet sections independently for each signal
    clean_q = select_quiet_samples(clean, frame_length=2048, hop_length=512, quiet_percentile=20.0)
    noisy_q = select_quiet_samples(noisy, frame_length=2048, hop_length=512, quiet_percentile=20.0)
    hard_q = select_quiet_samples(hard, frame_length=2048, hop_length=512, quiet_percentile=20.0)
    learned_q = select_quiet_samples(learned, frame_length=2048, hop_length=512, quiet_percentile=20.0)

    plt.figure(figsize=(10, 6))

    for sig, label in [
        (clean_q, "Clean (quiet frames)"),
        (noisy_q, "Noisy (quiet frames)"),
        (hard_q, "Hard threshold (quiet frames)"),
        (learned_q, "Learned multilevel (quiet frames)"),
    ]:
        f, s = average_spectrum(sig, sr, n_fft=2048, hop_length=1024)
        plt.plot(f[1:], s[1:], label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Average Spectrum Over Quiet Sections")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    # Load files
    clean, sr1 = load_audio(CLEAN_PATH)
    noisy, sr2 = load_audio(NOISY_PATH)
    hard, sr3 = load_audio(HARD_PATH)
    learned, sr4 = load_audio(LEARNED_PATH)

    if not (sr1 == sr2 == sr3 == sr4):
        raise ValueError("Sample rates do not match among input audio files.")

    clean, noisy, hard, learned = ensure_same_length([clean, noisy, hard, learned])

    # Generate figures
    plot_psd_comparison(clean, noisy, hard, learned, sr1, FIG_PSD_PATH)
    plot_spectrogram_comparison(clean, noisy, hard, learned, sr1, FIG_SPEC_PATH)
    plot_quiet_section_average_spectrum(clean, noisy, hard, learned, sr1, FIG_QUIET_PATH)

    print("Saved:")
    print(" -", FIG_PSD_PATH)
    print(" -", FIG_SPEC_PATH)
    print(" -", FIG_QUIET_PATH)


if __name__ == "__main__":
    main()