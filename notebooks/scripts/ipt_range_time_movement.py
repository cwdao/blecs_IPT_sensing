"""IPT movement recording → stepped-frequency range-time figures.

One logged frame is one slow-time sample: a sparse complex frequency response
H(f) across BLE channels. IFFT along frequency produces one fast-time / range
profile. Stacking profiles yields a range-time image, the same product as a
stepped-frequency radar.

Usage
-----
    python notebooks/scripts/ipt_range_time_movement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "sampleData" / "ipt_movement_test" / "IPT_frames_all_20261125_000521.jsonl"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

C_MPS = 299_792_458.0
DF_HZ = 1.0e6  # logged channel index step; project BLE tone spacing
F0_HZ = 2402.0e6  # assumed RF of channel index 0
N_GRID = 75  # channel indices 0..74; 21..23 are never present
NFFT = 512
DT_S = 0.030
SKIP_S = 5.0
DURATION_S = 16.0
STRUCTURAL_GAP = {21, 22, 23}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.grid": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "font.size": 10,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def load_ipt(path: Path):
    frames = []
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("record_type")
            if kind == "frame":
                frames.append(rec)
            elif kind == "event":
                events.append(rec)
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    t_ms = np.asarray([float(fr["t_dev_ms"]) for fr in frames], dtype=float)
    t_s = (t_ms - t_ms[0]) / 1000.0
    h = np.full((N_GRID, len(frames)), np.nan, dtype=np.complex128)
    for j, fr in enumerate(frames):
        for key, ch in (fr.get("channels") or {}).items():
            k = int(key)
            if k < 0 or k >= N_GRID:
                continue
            h[k, j] = complex(float(ch["I"]), float(ch["Q"]))
    return t_s, h, events


def interpolate_slow_time(t_s: np.ndarray, h: np.ndarray, dt: float = DT_S):
    """Fill short per-channel dropouts, then resample onto a uniform slow-time grid.

    Channels that are never observed (the 21–23 gap) stay zero. Zero-filling a
    tone that is only missing for a frame or two would modulate the IFFT.
    """
    t_u = np.arange(t_s[0], t_s[-1] + dt * 0.5, dt)
    out = np.zeros((h.shape[0], t_u.size), dtype=np.complex128)
    for k in range(h.shape[0]):
        if k in STRUCTURAL_GAP:
            continue
        z = h[k]
        good = np.isfinite(z.real)
        if int(good.sum()) < 2:
            continue
        out[k] = (
            np.interp(t_u, t_s[good], z.real[good])
            + 1j * np.interp(t_u, t_s[good], z.imag[good])
        )
    return t_u, out


def segments_from_events(t_first_dev_ms: float, events) -> list[dict]:
    rel = [(float(ev["nearest_t_dev_ms"]) - t_first_dev_ms) / 1000.0 for ev in events]
    if len(rel) < 2:
        raise RuntimeError(f"need at least 2 event markers, got {len(rel)}")
    # Recording starts in slow breathing. Marker 0 ends breathing / starts the
    # gesture. Marker 1 ends the gesture / starts jumping. Marker 2, if present,
    # is the end of the jumping bout.
    specs = [
        ("慢速呼吸", 0.0, rel[0], SKIP_S),
        ("上下挥手", rel[0], rel[1], SKIP_S),
        ("上下跳跃", rel[1], rel[2] if len(rel) > 2 else rel[1] + 60.0, SKIP_S),
    ]
    out = []
    for name, seg_lo, seg_hi, skip in specs:
        start = seg_lo + skip
        end = start + DURATION_S
        if end > seg_hi:
            raise RuntimeError(
                f"{name}: window [{start:.2f}, {end:.2f}] exceeds segment end {seg_hi:.2f}"
            )
        out.append(
            {
                "name": name,
                "start_s": start,
                "end_s": end,
                "segment_lo_s": seg_lo,
                "segment_hi_s": seg_hi,
            }
        )
    return out


def range_profiles(h_ft: np.ndarray, nfft: int = NFFT):
    """IFFT along frequency. h_ft shape (n_freq, n_slow).

    Returns fftshifted profiles and a monotonic two-way range axis in meters.
    Phase convention matches H(f) = exp(-j 2π f τ), τ = 2R/c.
    """
    window = np.hanning(h_ft.shape[0])
    spec = np.zeros((nfft, h_ft.shape[1]), dtype=np.complex128)
    spec[: h_ft.shape[0]] = h_ft * window[:, None]
    cir = np.fft.ifft(spec, axis=0)
    tau = np.fft.fftfreq(nfft, d=DF_HZ)
    rng = C_MPS * tau / 2.0
    return np.fft.fftshift(cir, axes=0), np.fft.fftshift(rng)


def self_check_range_axis() -> float:
    """A 5 m two-way reflector should peak near +5 m even with the 3 MHz gap."""
    rng_true = 5.0
    tau = 2.0 * rng_true / C_MPS
    k = np.arange(N_GRID, dtype=float)
    h = np.exp(-1j * 2.0 * np.pi * (k * DF_HZ) * tau)
    h[list(STRUCTURAL_GAP)] = 0.0
    cir, rng = range_profiles(h[:, None])
    peak = float(rng[int(np.argmax(np.abs(cir[:, 0])))])
    if abs(peak - rng_true) > 0.35:
        raise RuntimeError(f"range-axis self-check failed: peak {peak:.3f} m, expected {rng_true}")
    return peak


def slice_time(t_u: np.ndarray, h_u: np.ndarray, start: float, end: float):
    mask = (t_u >= start) & (t_u < end)
    if int(mask.sum()) < 16:
        raise RuntimeError(f"segment [{start}, {end}] has too few samples")
    return t_u[mask] - start, h_u[:, mask]


def db20(mag: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    ref = np.max(mag)
    if ref <= 0:
        return np.full_like(mag, floor_db, dtype=float)
    db = 20.0 * np.log10(mag / ref + 1e-12)
    return np.clip(db, floor_db, 0.0)


def dominant_bin(mag_mean: np.ndarray) -> int:
    return int(np.argmax(mag_mean))


def peak_to_median_db(mag: np.ndarray) -> float:
    med = float(np.median(mag))
    peak = float(np.max(mag))
    if med <= 0:
        return float("inf")
    return 20.0 * np.log10(peak / med)


def displacement_cm(z: np.ndarray, wavelength_m: float, fs: float) -> np.ndarray:
    """Two-way phase to displacement. Linear trend removed; 0.05 Hz highpass."""
    phi = np.unwrap(np.angle(z))
    x = np.arange(phi.size, dtype=float)
    slope, intercept = np.polyfit(x, phi, 1)
    phi = phi - (slope * x + intercept)
    disp = wavelength_m * phi / (4.0 * np.pi)
    wn = 0.05 / (fs / 2.0)
    b, a = signal.butter(2, wn, btype="highpass")
    disp = signal.filtfilt(b, a, disp)
    return disp * 100.0


def spectrum_db(x: np.ndarray, fs: float, fmax: float = 8.0):
    x = x - np.mean(x)
    win = np.hanning(x.size)
    spec = np.fft.rfft(x * win)
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mag = np.abs(spec)
    keep = freq <= fmax
    return freq[keep], mag[keep]


def band_power_ratio(freq: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
    p = mag**2
    total = float(np.sum(p[freq >= 0.05]))
    if total <= 0:
        return 0.0
    band = (freq >= lo) & (freq < hi)
    return float(np.sum(p[band]) / total)


def process_segment(t_rel: np.ndarray, h_seg: np.ndarray, fs: float) -> dict:
    h_mean = h_seg.mean(axis=1, keepdims=True)
    cir_raw, rng = range_profiles(h_seg)
    cir_static, _ = range_profiles(h_mean)
    cir_mti, _ = range_profiles(h_seg - h_mean)
    static_mag = np.abs(cir_static[:, 0])
    mti_energy = np.mean(np.abs(cir_mti) ** 2, axis=1)
    i_static = dominant_bin(static_mag)
    i_mti = dominant_bin(mti_energy)
    z_static_bin = cir_raw[i_static]
    lam = C_MPS / (F0_HZ + 0.5 * (N_GRID - 1) * DF_HZ)
    disp = displacement_cm(z_static_bin, lam, fs)
    dphi = np.diff(np.unwrap(np.angle(z_static_bin)))
    freq, mag = spectrum_db(disp, fs)
    # micro-Doppler of the static-peak bin after clutter removal
    z_mti = cir_mti[i_static]
    z_mti = z_mti - np.mean(z_mti)
    rd = np.fft.fftshift(np.fft.fft(cir_mti, axis=1), axes=1)
    fd = np.fft.fftshift(np.fft.fftfreq(cir_mti.shape[1], d=1.0 / fs))
    return {
        "t_rel": t_rel,
        "rng": rng,
        "cir_mti": cir_mti,
        "static_mag": static_mag,
        "mti_energy": mti_energy,
        "i_static": i_static,
        "i_mti": i_mti,
        "range_static_m": float(rng[i_static]),
        "range_mti_m": float(rng[i_mti]),
        "focus_db": peak_to_median_db(static_mag),
        "disp_cm": disp,
        "disp_rms_cm": float(np.sqrt(np.mean(disp**2))),
        "disp_pp_cm": float(np.percentile(disp, 99) - np.percentile(disp, 1)),
        "frac_dphi_near_nyquist": float(np.mean(np.abs(dphi) > 0.8 * np.pi)),
        "spec_freq": freq,
        "spec_mag": mag,
        "breath_band_ratio": band_power_ratio(freq, mag, 0.10, 0.40),
        "motion_band_ratio": band_power_ratio(freq, mag, 0.50, 5.0),
        "rd": rd,
        "fd": fd,
        "z_mti_rms": float(np.sqrt(np.mean(np.abs(z_mti) ** 2))),
    }


def crop_range(rng: np.ndarray, arr: np.ndarray, rmin: float, rmax: float):
    mask = (rng >= rmin) & (rng <= rmax)
    if arr.ndim == 1:
        return rng[mask], arr[mask]
    return rng[mask], arr[mask]


def shade_windows(ax, segs, color="C1") -> None:
    for seg in segs:
        ax.axvspan(seg["start_s"], seg["end_s"], color=color, alpha=0.18, lw=0)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    peak_check = self_check_range_axis()
    t_s, h, events = load_ipt(DATA_PATH)
    # recover absolute device time of first frame for event alignment
    with DATA_PATH.open(encoding="utf-8") as f:
        first_dev = None
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "frame":
                first_dev = float(rec["t_dev_ms"])
                break
    if first_dev is None:
        raise RuntimeError("no frame")

    t_u, h_u = interpolate_slow_time(t_s, h)
    fs = 1.0 / DT_S
    segs = segments_from_events(first_dev, events)
    processed = []
    for seg in segs:
        t_rel, h_seg = slice_time(t_u, h_u, seg["start_s"], seg["end_s"])
        result = process_segment(t_rel, h_seg, fs)
        result["meta"] = seg
        processed.append(result)
        print(
            f"{seg['name']}: t={seg['start_s']:.2f}-{seg['end_s']:.2f}s  "
            f"static={result['range_static_m']:.2f} m  mti={result['range_mti_m']:.2f} m  "
            f"focus={result['focus_db']:.1f} dB  disp_rms={result['disp_rms_cm']:.2f} cm  "
            f"pp={result['disp_pp_cm']:.2f} cm  "
            f"breath_band={result['breath_band_ratio']:.2f}  "
            f"motion_band={result['motion_band_ratio']:.2f}  "
            f"unwrap_stress={result['frac_dphi_near_nyquist']:.3f}"
        )

    # Common range window: union of static peaks, padded, at least ±6 m.
    centers = [p["range_static_m"] for p in processed]
    center = float(np.median(centers))
    # Main lobe is about 4 m wide (resolution ~2 m). Keep the display on that lobe.
    rmin, rmax = center - 6.0, center + 6.0

    # ----- timeline: frame-to-frame complex change, full record -----
    d = np.diff(h_u, axis=1)
    innov = np.sqrt(np.mean(np.abs(d) ** 2, axis=0))
    win = max(1, int(round(1.0 / DT_S)))
    kernel = np.ones(win) / win
    innov_s = np.convolve(innov, kernel, mode="same")
    t_innov = t_u[1:]

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.plot(t_innov, innov_s, color="#1f4e79", lw=1.0)
    for ev in events:
        te = (float(ev["nearest_t_dev_ms"]) - first_dev) / 1000.0
        ax.axvline(te, color="#c0392b", ls="--", lw=1.0)
    shade_windows(ax, segs)
    ax.set_xlabel("慢时间 (s)，相对记录起点")
    ax.set_ylabel("信道复帧差 RMS")
    ax.set_title("整段记录的慢时间变化强度（红线为 event，色带为作图窗口）")
    ax.set_xlim(t_u[0], t_u[-1])
    fig.tight_layout()
    timeline_path = FIG_DIR / "ipt_movement_timeline.png"
    fig.savefig(timeline_path)
    plt.close(fig)

    # ----- static range profiles -----
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for p, color in zip(processed, ("#1f4e79", "#e67e22", "#27ae60")):
        rr, mag = crop_range(p["rng"], p["static_mag"], rmin, rmax)
        db = db20(mag, floor_db=-40)
        ax.plot(rr, db, color=color, lw=1.4, label=p["meta"]["name"])
    ax.set_xlabel("双程距离 (m)")
    ax.set_ylabel("归一化幅度 (dB)")
    ax.set_title("各段平均信道频率响应的距离剖面（静态聚焦）")
    ax.legend()
    ax.set_ylim(-40, 1)
    fig.tight_layout()
    static_path = FIG_DIR / "ipt_movement_static_range.png"
    fig.savefig(static_path)
    plt.close(fig)

    # ----- range-time, shared scale and self-normalized -----
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharex=True, sharey=True)
    mti_crops = []
    for p in processed:
        rr, mag = crop_range(p["rng"], np.abs(p["cir_mti"]), rmin, rmax)
        mti_crops.append((rr, mag))
    global_max = max(float(np.max(mag)) for _, mag in mti_crops)
    im = None
    for col, (p, (rr, mag)) in enumerate(zip(processed, mti_crops)):
        db_shared = np.clip(20.0 * np.log10(mag / global_max + 1e-12), -28, 0)
        db_self = db20(mag, floor_db=-28)
        extent = [float(p["t_rel"][0]), float(p["t_rel"][-1]), float(rr[0]), float(rr[-1])]
        peak_hz = float(p["spec_freq"][int(np.argmax(p["spec_mag"]))])
        for row, db in enumerate((db_shared, db_self)):
            ax = axes[row, col]
            im = ax.imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=-28,
                vmax=0,
                interpolation="bilinear",
            )
            ax.axhline(p["range_static_m"], color="cyan", lw=0.7, ls="--", alpha=0.85)
            if row == 1:
                ax.set_xlabel("慢时间 (s)")
        axes[0, col].set_title(f"{p['meta']['name']}\n位移谱峰 {peak_hz:.2f} Hz")
    axes[0, 0].set_ylabel("双程距离 (m)\n上排：三幅共用色标")
    axes[1, 0].set_ylabel("双程距离 (m)\n下排：各幅自行归一化")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="相对参考 (dB)")
    fig.suptitle(
        f"杂波抑制后的距离-时间图（窗 {DURATION_S:.0f} s，fs = {fs:.2f} Hz，青线为静态主峰）",
        y=0.98,
    )
    fig.subplots_adjust(top=0.88, wspace=0.18, hspace=0.28)
    rti_path = FIG_DIR / "ipt_movement_rti.png"
    fig.savefig(rti_path)
    plt.close(fig)

    # ----- range-Doppler -----
    fd_lim = 2.5
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    rd_crops = []
    for p in processed:
        rr, rd = crop_range(p["rng"], np.abs(p["rd"]), rmin, rmax)
        keep = (p["fd"] >= -fd_lim) & (p["fd"] <= fd_lim)
        rd_crops.append((rr, p["fd"][keep], rd[:, keep]))
    rd_max = max(float(np.max(rd)) for _, _, rd in rd_crops)
    im = None
    for ax, p, (rr, fd, rd) in zip(axes, processed, rd_crops):
        db = np.clip(20.0 * np.log10(rd / rd_max + 1e-12), -28, 0)
        extent = [float(fd[0]), float(fd[-1]), float(rr[0]), float(rr[-1])]
        im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=-28,
            vmax=0,
            interpolation="nearest",
        )
        ax.axhline(p["range_static_m"], color="cyan", lw=0.7, ls="--", alpha=0.85)
        ax.axvline(0.0, color="w", lw=0.4, alpha=0.5)
        ax.set_title(p["meta"]["name"])
        ax.set_xlabel("多普勒 (Hz)")
    axes[0].set_ylabel("双程距离 (m)")
    fig.subplots_adjust(left=0.07, right=0.90, top=0.82, bottom=0.16, wspace=0.18)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.62])
    fig.colorbar(im, cax=cax, label="三幅共用 (dB)")
    fig.suptitle("距离-多普勒图（±2.5 Hz，三幅共用色标，青线为静态主峰）", y=0.96)
    rd_path = FIG_DIR / "ipt_movement_range_doppler.png"
    fig.savefig(rd_path)
    plt.close(fig)

    # ----- displacement + spectrum -----
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.4))
    colors = ("#1f4e79", "#e67e22", "#27ae60")
    for p, color in zip(processed, colors):
        axes[0].plot(p["t_rel"], p["disp_cm"], color=color, lw=1.0, label=p["meta"]["name"])
        spec_db = 20.0 * np.log10(p["spec_mag"] / (np.max(p["spec_mag"]) + 1e-12) + 1e-12)
        axes[1].plot(p["spec_freq"], spec_db, color=color, lw=1.3, label=p["meta"]["name"])
    axes[0].set_ylabel("主距离单元位移 (cm)")
    axes[0].set_xlabel("慢时间 (s)，各段各自从 0 起算")
    axes[0].set_title("静态主峰距离单元的双程相位解调（去趋势、0.05 Hz 高通）")
    axes[0].legend(loc="upper right")
    axes[1].axvspan(0.10, 0.40, color="#1f4e79", alpha=0.08)
    axes[1].set_xlim(0, 8)
    axes[1].set_ylim(-40, 1)
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("各段自行归一化 (dB)")
    axes[1].set_title("位移谱（浅色带为 0.10–0.40 Hz）")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    phase_path = FIG_DIR / "ipt_movement_peakbin.png"
    fig.savefig(phase_path)
    plt.close(fig)

    summary = {
        "data": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
        "n_frames_raw": int(h.shape[1]),
        "duration_s": float(t_s[-1] - t_s[0]),
        "dt_nominal_s": DT_S,
        "fs_hz": fs,
        "nyquist_hz": fs / 2.0,
        "n_channels_grid": N_GRID,
        "structural_gap": sorted(STRUCTURAL_GAP),
        "tones_present_in_log": int(np.sum(np.any(np.isfinite(h.real), axis=1))),
        "median_tones_per_frame": float(np.median(np.sum(np.isfinite(h.real), axis=0))),
        "range_resolution_m": C_MPS / (2.0 * (N_GRID - 1) * DF_HZ),
        "unambiguous_range_m": C_MPS / (2.0 * DF_HZ),
        "wavelength_m": C_MPS / (F0_HZ + 0.5 * (N_GRID - 1) * DF_HZ),
        "self_check_peak_m": peak_check,
        "range_window_m": [rmin, rmax],
        "events_rel_s": [
            (float(ev["nearest_t_dev_ms"]) - first_dev) / 1000.0 for ev in events
        ],
        "segments": [],
        "figures": [
            str(timeline_path.relative_to(ROOT)).replace("\\", "/"),
            str(static_path.relative_to(ROOT)).replace("\\", "/"),
            str(rti_path.relative_to(ROOT)).replace("\\", "/"),
            str(rd_path.relative_to(ROOT)).replace("\\", "/"),
            str(phase_path.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    for p in processed:
        i_peak = int(np.argmax(p["spec_mag"]))
        summary["segments"].append(
            {
                "name": p["meta"]["name"],
                "start_s": p["meta"]["start_s"],
                "end_s": p["meta"]["end_s"],
                "segment_lo_s": p["meta"]["segment_lo_s"],
                "segment_hi_s": p["meta"]["segment_hi_s"],
                "range_static_m": p["range_static_m"],
                "range_mti_m": p["range_mti_m"],
                "focus_db": p["focus_db"],
                "mti_bin_rms": p["z_mti_rms"],
                "disp_rms_cm": p["disp_rms_cm"],
                "disp_pp_cm": p["disp_pp_cm"],
                "frac_dphi_near_nyquist": p["frac_dphi_near_nyquist"],
                "spec_peak_hz": float(p["spec_freq"][i_peak]),
                "breath_band_ratio_0.10_0.40": p["breath_band_ratio"],
                "motion_band_ratio_0.50_5": p["motion_band_ratio"],
            }
        )
    summary_path = REPORT_DIR / "ipt_range_time_movement_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("self-check peak m", round(peak_check, 3))
    print("wrote", summary_path)
    for fig_path in summary["figures"]:
        print("figure", fig_path)


if __name__ == "__main__":
    main()
