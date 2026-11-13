"""
Figures for the Lumerical taper study.

Reads  data/sweep_varfdtd.csv   (from lumerical/taper_sweep_varfdtd.py)
and optionally data/fdtd3d_check.csv (from lumerical/taper_check_fdtd3d.py)
and writes figures to images/.

    python plot_results.py
    python plot_results.py --selftest     # runs on synthetic data, writes to a temp folder
"""
import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lumerical"))
import taper_geometry as g  # noqa: E402

BLUE, ORANGE, RED = "#2a6fbb", "#e07a1f", "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "images"))
    args = ap.parse_args()

    if args.selftest:
        import tempfile
        args.data = tempfile.mkdtemp(); args.out = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        with open(os.path.join(args.data, "sweep_varfdtd.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["stage", "m", "T_total", "T_TE0"])
            for m in np.linspace(0.2, 4, 15):
                v = 0.98 - 0.02 * (m - 1.4) ** 2 + rng.normal(0, 1e-3); w.writerow(["coarse", m, v, v - 0.01])
            for m in np.linspace(1.0, 1.8, 13):
                v = 0.98 - 0.02 * (m - 1.4) ** 2 + rng.normal(0, 1e-3); w.writerow(["fine", m, v, v - 0.01])
        with open(os.path.join(args.data, "fdtd3d_check.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["label", "m", "T_total", "T_TE0"])
            w.writerow(["linear", 1.0, 0.95, 0.94]); w.writerow(["optimized", 1.4, 0.98, 0.97])
    os.makedirs(args.out, exist_ok=True)

    rows = read_csv(os.path.join(args.data, "sweep_varfdtd.csv"))
    stage = np.array([r["stage"] for r in rows])
    m = np.array([float(r["m"]) for r in rows])
    t_tot = np.array([float(r["T_total"]) for r in rows])
    t_te0 = np.array([float(r["T_TE0"]) for r in rows])
    use_te0 = np.isfinite(t_te0).all()
    fom = t_te0 if use_te0 else t_tot
    ylab = "Transmission into TE$_0$ of the output waveguide" if use_te0 else "Transmitted power into the output waveguide"
    k = int(np.argmax(fom)); m_opt, f_opt = m[k], fom[k]
    lin = np.argmin(np.abs(m - 1.0))

    # Fig 1: shape family (pure geometry)
    x = np.linspace(0, g.L_TAPER, 400) * 1e6
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for mm, col in zip([0.5, 1.0, 2.0, 3.0], ["#8e44ad", BLUE, ORANGE, "#16a085"]):
        w = g.width(x * 1e-6, mm) * 1e6
        ax.plot(x, w / 2, color=col, label=f"m = {mm:g}"); ax.plot(x, -w / 2, color=col)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("±w/2 (µm)"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Power-law taper family simulated in Lumerical")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "01_taper_shapes.png"), dpi=200); plt.close(fig)

    # Fig 2: coarse sweep
    c = stage == "coarse"
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(m[c], fom[c], "o-", color=BLUE); ax.set_xlabel("Shape exponent m"); ax.set_ylabel(ylab)
    ax.set_title("varFDTD: coarse sweep of the exponent")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "02_varfdtd_sweep_coarse.png"), dpi=200); plt.close(fig)

    # Fig 3: fine sweep
    fi = stage == "fine"
    order = np.argsort(m[fi])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(m[fi][order], fom[fi][order], "o-", color=BLUE, ms=4)
    ax.plot(m_opt, f_opt, "*", color=RED, ms=16, label=f"best m ≈ {m_opt:.2f}  ({f_opt:.4f})")
    if abs(m[lin] - 1.0) < 0.05:
        ax.axhline(fom[lin], color="gray", ls="--", lw=1, label=f"≈ linear (m = {m[lin]:.2f}): {fom[lin]:.4f}")
    ax.set_xlabel("Shape exponent m"); ax.set_ylabel(ylab); ax.legend(); ax.set_title("varFDTD: fine sweep around the optimum")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "03_varfdtd_sweep_fine.png"), dpi=200); plt.close(fig)

    # Fig 4: 3D FDTD cross-check (optional)
    p3 = os.path.join(args.data, "fdtd3d_check.csv")
    if os.path.exists(p3):
        r3 = read_csv(p3)
        labels = [r["label"] for r in r3]
        v3 = [float(r["T_TE0"]) if np.isfinite(float(r["T_TE0"])) else float(r["T_total"]) for r in r3]
        # varFDTD values at the same m
        v2 = []
        for r in r3:
            j = np.argmin(np.abs(m - float(r["m"]))); v2.append(fom[j])
        xx = np.arange(len(labels)); wb = 0.38
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(xx - wb / 2, v2, wb, color=BLUE, label="2.5D varFDTD")
        ax.bar(xx + wb / 2, v3, wb, color=ORANGE, label="3D FDTD")
        ax.set_xticks(xx); ax.set_xticklabels([f"{l}\n(m = {float(r['m']):.2f})" for l, r in zip(labels, r3)])
        ax.set_ylim(min(v2 + v3) - 0.03, 1.005); ax.set_ylabel(ylab.split(" of ")[0]); ax.legend()
        ax.set_title("varFDTD vs 3D FDTD")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "04_varfdtd_vs_fdtd3d.png"), dpi=200); plt.close(fig)

    print(f"best m = {m_opt:.3f}, figure of merit = {f_opt:.4f}")
    if args.selftest:
        print("selftest ok; files:", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
