"""
Power-law shape optimization of an SOI waveguide taper using a 2D
(effective-index) eigenmode-expansion (EME) model.

    w(x) = w_out - (w_out - w_in) * (1 - x/L)**m ,   0 <= x <= L

m = 1 is a linear taper, m < 1 widens fastest near the wide end,
m > 1 widens fastest near the narrow end.

Run:  python taper_eme.py        (writes figures to ../images, results to ../results)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
RES = os.path.join(HERE, "results")
os.makedirs(IMG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ----------------------------------------------------------------------------
# Parameters (effective-index model of a 220 nm SOI strip, TE polarization)
# ----------------------------------------------------------------------------
WAVELENGTH = 1.55      # um
N_CORE = 2.85          # slab effective index of 220 nm Si (TE), approx.
N_CLAD = 1.444         # SiO2
W_IN = 0.5             # um, input waveguide width
W_OUT = 3.0            # um, output (wide) waveguide width
L_TAPER = 10.0         # um, taper length (default)
HALF_DOMAIN = 5.0      # um, half-width of transverse computational box
DY = 0.025             # um, transverse grid step
N_MODES = 70           # modes kept in every cross-section
N_WIDTH_TABLE = 251    # number of tabulated widths between W_IN and W_OUT
K0 = 2 * np.pi / WAVELENGTH

y = np.arange(-HALF_DOMAIN + DY / 2, HALF_DOMAIN, DY)   # cell centres
NY = y.size


# ----------------------------------------------------------------------------
# Cross-section mode solver (scalar TE, 1D transverse profile)
# ----------------------------------------------------------------------------
def index_squared(width):
    """n^2 on the grid with sub-pixel (fill-fraction) smoothing at the core edges."""
    lo = np.clip(y + DY / 2, -width / 2, width / 2)
    hi = np.clip(y - DY / 2, -width / 2, width / 2)
    fill = (lo - hi) / DY
    return fill * N_CORE**2 + (1 - fill) * N_CLAD**2


def solve_modes(width, n_modes=N_MODES):
    n2 = index_squared(width)
    d = 2.0 / DY**2 - K0**2 * n2
    e = -np.ones(NY - 1) / DY**2
    ev, vec = eigh_tridiagonal(d, e, select="i", select_range=(0, n_modes - 1))
    beta2 = -ev
    keep = beta2 > 0                       # propagating modes only
    return np.sqrt(beta2[keep]), vec[:, keep]


WIDTHS = np.linspace(W_IN, W_OUT, N_WIDTH_TABLE)
_TABLE = None


def mode_table():
    global _TABLE
    if _TABLE is None:
        betas, vecs = [], []
        kmin = N_MODES
        for w in WIDTHS:
            b, v = solve_modes(w)
            betas.append(b); vecs.append(v)
            kmin = min(kmin, b.size)
        _TABLE = (np.array([b[:kmin] for b in betas]),
                  np.array([v[:, :kmin] for v in vecs]))
    return _TABLE


# ----------------------------------------------------------------------------
# Taper geometry
# ----------------------------------------------------------------------------
def taper_width(x, m, L=L_TAPER):
    return W_OUT - (W_OUT - W_IN) * (1.0 - x / L) ** m


def width_index(w):
    return np.clip(np.rint((w - W_IN) / (W_OUT - W_IN) * (N_WIDTH_TABLE - 1)).astype(int),
                   0, N_WIDTH_TABLE - 1)


# ----------------------------------------------------------------------------
# EME propagation through a staircase of uniform sections
# ----------------------------------------------------------------------------
def propagate(m, L=L_TAPER, n_sections=300, return_field=False):
    betas, vecs = mode_table()
    dz = L / n_sections
    xc = (np.arange(n_sections) + 0.5) * dz
    idx = width_index(taper_width(xc, m, L))

    a = vecs[idx[0]].T @ vecs[0][:, 0]               # input TE0 of w_in, in first section's basis
    fields = []
    for i in range(n_sections):
        if return_field:
            fields.append((xc[i], vecs[idx[i]] @ a))
        a = a * np.exp(1j * betas[idx[i]] * dz)      # propagate through section i
        nxt = vecs[idx[i + 1]] if i + 1 < n_sections else vecs[-1]
        a = nxt.T @ (vecs[idx[i]] @ a)               # project onto next basis
    # a is now expressed in the output waveguide modes (table[-1])
    # Remove the output waveguide's own propagation phase freedom: only |a|^2 matters.
    power = np.abs(a) ** 2
    if return_field:
        return power, fields
    return power


def even_mode_ids(n=5):
    """Indices of the first n symmetric modes of the output waveguide."""
    _, vecs = mode_table()
    v = vecs[-1]
    sym = np.sum(v * v[::-1, :], axis=0)           # +1 even, -1 odd
    return np.where(sym > 0)[0][:n]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE, RED = "#2a6fbb", "#e07a1f", "#c0392b"


def main():
    mode_table()
    # ---- coarse and fine sweeps ----
    m_coarse = np.concatenate([np.linspace(0.1, 1.0, 10, endpoint=False), np.linspace(1.0, 4.0, 25)])
    T_coarse = np.array([propagate(m)[0] for m in m_coarse])
    m_fine = np.linspace(0.6, 2.0, 57)
    T_fine = np.array([propagate(m)[0] for m in m_fine])
    m_opt = m_fine[np.argmax(T_fine)]
    T_opt = T_fine.max()
    T_lin = propagate(1.0)[0]

    # ---- Fig 1: shape family ----
    x = np.linspace(0, L_TAPER, 400)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m, col in zip([0.5, 1.0, 2.0, 3.0], ["#8e44ad", BLUE, ORANGE, "#16a085"]):
        w = taper_width(x, m)
        ax.plot(x, w / 2, color=col, label=f"m = {m:g}")
        ax.plot(x, -w / 2, color=col)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("Taper half-width profile, ±w/2 (µm)")
    ax.set_title("Power-law taper family  w(x) = w$_{out}$ − (w$_{out}$−w$_{in}$)(1 − x/L)$^m$")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); fig.savefig(f"{IMG}/01_taper_shapes.png", dpi=200); plt.close(fig)

    # ---- Fig 2: coarse sweep ----
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(m_coarse, T_coarse, "o-", color=BLUE)
    ax.set_xlabel("Shape exponent m"); ax.set_ylabel("Transmission into TE$_0$ of output waveguide")
    ax.set_title("Coarse sweep of the shape exponent")
    ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(f"{IMG}/02_sweep_coarse.png", dpi=200); plt.close(fig)

    # ---- Fig 3: fine sweep ----
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(m_fine, T_fine, "o-", color=BLUE, ms=3.5)
    ax.plot(m_opt, T_opt, "*", color=RED, ms=16, label=f"optimum m ≈ {m_opt:.2f}  (T = {T_opt:.4f})")
    ax.axhline(T_lin, ls="--", color="gray", lw=1, label=f"linear taper (m = 1): T = {T_lin:.4f}")
    ax.set_xlabel("Shape exponent m"); ax.set_ylabel("Transmission into TE$_0$")
    ax.set_title("Fine sweep around the optimum"); ax.legend(loc="lower center")
    fig.tight_layout(); fig.savefig(f"{IMG}/03_sweep_fine.png", dpi=200); plt.close(fig)

    # ---- Fig 4: field maps (linear vs optimum) ----
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    for ax, m, title in zip(axs, [1.0, m_opt], ["Linear taper (m = 1)", f"Optimized taper (m = {m_opt:.2f})"]):
        p, fields = propagate(m, return_field=True)
        xs = np.array([f[0] for f in fields])
        I = np.array([np.abs(f[1]) ** 2 for f in fields]).T
        I = I / I.max()
        im = ax.pcolormesh(xs, y, I, cmap="inferno", shading="auto", vmin=0, vmax=1)
        w = taper_width(xs, m)
        ax.plot(xs, w / 2, "w--", lw=0.8); ax.plot(xs, -w / 2, "w--", lw=0.8)
        ax.set_ylim(-3, 3); ax.set_ylabel("y (µm)")
        ax.set_title(f"{title}:  T$_{{TE0}}$ = {p[0]:.3f}"); ax.grid(False)
    axs[1].set_xlabel("x (µm)")
    fig.tight_layout(); fig.savefig(f"{IMG}/04_field_maps.png", dpi=200); plt.close(fig)

    # ---- Fig 5: modal content at output ----
    ids = even_mode_ids(5)
    p_opt = propagate(m_opt)
    p_lin = propagate(1.0)
    labels = [f"TE$_{{{i}}}$" for i in ids]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xpos = np.arange(len(ids)); wbar = 0.38
    ax.bar(xpos - wbar / 2, p_lin[ids], wbar, color="gray", label="linear (m = 1)")
    ax.bar(xpos + wbar / 2, p_opt[ids], wbar, color=BLUE, label=f"optimized (m = {m_opt:.2f})")
    ax.set_yscale("log"); ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("Power in output mode (log)"); ax.set_xlabel("Even TE modes of the output waveguide")
    ax.set_title("Where does the power go?"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{IMG}/05_output_modes.png", dpi=200); plt.close(fig)

    # ---- Fig 6: transmission vs taper length ----
    Ls = np.linspace(3, 30, 19)
    T_L_lin = [propagate(1.0, L=l, n_sections=int(max(150, l * 30)))[0] for l in Ls]
    T_L_opt = [propagate(m_opt, L=l, n_sections=int(max(150, l * 30)))[0] for l in Ls]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(Ls, T_L_lin, "s-", color="gray", label="linear (m = 1)")
    ax.plot(Ls, T_L_opt, "o-", color=BLUE, label=f"optimized shape (m = {m_opt:.2f})")
    ax.set_xlabel("Taper length L (µm)"); ax.set_ylabel("Transmission into TE$_0$")
    ax.set_title("Length needed for a given transmission"); ax.legend(); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(f"{IMG}/06_length_scan.png", dpi=200); plt.close(fig)

    # ---- Fig 7: numerical convergence ----
    secs = np.array([20, 40, 80, 150, 300, 600, 1000])
    T_conv = [propagate(m_opt, n_sections=s)[0] for s in secs]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.semilogx(secs, T_conv, "o-", color=BLUE)
    ax.set_xlabel("Number of uniform sections"); ax.set_ylabel("T$_{TE0}$")
    ax.set_title("Convergence of the staircase approximation")
    fig.tight_layout(); fig.savefig(f"{IMG}/07_convergence.png", dpi=200); plt.close(fig)

    # ---- Sanity checks and summary ----
    total_power = propagate(m_opt).sum()
    summary = {
        "w_in_um": W_IN, "w_out_um": W_OUT, "L_um": L_TAPER,
        "n_core": N_CORE, "n_clad": N_CLAD, "wavelength_um": WAVELENGTH,
        "m_opt": float(m_opt), "T_opt": float(T_opt), "T_linear": float(T_lin),
        "total_power_captured_opt": float(total_power),
        "even_mode_power_opt": {int(i): float(p_opt[i]) for i in ids},
        "convergence": {int(s): float(t) for s, t in zip(secs, T_conv)},
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savetxt(os.path.join(RES, "sweep_fine.csv"), np.c_[m_fine, T_fine],
               delimiter=",", header="m,T_TE0", comments="")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
