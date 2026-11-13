"""
Shape sweep of an SOI taper with Ansys Lumerical MODE (2.5D varFDTD), driven by lumapi.

Taper:  w(x) = w_out - (w_out - w_in) * (1 - x/L)^m
Stage 1: coarse sweep of the exponent m
Stage 2: fine sweep around the best coarse value
For each m the script records the power transmitted into the output waveguide and,
when the mode-expansion monitor is available, the power in its fundamental mode.

    python taper_sweep_varfdtd.py [--hide]

Output: ../data/sweep_varfdtd.csv   (columns: stage, m, T_total, T_TE0)
"""
import argparse
import csv
import os
import numpy as np
from lum_utils import import_lumapi, set_props, try_values
import taper_geometry as g

lumapi = import_lumapi()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

X_MIN = -g.LEAD - 0.5e-6
X_MAX = g.L_TAPER + g.LEAD + 0.5e-6
Y_HALF = 2.5e-6
MONITOR_X = g.L_TAPER + g.LEAD - 0.3e-6


def build(sim):
    sim.newproject()
    sim.addvarfdtd()
    set_props(sim, {
        "name": "varFDTD",
        "x min": X_MIN, "x max": X_MAX,
        "y": 0, "y span": 2 * Y_HALF,
        "z": g.H_SI / 2, "z span": 1.0e-6,
        "background index": g.N_SIO2,
        "simulation time": 1500e-15,
    }, "varFDTD region")
    try:
        sim.setactivesolver("varFDTD")
    except Exception:
        pass

    sim.addpoly()
    set_props(sim, {
        "name": "taper", "x": 0, "y": 0, "z": g.H_SI / 2, "z span": g.H_SI,
        "material": "Si (Silicon) - Palik",
        "vertices": g.polygon_vertices(1.0),
    }, "taper")

    sim.addmodesource()
    set_props(sim, {
        "name": "source", "injection axis": "x",
        "x": -g.LEAD + 0.5e-6, "y": 0, "y span": 2.5e-6,
        "wavelength start": g.WAVELENGTH, "wavelength stop": g.WAVELENGTH,
    }, "source")
    try_values(sim, "mode selection", ["fundamental TE mode", "fundamental mode"], "source")

    sim.addpower()
    set_props(sim, {
        "name": "out", "monitor type": "Linear Y",
        "x": MONITOR_X, "y": 0, "y span": 2 * Y_HALF - 0.4e-6,
        "override global monitor settings": 1, "frequency points": 1,
    }, "power monitor")

    # Mode-expansion monitor on the output waveguide (optional: failures are reported, not fatal)
    try:
        sim.addmodeexpansion()
        set_props(sim, {
            "name": "exp_out", "monitor type": "Linear Y",
            "x": MONITOR_X, "y": 0, "y span": 2 * Y_HALF - 0.4e-6,
            "override global monitor settings": 1, "frequency points": 1,
        }, "expansion monitor")
        try_values(sim, "mode selection", ["fundamental TE mode", "fundamental mode"], "expansion monitor")
        sim.setexpansion("out", "exp_out")
    except Exception as exc:
        print(f"[warning] mode-expansion monitor not configured ({exc}); T_TE0 will be NaN")


def run_one(sim, m):
    sim.switchtolayout()
    sim.setnamed("taper", "vertices", g.polygon_vertices(m))
    sim.run()
    t_total = float(np.abs(np.array(sim.transmission("out"))).flatten()[0])
    t_te0 = float("nan")
    try:
        res = sim.getresult("exp_out", "expansion for out")
        t_te0 = float(np.abs(np.array(res["T_forward"])).flatten()[0])
    except Exception:
        pass
    return t_total, t_te0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hide", action="store_true")
    ap.add_argument("--m-min", type=float, default=0.2)
    ap.add_argument("--m-max", type=float, default=4.0)
    ap.add_argument("--n-coarse", type=int, default=15)
    ap.add_argument("--n-fine", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    rows = []
    with lumapi.MODE(hide=args.hide) as sim:
        build(sim)
        sim.save(os.path.join(DATA, "taper_varfdtd.lms"))

        coarse = np.linspace(args.m_min, args.m_max, args.n_coarse)
        for m in coarse:
            t, t0 = run_one(sim, m)
            rows.append(("coarse", m, t, t0)); print(f"coarse m={m:.3f}  T={t:.4f}  T_TE0={t0:.4f}")

        fom = np.array([r[3] if np.isfinite(r[3]) else r[2] for r in rows])
        m_best = coarse[int(np.argmax(fom))]
        step = coarse[1] - coarse[0]
        for m in np.linspace(max(0.05, m_best - step), m_best + step, args.n_fine):
            t, t0 = run_one(sim, m)
            rows.append(("fine", m, t, t0)); print(f"fine   m={m:.3f}  T={t:.4f}  T_TE0={t0:.4f}")

    with open(os.path.join(DATA, "sweep_varfdtd.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "m", "T_total", "T_TE0"]); w.writerows(rows)
    print("wrote data/sweep_varfdtd.csv")


if __name__ == "__main__":
    main()
