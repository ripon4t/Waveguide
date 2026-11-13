"""
Full-vector 3D FDTD check (Ansys Lumerical FDTD) of the best taper found by the varFDTD sweep,
compared with a linear taper (m = 1). Uses lumapi.

    python taper_check_fdtd3d.py [--m 1.45] [--hide]

If --m is omitted, the best m from ../data/sweep_varfdtd.csv is used.
Output: ../data/fdtd3d_check.csv  (columns: label, m, T_total, T_TE0)
Note: 3D FDTD is much slower than varFDTD; raise --mesh-accuracy only if you need to.
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
MONITOR_X = g.L_TAPER + g.LEAD - 0.3e-6


def best_m_from_csv():
    path = os.path.join(DATA, "sweep_varfdtd.csv")
    best, m_best = -1.0, 1.0
    with open(path) as f:
        for row in csv.DictReader(f):
            t0 = float(row["T_TE0"]); t = float(row["T_total"])
            v = t0 if np.isfinite(t0) else t
            if v > best:
                best, m_best = v, float(row["m"])
    return m_best


def build(sim, mesh_accuracy):
    sim.newproject()
    sim.addfdtd()
    set_props(sim, {
        "x min": -g.LEAD - 0.3e-6, "x max": g.L_TAPER + g.LEAD + 0.5e-6,
        "y min": -2.5e-6, "y max": 2.5e-6,
        "z min": -1.0e-6, "z max": g.H_SI + 1.0e-6,
        "background material": "SiO2 (Glass) - Palik",
        "mesh accuracy": mesh_accuracy,
    }, "FDTD region")

    sim.addpoly()
    set_props(sim, {
        "name": "taper", "x": 0, "y": 0, "z min": 0, "z max": g.H_SI,
        "material": "Si (Silicon) - Palik",
        "vertices": g.polygon_vertices(1.0),
    }, "taper")

    sim.addmode()
    set_props(sim, {
        "name": "source", "injection axis": "x-axis", "direction": "Forward",
        "x": -g.LEAD + 0.5e-6, "y": 0, "y span": 2.5e-6,
        "z": g.H_SI / 2, "z span": 1.8e-6,
        "center wavelength": g.WAVELENGTH, "wavelength span": 0,
    }, "source")
    try_values(sim, "mode selection", ["fundamental TE mode", "fundamental mode"], "source")

    sim.addpower()
    set_props(sim, {
        "name": "out", "monitor type": "2D X-normal",
        "x": MONITOR_X, "y": 0, "y span": 4.6e-6, "z": g.H_SI / 2, "z span": 1.8e-6,
        "override global monitor settings": 1, "frequency points": 1,
    }, "power monitor")

    try:
        sim.addmodeexpansion()
        set_props(sim, {
            "name": "exp_out", "monitor type": "2D X-normal",
            "x": MONITOR_X, "y": 0, "y span": 4.6e-6, "z": g.H_SI / 2, "z span": 1.8e-6,
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
    ap.add_argument("--m", type=float, default=None, help="optimized exponent (default: best of the sweep)")
    ap.add_argument("--mesh-accuracy", type=int, default=2)
    ap.add_argument("--hide", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    m_opt = args.m if args.m is not None else best_m_from_csv()

    rows = []
    with lumapi.FDTD(hide=args.hide) as sim:
        build(sim, args.mesh_accuracy)
        sim.save(os.path.join(DATA, "taper_fdtd3d.lms"))
        for label, m in (("linear", 1.0), ("optimized", m_opt)):
            t, t0 = run_one(sim, m)
            rows.append((label, m, t, t0)); print(f"{label:10s} m={m:.3f}  T={t:.4f}  T_TE0={t0:.4f}")
    with open(os.path.join(DATA, "fdtd3d_check.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "m", "T_total", "T_TE0"]); w.writerows(rows)
    print("wrote data/fdtd3d_check.csv")


if __name__ == "__main__":
    main()
