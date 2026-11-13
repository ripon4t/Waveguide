"""Shared taper definition used by the varFDTD sweep and the 3D FDTD check."""
import numpy as np

W_IN = 0.5e-6        # input waveguide width (m)
W_OUT = 3.0e-6       # output waveguide width (m)
L_TAPER = 10.0e-6    # taper length (m)
LEAD = 2.0e-6        # straight lead-in / lead-out length (m)
H_SI = 0.22e-6       # silicon thickness (m)
N_SIO2 = 1.444
WAVELENGTH = 1.55e-6


def width(x, m, L=L_TAPER):
    """w(x) = w_out - (w_out - w_in) * (1 - x/L)^m ,  0 <= x <= L."""
    return W_OUT - (W_OUT - W_IN) * (1.0 - np.asarray(x) / L) ** m


def polygon_vertices(m, n_points=120, L=L_TAPER):
    """Closed outline (N x 2, metres) of lead-in + taper + lead-out, centred on y = 0."""
    x = np.linspace(0.0, L, n_points)
    w = width(x, m, L)
    upper = np.column_stack([np.r_[-LEAD, x, L + LEAD], np.r_[W_IN / 2, w / 2, W_OUT / 2]])
    lower = upper[::-1].copy()
    lower[:, 1] *= -1
    return np.vstack([upper, lower])
