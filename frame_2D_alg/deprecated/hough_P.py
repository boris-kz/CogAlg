import numpy as np
THETA_RES = 360         # bins per 180 degrees
UNIT_RHO_RES = 20       # bins per 1 unit of rho
RHO_TOL_BINS = 10       # tolerance for rho (number of bins)
THETA_TOL_BINS = 10     # tolerance for theta (number of bins)
THETA_TOL = THETA_TOL_BINS / THETA_RES * np.pi

def hough_check(rt_olp__, x, y, t):
    new_rt__ = np.zeros_like(rt_olp__, dtype=bool)

    thetas = np.linspace(t - THETA_TOL, t + THETA_TOL, 2 * THETA_TOL_BINS + 1)
    thetas[thetas < 0] += np.pi
    thetas[thetas > np.pi] -= np.pi
    rhos = np.abs(np.hypot(y, x)*np.cos(thetas - np.arctan2(y, x)))         # Hough transform
    rho_bins = np.round(rhos * UNIT_RHO_RES).astype(int)
    theta_bins = np.round(thetas * THETA_RES / np.pi).astype(int)

    for rho_bins_tolerance in range(-RHO_TOL_BINS, RHO_TOL_BINS + 1):
        new_rt__[theta_bins % THETA_RES, rho_bins + rho_bins_tolerance] = True

    return new_rt__ & rt_olp__

def new_rt_olp_array(shape):
    rho_res = int(np.ceil(UNIT_RHO_RES * np.hypot(*shape)))
    return np.ones((THETA_RES, rho_res), dtype=bool)

def mean_theta_olp(rt_olp__):
    return rt_olp__.nonzero()[0].mean() * np.pi / THETA_RES