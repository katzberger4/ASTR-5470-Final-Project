import numpy as np
from .utils import nfw_acc

class ProgenitorPotential:
    """
    NFW progenitor with a simple mass-loss law and a user-supplied orbit.
    """

    def __init__(self, M0, r_s, c, t_start, t_end, orbit_fn):
        """
        M0: initial mass (Msun)
        r_s: scale radius (kpc)
        c: concentration
        t_start, t_end: Gyr, for mass-loss interval
        orbit_fn: callable t -> (3,) progenitor position in kpc
        """
        self.M0 = M0
        self.r_s = r_s
        self.c = c
        self.t_start = t_start
        self.t_end = t_end
        self.orbit_fn = orbit_fn

    def mass(self, t):
        if t <= self.t_start:
            return self.M0
        if t >= self.t_end:
            return 0.0
        frac = 1.0 - (t - self.t_start) / (self.t_end - self.t_start)
        return self.M0 * max(0.0, frac)

    def center(self, t):
        return np.asarray(self.orbit_fn(t), dtype=float)

    def acceleration(self, positions, t):
        positions = np.asarray(positions, dtype=float)
        r_prog = self.center(t)
        rel = positions - r_prog[None, :]
        M = self.mass(t)
        if M <= 0.0:
            return np.zeros_like(positions)
        return nfw_acc(rel, M_vir=M, r_s=self.r_s, c=self.c)
