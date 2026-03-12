import numpy as np

class LeapfrogIntegrator:
    """
    Kick–drift–kick leapfrog.
    Positions in kpc, velocities in km/s, dt in Myr, time in Gyr.
    """

    def __init__(self):
        # 1 km/s * 1 Myr ≈ 0.0010227 kpc
        self.kpc_per_kms_Myr = 0.0010227121650537077

    def step(self, particles, t, dt, acceleration_fn):
        """
        particles: ParticleSet
        t: current time (Gyr)
        dt: timestep (Myr)
        acceleration_fn: a(positions, t) -> (N, 3)
        """
        pos = particles.positions
        vel = particles.velocities

        # First half-kick
        a = acceleration_fn(pos, t)
        vel_half = vel + 0.5 * dt * a  # km/s

        # Drift
        pos_new = pos + vel_half * dt * self.kpc_per_kms_Myr

        # Second half-kick
        t_new = t + dt / 1e3  # convert Myr -> Gyr
        a_new = acceleration_fn(pos_new, t_new)
        vel_new = vel_half + 0.5 * dt * a_new

        particles.positions = pos_new
        particles.velocities = vel_new

        return t_new
