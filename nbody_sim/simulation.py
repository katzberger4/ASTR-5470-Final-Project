import numpy as np
from .integrators import LeapfrogIntegrator
from .io.output import SimulationResult

class NBodySimulation:
    def __init__(self, config, particles, external_potential, progenitor=None):
        self.config = config
        self.particles = particles
        self.external = external_potential
        self.progenitor = progenitor

        if config.integrator != "leapfrog":
            raise ValueError("Only 'leapfrog' integrator is implemented")
        self.integrator = LeapfrogIntegrator()

    def total_acceleration(self, positions, t):
        a = self.external.acceleration(positions, t)
        if self.progenitor is not None:
            a += self.progenitor.acceleration(positions, t)
        return a

    def run(self, store_every=1):
        cfg = self.config
        n_steps = int(round((cfg.t_end - cfg.t_start) * 1e3 / cfg.dt)) + 1

        times = []
        pos_hist = []
        vel_hist = []

        t = cfg.t_start
        for i in range(n_steps):
            if i % store_every == 0:
                times.append(t)
                pos_hist.append(self.particles.positions.copy())
                vel_hist.append(self.particles.velocities.copy())

            if i == n_steps - 1:
                break

            t = self.integrator.step(
                self.particles,
                t,
                cfg.dt,
                self.total_acceleration,
            )

        times = np.array(times)
        pos_hist = np.stack(pos_hist, axis=0)
        vel_hist = np.stack(vel_hist, axis=0)
        return SimulationResult(times, pos_hist, vel_hist)
