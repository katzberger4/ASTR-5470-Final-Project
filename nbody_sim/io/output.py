import numpy as np

class SimulationResult:
    def __init__(self, times, positions, velocities):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)

    @property
    def n_steps(self):
        return self.times.size

    @property
    def n_particles(self):
        return self.positions.shape[1]
