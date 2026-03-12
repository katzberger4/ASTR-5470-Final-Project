import numpy as np

class ParticleSet:
    def __init__(self, positions, velocities, masses=None):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have same shape")
        self.masses = None if masses is None else np.asarray(masses, dtype=float)

    @property
    def n(self):
        return self.positions.shape[0]
