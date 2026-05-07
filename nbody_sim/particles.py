import numpy as np

class ParticleSet:
    def __init__(self, positions, velocities, masses=None):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have same shape")

        # Assign equal masses if none provided
        if masses is None:
            self.masses = np.ones(self.positions.shape[0], dtype=float)
        else:
            self.masses = np.asarray(masses, dtype=float)

    @property
    def n(self):
        return self.positions.shape[0]
