from dataclasses import dataclass

@dataclass
class SimulationConfig:
    t_start: float      # Gyr
    t_end: float        # Gyr
    dt: float           # Myr
    softening: float    # kpc
    integrator: str = "leapfrog"
