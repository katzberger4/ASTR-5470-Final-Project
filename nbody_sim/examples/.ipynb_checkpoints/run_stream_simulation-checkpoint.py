import numpy as np

from nbody_sim import (
    SimulationConfig,
    ParticleSet,
    NBodySimulation,
)
from nbody_sim.potentials import ExternalPotential, ProgenitorPotential

def progenitor_orbit(t):
    # Simple circular orbit in x–y plane
    R = 15.0  # kpc
    v_circ = 220.0  # km/s
    kpc_per_kms_Gyr = 1.0227121650537077
    omega = (v_circ / R) * kpc_per_kms_Gyr / R  # crude angular frequency
    phi = omega * t
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = 0.0
    return np.array([x, y, z])

def main():
    # Initial particle cloud around progenitor
    N = 1000
    pos0 = np.zeros((N, 3))
    vel0 = np.zeros((N, 3))
    vel0 += np.random.normal(0.0, 7.0, size=(N, 3))  # km/s dispersion

    particles = ParticleSet(pos0, vel0)

    config = SimulationConfig(
        t_start=-3.0,
        t_end=0.0,
        dt=1.0,       # Myr
        softening=0.01,
    )

    external = ExternalPotential()
    progenitor = ProgenitorPotential(
        M0=6e6,
        r_s=0.1,
        c=20.0,
        t_start=config.t_start,
        t_end=config.t_end,
        orbit_fn=progenitor_orbit,
    )

    sim = NBodySimulation(config, particles, external, progenitor)
    result = sim.run()

    print("Times:", result.times.shape)
    print("Positions:", result.positions.shape)
    print("Velocities:", result.velocities.shape)

    return result

if __name__ == "__main__":
    main()
