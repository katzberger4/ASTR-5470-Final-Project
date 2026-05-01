import numpy as np
import matplotlib.pyplot as plt

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
    omega = (v_circ / R) * kpc_per_kms_Gyr / R
    phi = omega * t
    return np.array([R*np.cos(phi), R*np.sin(phi), 0.0])


def main():
    # Initial particle cloud
    N = 1000

    # Give particles a small spatial extent so the stream forms
    pos0 = np.random.normal(0.0, 0.05, size=(N, 3))   # 50 pc dispersion
    vel0 = np.random.normal(0.0, 7.0, size=(N, 3))    # km/s dispersion

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

    # Stream plot (final snapshot)
    final_pos = result.positions[-1]   # shape (N, 3)

    x = final_pos[:, 0]
    y = final_pos[:, 1]

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=2, alpha=0.6)
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.title("Final Stellar Stream (x–y projection)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("stream.png", dpi=200)

    print("Times:", result.times.shape)
    print("Positions:", result.positions.shape)
    print("Velocities:", result.velocities.shape)

    return result


if __name__ == "__main__":
    main()
