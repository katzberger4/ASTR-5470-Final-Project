"""
Test 2: Angular momentum evolution for an N-body system.
"""

import numpy as np
import matplotlib.pyplot as plt

from nbody_sim import (
    SimulationConfig,
    ParticleSet,
    NBodySimulation,
)
from nbody_sim.potentials import ExternalPotential, ProgenitorPotential


def progenitor_orbit(t):
    R = 15.0
    v_circ = 220.0
    kpc_per_kms_Gyr = 1.0227121650537077
    omega = (v_circ / R) * kpc_per_kms_Gyr / R
    phi = omega * t
    return np.array([R*np.cos(phi), R*np.sin(phi), 0.0])


def angular_momentum(pos, vel, m):
    """Vectorized total angular momentum."""
    return np.sum(m[:, None] * np.cross(pos, vel), axis=0)


def main():
    # Optimized test parameters
    N = 200
    dt = 0.01
    stride = 10

    # Random initial positions (avoid r=0)
    pos0 = np.random.normal(0.0, 0.05, size=(N, 3))
    vel0 = np.random.normal(0.0, 7.0, size=(N, 3))

    particles = ParticleSet(pos0, vel0)

    # Ensure masses exist
    if particles.masses is None:
        particles.masses = np.ones(N)

    masses = particles.masses

    config = SimulationConfig(
        t_start=-2.0,
        t_end=0.0,
        dt=dt,
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

    times = result.times
    positions = result.positions
    velocities = result.velocities

    # Compute angular momentum on thinned grid
    idx = np.arange(0, len(times), stride)

    L = np.array([
        angular_momentum(positions[i], velocities[i], masses)
        for i in idx
    ])

    Lnorm = np.linalg.norm(L, axis=1)


    plt.figure()
    plt.plot(times[idx], Lnorm, marker="o")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("|L|")
    plt.title("Angular Momentum Evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("angular_momentum.png", dpi=200)

    print(f"Used N={N}, dt={dt}, stride={stride}")
    print("Δ|L|/|L₀| =", (Lnorm[-1] - Lnorm[0]) / Lnorm[0])


if __name__ == "__main__":
    main()
