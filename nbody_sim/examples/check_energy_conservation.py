"""
Test 1: Energy conservation for an N-body system in an external potential.
"""

import numpy as np
import matplotlib.pyplot as plt

from nbody_sim import (
    SimulationConfig,
    ParticleSet,
    NBodySimulation,
)
from nbody_sim.potentials import ExternalPotential, ProgenitorPotential


# Progenitor orbit (simple circular orbit)
def progenitor_orbit(t):
    R = 15.0  # kpc
    v_circ = 220.0  # km/s
    kpc_per_kms_Gyr = 1.0227121650537077
    omega = (v_circ / R) * kpc_per_kms_Gyr / R
    phi = omega * t
    return np.array([R * np.cos(phi), R * np.sin(phi), 0.0])


# Vectorized total energy with softening
def total_energy_vec(pos, vel, m, G=1.0, eps=1e-3):
    """
    Compute total energy (kinetic + potential) using vectorized operations.
    Softening eps prevents divide-by-zero.
    """
    # Kinetic energy
    v2 = np.sum(vel**2, axis=1)
    T = 0.5 * np.sum(m * v2)

    # Pairwise potential energy
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2) + eps**2)

    iu, ju = np.triu_indices(len(m), k=1)
    U = -G * np.sum((m[iu] * m[ju]) / r[iu, ju])

    return T + U


def main():
    # Optimized test parameters
    N = 200
    dt = 0.01
    stride = 10  # compute energy every 10 snapshots

    # Randomized initial positions (prevents r=0)
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

    # Run simulation
    sim = NBodySimulation(config, particles, external, progenitor)
    result = sim.run()

    times = result.times
    positions = result.positions
    velocities = result.velocities

    # Compute energy on thinned grid
    idx = np.arange(0, len(times), stride)

    energies = np.array([
        total_energy_vec(positions[i], velocities[i], masses)
        for i in idx
    ])

    rel_err = (energies - energies[0]) / energies[0]

    plt.figure()
    plt.plot(times[idx], rel_err, marker="o")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("Relative Energy Error")
    plt.title("Energy Conservation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_conservation.png", dpi=200)

    print(f"Used N={N}, dt={dt}, stride={stride}")
    print("Max |ΔE/E| =", np.max(np.abs(rel_err)))


if __name__ == "__main__":
    main()
    