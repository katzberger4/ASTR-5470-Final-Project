"""
Test 3: Orbit visualization + radial profile evolution.
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


def main():
    # Optimized test parameters
    N = 300
    dt = 0.01
    orbit_sample = 80  # number of orbits to plot

    # Random initial positions (avoid r=0)
    pos0 = np.random.normal(0.0, 0.05, size=(N, 3))
    vel0 = np.random.normal(0.0, 7.0, size=(N, 3))

    particles = ParticleSet(pos0, vel0)

    # Ensure masses exist
    if particles.masses is None:
        particles.masses = np.ones(N)

    config = SimulationConfig(
        t_start=-3.0,
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

    positions = result.positions

    # Orbits
    plt.figure(figsize=(6, 6))
    for i in range(min(orbit_sample, N)):
        plt.plot(positions[:, i, 0], positions[:, i, 1], lw=0.5, alpha=0.6)

    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.title("Particle Orbits (x–y projection)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("orbits_xy.png", dpi=200)

    # Radial profiles
    r_initial = np.linalg.norm(positions[0], axis=1)
    r_final = np.linalg.norm(positions[-1], axis=1)

    plt.figure()
    bins = np.linspace(0, max(r_initial.max(), r_final.max()), 40)
    plt.hist(r_initial, bins=bins, alpha=0.5, label="Initial")
    plt.hist(r_final, bins=bins, alpha=0.5, label="Final")
    plt.xlabel("r [kpc]")
    plt.ylabel("N")
    plt.title("Radial Profile Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("radial_profiles.png", dpi=200)

    print(f"Used N={N}, dt={dt}, plotted {orbit_sample} orbits")


if __name__ == "__main__":
    main()
