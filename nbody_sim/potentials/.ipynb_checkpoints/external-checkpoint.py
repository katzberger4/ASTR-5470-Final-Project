import numpy as np
from .utils import miyamoto_nagai_acc, nfw_acc, plummer_acc

class ExternalPotential:
    """
    Milky Way (disk + halo) + simple moving LMC halo.
    Units: positions in kpc, velocities in km/s, time in Gyr.
    """

    def __init__(
        self,
        mw_disk_mass=6e10,
        mw_disk_a=6.5,
        mw_disk_b=0.26,
        mw_halo_mass=1e12,
        mw_halo_rs=20.0,
        mw_halo_c=10.0,
        lmc_mass=1.5e11,
        lmc_rs=10.0,
        lmc_c=12.0,
        lmc_r0=np.array([0.0, -50.0, 0.0]),
        lmc_v=np.array([-50.0, 0.0, 0.0]),  # km/s
        halo_profile="nfw",   # "nfw" or "plummer"
    ):
        self.mw_disk_mass = mw_disk_mass
        self.mw_disk_a = mw_disk_a
        self.mw_disk_b = mw_disk_b
        self.mw_halo_mass = mw_halo_mass
        self.mw_halo_rs = mw_halo_rs
        self.mw_halo_c = mw_halo_c

        self.lmc_mass = lmc_mass
        self.lmc_rs = lmc_rs
        self.lmc_c = lmc_c
        self.lmc_r0 = np.asarray(lmc_r0, dtype=float)
        self.lmc_v = np.asarray(lmc_v, dtype=float)

        self.halo_profile = halo_profile

        # 1 km/s * 1 Gyr ≈ 1.0227 kpc
        self.kpc_per_kms_Gyr = 1.0227121650537077

    def lmc_center(self, t):
        return self.lmc_r0 + self.lmc_v * self.kpc_per_kms_Gyr * t

    def acceleration(self, positions, t):
        positions = np.asarray(positions, dtype=float)

        # Milky Way disk
        a_disk = miyamoto_nagai_acc(
            positions,
            M=self.mw_disk_mass,
            a=self.mw_disk_a,
            b=self.mw_disk_b,
        )

        # Choose MW halo profile
        if self.halo_profile == "nfw":
            a_halo = nfw_acc(
                positions,
                M_vir=self.mw_halo_mass,
                r_s=self.mw_halo_rs,
                c=self.mw_halo_c,
            )
        elif self.halo_profile == "plummer":
            a_halo = plummer_acc(
                positions,
                M=self.mw_halo_mass,
                a=self.mw_halo_rs,
            )
        else:
            raise ValueError("halo_profile must be 'nfw' or 'plummer'")

        # LMC halo
        r_lmc = self.lmc_center(t)
        rel = positions - r_lmc[None, :]

        if self.halo_profile == "nfw":
            a_lmc = nfw_acc(
                rel,
                M_vir=self.lmc_mass,
                r_s=self.lmc_rs,
                c=self.lmc_c,
            )
        else:
            a_lmc = plummer_acc(
                rel,
                M=self.lmc_mass,
                a=self.lmc_rs,
            )

        return a_disk + a_halo + a_lmc
