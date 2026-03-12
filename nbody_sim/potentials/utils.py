import numpy as np

# Gravitational constant in kpc (km/s)^2 / Msun
G = 4.30091e-6

def miyamoto_nagai_acc(positions, M, a, b):
    x, y, z = positions.T
    R = np.sqrt(x**2 + y**2)
    B = a + np.sqrt(z**2 + b**2)
    denom = (R**2 + B**2)**1.5 + 1e-30

    ax = -G * M * x / denom
    ay = -G * M * y / denom
    az = -G * M * B / denom * z / (np.sqrt(z**2 + b**2) + 1e-30)
    return np.vstack((ax, ay, az)).T

def nfw_enclosed_mass(r, M_vir, r_s, c):
    r = np.asarray(r)
    f_c = np.log(1 + c) - c / (1 + c)
    x = r / r_s
    return M_vir * (np.log(1 + x) - x / (1 + x)) / f_c

def nfw_acc(positions, M_vir, r_s, c):
    r_vec = positions
    r = np.linalg.norm(r_vec, axis=1) + 1e-30
    M_enc = nfw_enclosed_mass(r, M_vir, r_s, c)
    a_mag = -G * M_enc / r**3
    return (a_mag[:, None] * r_vec)
