import numpy as np

def fourier_a_component(mass, pos, m):
    phi = np.arctan2(pos[:,1], pos[:,0])
    am = (mass * np.cos(m * phi)).sum()
    return am

def fourier_b_component(mass, pos, m):
    phi = np.arctan2(pos[:,1], pos[:,0])
    bm = (mass * np.sin(m * phi)).sum()
    return bm

def fourier_AM_component(am, bm, a0):
    return np.sqrt(np.array(am)**2 + np.array(bm)**2) / np.array(a0)

def fourier_phi_component(am, bm):
    return np.arctan(np.array(am) / np.array(bm)) / 2
