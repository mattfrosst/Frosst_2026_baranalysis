from   periodic_kdtree              import PeriodicCKDTree
from   colibre_utility              import rotation_matrix_from_vectors
import numpy                        as     np
import h5py                         as     h5
import swiftsimio                   as     sw
import unyt
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# --------------------------------------------------------------------------
# Face-on radial profiles of gas, HI, H2 and star formation for every galaxy, in bins of
# R / r50_stars (the same normalisation and face-on frame as Calculate_Bar_Prof.py).
#
# For each galaxy, gas particles are rotated face-on using the stellar angular momentum within
# 50 kpc, kept within a slab |z| < ZCUT_RHALF * r50, and binned in cylindrical radius R.
# Per bin we store the summed gas, HI and H2 masses, SFR and particle number.
#   HI  mass = m * X_H * f_HI        (SpeciesFractions are n_i / n_H)
#   H2  mass = m * X_H * 2 * f_H2
#   SFR      = particle SFR, negative values clipped to 0
#
# Differences from Calculate_Bar_Prof.py:
#   * bin sums are ADDED across snapshot sub-files, so galaxies whose particles are split
#     between files get complete profiles;
#   * particle offsets from the halo centre are wrapped periodically;
#   * SOAP columns are freed after use (see Calculate_Gal_Table.py) to keep memory low.
# --------------------------------------------------------------------------

# ---- Simulation information ----
# --- Local test path ---
#BasePath     = "/Users/matthewfrosst/Documents/COLIBRE/" ; SnapBase = "colibre_"
#BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#OutBase      = "/Users/matthewfrosst/Documents/COLIBRE/Frosst_2026_Outputs/"

# --- COSMA paths ---
BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
OutBase      = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"
BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

fname        = "Gas_Prof_"

# ---- analysis Information ----
Nstar_min    = 5e3  # Minimum number of stellar particles (same selection as the bar catalogues)
Nstar_max    = 1e10 # Maximum number of stellar particles

# Radial bins in units of r50_stars: a central bin 0-0.1 r50, then log-spaced to R_MAX_RHALF
R_MAX_RHALF  = 6.0
N_LOG_BINS   = 14
R_EDGES      = np.concatenate([[0.0], np.logspace(-1, np.log10(R_MAX_RHALF), N_LOG_BINS + 1)])  # 15 bins
ZCUT_RHALF   = 1.0  # keep gas with |z| < ZCUT_RHALF * r50 (removes gas well above / below the disc)

# --------------------------------------------------------------------------
# ---- Memory-friendly SOAP reads (as in Calculate_Gal_Table.py) ----
def free_column(group, name):
    try:
        delattr(group, name)               # swiftsimio's deleter resets its cached copy
    except AttributeError:
        setattr(group, '_' + name, None)   # fallback: clear the cache attribute directly

def read_selected(group, name, idx):
    full = getattr(group, name)
    sel  = full[idx]
    del full
    free_column(group, name)
    return sel

def peak_mem_gb():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == 'darwin' else peak / 1e6   # bytes on macOS, kB on Linux

# --------------------------------------------------------------------------
# ---- Profile helpers ----
def periodic_offset(pos, centre, boxsize):
    # particle offsets from the centre, wrapped into [-L/2, L/2)
    d = pos - centre
    return (d + 0.5*boxsize) % boxsize - 0.5*boxsize

def bin_profile(offsets, weights, trans, r50, edges=R_EDGES, zcut=ZCUT_RHALF):
    # Rotate offsets [kpc] face-on, normalise by r50 and bin in cylindrical R within |z| < zcut.
    # weights: (N, n_quantities). Returns (n_bins, n_quantities) summed weights and (n_bins,) counts.
    x = (trans @ offsets.T).T / r50
    R = np.hypot(x[:, 0], x[:, 1])
    keep = (np.abs(x[:, 2]) < zcut) & (R < edges[-1])
    ib = np.digitize(R[keep], edges) - 1
    sums = np.zeros((len(edges) - 1, weights.shape[1]))
    for q in range(weights.shape[1]):
        sums[:, q] = np.bincount(ib, weights=weights[keep, q], minlength=len(edges) - 1)
    counts = np.bincount(ib, minlength=len(edges) - 1)
    return sums, counts

# --------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, default=snap,
                        help=f"snapshot number to analyse (default: {snap}, set at the top of this script)")
    args = parser.parse_args()
    snap = args.snap

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # --- Read SOAP halo data
        print('SOAP file: ', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
        del Nstar_subhalo ; free_column(soap.bound_subhalo, 'number_of_star_particles')
        print('- snapshot        :', snap)
        print('  Number of galaxies selected: ', len(lhalo))

        rhalf_stars = read_selected(soap.bound_subhalo, 'half_mass_radius_stars', lhalo)
        mstar       = read_selected(soap.exclusive_sphere_50kpc, 'stellar_mass', lhalo)
        angJ_stars  = read_selected(soap.inclusive_sphere_50kpc, 'angular_momentum_stars', lhalo)
        halo_centre = read_selected(soap.input_halos, 'halo_centre', lhalo)
        is_central  = read_selected(soap.input_halos, 'is_central', lhalo)
        TrackId     = read_selected(soap.input_halos_hbtplus, 'track_id', lhalo)

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        mstar.convert_to_units('Msun')      ; mstar.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()
        angJ_stars.convert_to_units('Msun*kpc*km/s') ; angJ_stars.convert_to_physical()
        print(f'  Peak memory after SOAP reads: {peak_mem_gb():.1f} GB')

        r50     = rhalf_stars.value
        centres = halo_centre.value
        # face-on rotation for each galaxy (stellar angular momentum within 50 kpc, as Calculate_Bar_Prof.py)
        trans_all = np.array([rotation_matrix_from_vectors(angJ_stars[i, :].value, [0, 0, 1]) for i in range(len(lhalo))])
        # 3D search radius that encloses the cylinder R < R_MAX, |z| < ZCUT
        r_search = r50 * np.hypot(R_EDGES[-1], ZCUT_RHALF)

        # --- output arrays: summed per bin, accumulated over all snapshot sub-files
        quantities = ['GasMass', 'HIMass', 'H2Mass', 'SFR']
        n_bins  = len(R_EDGES) - 1
        prof    = np.zeros((len(lhalo), n_bins, len(quantities)))
        npart   = np.zeros((len(lhalo), n_bins), dtype=np.int64)

        # --- How many files do we need to look at? (same counting as Calculate_Bar_Prof.py)
        total_files = 0
        for root, _, filenames in os.walk(BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/"):
            total_files += sum(1 for f in filenames if not f.endswith('.old'))
        total_files -= 1  # subtract the master hdf5 file
        print('total files:', total_files)

        for ifile in range(total_files):
            Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+'.'+str(ifile)+".hdf5"
            print('SWIFT file:', Swiftfile)
            data        = sw.load(Swiftfile)
            ScaleFactor = data.metadata.scale_factor
            boxsize     = data.metadata.boxsize * ScaleFactor
            boxsize.convert_to_units('kpc')
            redshift    = 1.0/ScaleFactor - 1.0

            # --- read the gas particle data
            pos_gas = data.gas.coordinates ; pos_gas.convert_to_units('kpc') ; pos_gas.convert_to_physical()
            m_gas   = data.gas.masses      ; m_gas.convert_to_units('Msun')  ; m_gas.convert_to_physical()
            X_H     = np.asarray(data.gas.element_mass_fractions.hydrogen)
            f_HI    = np.asarray(data.gas.species_fractions.HI)
            f_H2    = np.asarray(data.gas.species_fractions.H2)
            sfr     = np.clip(data.gas.star_formation_rates.to_physical().to_value(unyt.Msun/unyt.yr), 0, None)
            print(' ...read gas particle data')

            m_gas   = m_gas.value
            weights = np.column_stack([m_gas, m_gas*X_H*f_HI, 2.0*m_gas*X_H*f_H2, sfr])
            del X_H, f_HI, f_H2, sfr

            gas_tree = PeriodicCKDTree(boxsize.value, pos_gas.value, leafsize=100) ; print(' Gas particle tree done...')
            pos_val  = pos_gas.value

            for ihalo in range(len(lhalo)):
                lgas = np.asarray(gas_tree.query_ball_point(centres[ihalo, :], r_search[ihalo]))
                if len(lgas) == 0:
                    continue
                offsets = periodic_offset(pos_val[lgas, :], centres[ihalo, :], boxsize.value)
                sums, counts = bin_profile(offsets, weights[lgas, :], trans_all[ihalo], r50[ihalo])
                prof[ihalo]  += sums      # ADD: a galaxy may be split across sub-files
                npart[ihalo] += counts

                if ihalo % 5000 == 0:
                    print(f'  galaxy {ihalo}/{len(lhalo)}')

            del data, pos_gas, pos_val, m_gas, weights, gas_tree
            print(f'  Peak memory so far: {peak_mem_gb():.1f} GB')

        # --- Write to hdf5
        fn = OutBase+Dir+RunDir+fname+ext3+".hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with h5.File(fn, "w") as output:
            grp0 = output.create_group("Header")
            grp0.create_dataset('Redshift',    data=redshift)
            grp0.create_dataset('ZCUT_RHALF',  data=ZCUT_RHALF)
            grp0.attrs['description'] = ('Face-on gas profiles in bins of R/r50_stars within |z| < ZCUT_RHALF*r50; '
                                         'bin sums of gas, HI, H2 mass [Msun] and SFR [Msun/yr]')

            grp1 = output.create_group("HaloData")
            grp1.create_dataset('TrackId',     data=np.asarray(TrackId),    dtype='int')
            grp1.create_dataset('is_central',  data=np.asarray(is_central), dtype='int')
            grp1.create_dataset('StellarMass', data=mstar.value,            dtype='float')
            grp1.create_dataset('r50_stars',   data=r50,                    dtype='float')

            grp2 = output.create_group("Profiles")
            grp2.create_dataset('R_edges_rhalf', data=R_EDGES, dtype='float')
            for q, name in enumerate(quantities):
                grp2.create_dataset(name, data=prof[:, :, q], dtype='float')
            grp2.create_dataset('NumGasPart', data=npart, dtype='int')
        print(' Done.')
