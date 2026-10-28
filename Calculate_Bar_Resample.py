from   Frosst_2026_fourieranalysis  import *
from   periodic_kdtree              import PeriodicCKDTree
from   colibre_utility              import *
import numpy                        as     np
import scipy                        as     scipy
import unyt                         as     unyt
import h5py                         as     h5
import swiftsimio                   as     sw
import scipy.optimize               as     sp_opt
import os
import sys
import time
import itertools
import warnings
import traceback
import matplotlib.pylab             as plt
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# Calculate_Bar_Resample.py
#
# Purpose: quantify how the error on measured bar properties (bar region,
# A2max, Omega, Phi2, ...) varies with the number of star particles used
# in the measurement.
#
# Method: for each galaxy, gather its star particles within the aperture
# ONCE (identical selection/centring/alignment/normalisation to
# Calculate_Bar_Prof.py / Calculate_Bar_Omega.py), then resample WITH
# replacement Ndraws times at the galaxy's own particle count N, and re-run
# the FULL pipeline (profile -> bar region -> Omega) on each resample. The
# scatter across draws, viewed as a function of each galaxy's N, gives the
# error-vs-resolution relation across the whole halo catalogue for free.
#
# This intentionally does NOT re-evolve any simulation and does NOT
# downsample N below the galaxy's true particle count -- it re-analyses the
# same fixed particle set repeatedly under resampling.
# ==========================================================================

# --------------------------------------------------------------------------
# --- Local test path ---
BasePath     = "/Users/23229092/Documents/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

# --- COSMA paths ---
#BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"

# --- Downsampling-resolution study boxes: start with L025_m5, add L050_m5 if affordable ---
#BoxDir       = ["L025_m5/"]                         ; RunDir   = "THERMAL_AGN_m5/" ; snap = 127
#BoxDir       = ["L050_m5/"]                         ; RunDir   = "THERMAL_AGN_m5/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
bname        = "Stars_Mproj_Bar_Resample_"

# ---- analysis Information ----
Nstar_min    = 200   # Minimum number of stellar particles -- deliberately dropped from 5e3 to see
                     # what happens at unreasonably low (numerically unresolved) particle counts.
Nstar_max    = 1e10 # Maximum number of stellar particles

# ---- resampling information ----
Ndraws       = 100    # number of bootstrap (with-replacement) draws per galaxy
SEED         = 42     # master seed, for reproducibility
Rcut         = 50.    # aperture radius [kpc], same as Calculate_Bar_Prof.py / Calculate_Bar_Omega.py

# --- bar-region finding parameters (identical to Calculate_Bar_Region.py) ---
BarRegionKwargs = dict(minA2Bar=0.2, maxDPhi2=15.0, minDexBar=0.15, minNumBar=200)


def bootstrap_one_galaxy(mass_gal, x_gal, y_gal, vx_gal, vy_gal, rhalf_val, xbin_linear, rng, ihalo, TrackId_val):
    """
    Run Ndraws bootstrap (with-replacement) resamples of a single galaxy's
    already-centred/aligned/normalised star-particle arrays through the full
    pipeline: profile binning -> bar-region finding -> bar Omega analysis.

    Returns a dict of length-Ndraws arrays (NaN where a draw failed).
    """
    Ngal = len(mass_gal)

    out = {
        'isbarred'     : np.full(Ndraws, np.nan),
        'nBar_region'  : np.full(Ndraws, np.nan),  # n particles in bar region (region-finding step)
        'R0Bar'        : np.full(Ndraws, np.nan),
        'R1Bar'        : np.full(Ndraws, np.nan),
        'maxA2Bar'     : np.full(Ndraws, np.nan),
        'nBar_omega'   : np.full(Ndraws, np.nan),  # n particles in bar region (Omega step)
        'R0_omega'     : np.full(Ndraws, np.nan),
        'Rm_omega'     : np.full(Ndraws, np.nan),
        'R1_omega'     : np.full(Ndraws, np.nan),
        'Phi2'         : np.full(Ndraws, np.nan),
        'Phi2err'      : np.full(Ndraws, np.nan),
        'Omega'        : np.full(Ndraws, np.nan),
        'Omegaerr'     : np.full(Ndraws, np.nan),
        'Cov'          : np.full(Ndraws, np.nan),
    }

    for idraw in range(Ndraws):
        try:
            # --- resample particle indices WITH replacement, same size as the full set ---
            lsamp = rng.integers(0, Ngal, size=Ngal)

            mass_s = mass_gal[lsamp]
            x_s    = x_gal[lsamp]
            y_s    = y_gal[lsamp]
            vx_s   = vx_gal[lsamp]
            vy_s   = vy_gal[lsamp]

            # -------------------------------------------------
            # --- Step 1: bar profile (as Calculate_Bar_Prof.py) ---
            # -------------------------------------------------
            bar_tool  = FourierMethodFast(mass_s, x_s, y_s, vx_s, vy_s)
            binData   = bar_tool.analyseBins(xbin_linear)

            nB_s       = binData[:, 0]
            R0_prof_s  = binData[:, 1]
            R1_prof_s  = binData[:, 3]
            A2_prof_s  = binData[:, 4]
            Phi2_prof_s= binData[:, 6]

            # -------------------------------------------------
            # --- Step 2: bar region (as Calculate_Bar_Region.py) ---
            # -------------------------------------------------
            nBar, b0, b1, R0Bar, R1Bar, maxA2Bar, isbarred = findBarRegion(
                nB_s, R0_prof_s, R1_prof_s, A2_prof_s, Phi2_prof_s, **BarRegionKwargs
            )

            out['isbarred'][idraw]    = isbarred
            out['nBar_region'][idraw] = nBar
            out['R0Bar'][idraw]       = R0Bar
            out['R1Bar'][idraw]       = R1Bar
            out['maxA2Bar'][idraw]    = maxA2Bar

            # -------------------------------------------------
            # --- Step 3: bar Omega (as Calculate_Bar_Omega.py), only if a bar region was found ---
            # -------------------------------------------------
            if R1Bar > 0.1:
                binOmega = bar_tool.analyseOmega(xbin_linear, R0Bar, R1Bar, tophat=True)

                out['nBar_omega'][idraw] = binOmega[0]
                out['R0_omega'][idraw]   = binOmega[1]
                out['Rm_omega'][idraw]   = binOmega[2]
                out['R1_omega'][idraw]   = binOmega[3]
                out['Phi2'][idraw]       = binOmega[4]
                out['Phi2err'][idraw]    = binOmega[5]
                out['Omega'][idraw]      = binOmega[6] / rhalf_val
                out['Omegaerr'][idraw]   = binOmega[7] / rhalf_val
                out['Cov'][idraw]        = binOmega[8]

        except Exception as e:
            print(f'  [WARN] galaxy ihalo={ihalo} (TrackId={TrackId_val}) draw={idraw} '
                  f'(N={Ngal}) failed: {e}')
            # leave this draw's entries as NaN and continue
            continue

    return out


for idir, Dir in enumerate(BoxDir):

    ext4 = str(snap).zfill(4)
    ext3 = str(snap).zfill(3)

    soap_file = BasePath + Dir + RunDir + "SOAP-HBT/halo_properties_" + ext4 + ".hdf5"

    # --- Read SOAP halo data
    print('SOAP file: ', soap_file)
    soap = sw.load(soap_file)

    # --- Read the selection data (Stellar resolution)
    Nstar_subhalo = soap.bound_subhalo.number_of_star_particles

    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (Using SOAP catalogs)
    lhalo = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]

    print('- snapshot        :', snap)
    print('  Number of galaxies selected: ', len(lhalo))

    # --- Get some properties of the galaxies from SOAP
    rhalf_stars  = soap.bound_subhalo.half_mass_radius_stars[lhalo]
    nstar        = soap.bound_subhalo.number_of_star_particles[lhalo]
    mstar        = soap.exclusive_sphere_50kpc.stellar_mass[lhalo]  # for the mass axis in downstream plots

    velCOM_stars = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[lhalo]
    angJ_stars   = soap.inclusive_sphere_50kpc.angular_momentum_stars[lhalo]

    halo_centre  = soap.input_halos.halo_centre[lhalo]
    TrackId      = soap.input_halos_hbtplus.track_id[lhalo]

    # --- convert Units
    velCOM_stars.convert_to_units('km/s')        ; velCOM_stars.convert_to_physical()
    rhalf_stars.convert_to_units('kpc')          ; rhalf_stars.convert_to_physical()
    halo_centre.convert_to_units('kpc')          ; halo_centre.convert_to_physical()
    mstar.convert_to_units('Msun')               ; mstar.convert_to_physical()
    angJ_stars.convert_to_units('Msun*kpc*km/s') ; angJ_stars.convert_to_physical()

    # --- profile bin definition (identical to Prof/Omega scripts) ---
    Nprof       = 41
    x_range     = [-1, 1]
    xbin        = np.linspace(x_range[0], x_range[1], Nprof)
    xbin_linear = np.append(0, 10 ** xbin[:])

    # ======================================================================
    # PHASE 1: gather each galaxy's star particles ONCE.
    # Mirrors the particle selection in Calculate_Bar_Prof.py / _Omega.py
    # exactly (aperture via KDTree, centred on halo_centre, velocity offset
    # by velCOM_stars, aligned to angJ_stars, normalised by rhalf_stars).
    # NOTE: as in the existing scripts, if a galaxy's particles are found in
    # more than one distributed snapshot file, the later file's selection
    # overwrites the earlier one -- this preserves the exact behaviour of
    # Calculate_Bar_Prof.py / Calculate_Bar_Omega.py. Worth double-checking
    # independently if you suspect apertures straddle file boundaries.
    # ======================================================================
    particle_cache = {}   # ihalo -> dict(mass, x, y, vx, vy, N)

    total_files = 0
    for root, _, filenames in os.walk(BasePath + Dir + RunDir + "snapshots/" + SnapBase + ext4 + "/"):
        total_files += sum(1 for f in filenames if not f.endswith('.old'))
    total_files -= 1  # subtract the master hdf5 file
    print('total files:', total_files)

    for ifile in range(total_files):
        Swiftfile = BasePath + Dir + RunDir + "snapshots/colibre_0127/" + SnapBase + ext4 + '.' + str(ifile) + ".hdf5" #locla
        #Swiftfile = BasePath + Dir + RunDir + "snapshots/" + SnapBase + ext4 + '.' + str(ifile) + ".hdf5"

        print('SWIFT file:', Swiftfile)
        data        = sw.load(Swiftfile)
        meta_data   = data.metadata
        ScaleFactor = meta_data.scale_factor
        boxsize     = meta_data.boxsize * ScaleFactor
        boxsize.convert_to_units('kpc')

        pos_stars  = data.stars.coordinates ; print(' ...read stellar coordinates')
        mass_stars = data.stars.masses      ; print(' ...read stellar masses')
        vel_stars  = data.stars.velocities  ; print(' ...read stellar velocities')

        pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()
        mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()
        vel_stars.convert_to_units('km/s')  ; vel_stars.convert_to_physical()

        print(' \n Building particle tree...')
        star_tree = PeriodicCKDTree(boxsize, pos_stars, leafsize=100) ; print(' Star particle tree done... \n')

        for ihalo, lh in enumerate(lhalo):
            lstar_tree = np.asarray(star_tree.query_ball_point(halo_centre[ihalo, :], Rcut))

            if len(lstar_tree) > 0:
                pos_tree = pos_stars[lstar_tree, :].value - halo_centre[ihalo, :].value
                vel_tree = vel_stars[lstar_tree, :].value - velCOM_stars[ihalo, :].value

                # --- align galaxy with z-component of AM within 50kpc ---
                trans    = rotation_matrix_from_vectors(angJ_stars[ihalo, :], [0, 0, 1])
                pos_tree = (trans @ pos_tree.T).T / rhalf_stars[ihalo].value
                vel_tree = (trans @ vel_tree.T).T

                particle_cache[ihalo] = dict(
                    mass = mass_stars[lstar_tree].value.copy(),
                    x    = pos_tree[:, 0].copy(),
                    y    = pos_tree[:, 1].copy(),
                    vx   = vel_tree[:, 0].copy(),
                    vy   = vel_tree[:, 1].copy(),
                    N    = len(lstar_tree),
                )

        fracs = round(float(ihalo + 1) / len(lhalo), 4)
        print(' file:', ifile, ' | f:', fracs)

    print('\n Particle gathering complete. Galaxies with particles found: ', len(particle_cache), ' / ', len(lhalo))

    # ======================================================================
    # PHASE 2: bootstrap resampling + full pipeline re-run per galaxy
    # ======================================================================
    Ngal = len(lhalo)

    # --- per-galaxy, per-draw outputs (shape: [Ngal, Ndraws]) ---
    field_names = ['isbarred', 'nBar_region', 'R0Bar', 'R1Bar', 'maxA2Bar',
                   'nBar_omega', 'R0_omega', 'Rm_omega', 'R1_omega',
                   'Phi2', 'Phi2err', 'Omega', 'Omegaerr', 'Cov']
    results = {fname: np.full((Ngal, Ndraws), np.nan) for fname in field_names}

    Nstar_used = np.full(Ngal, -1, dtype=int)  # true particle count entering the bootstrap

    rng_master = np.random.default_rng(SEED)

    print('\n Running bootstrap (Ndraws=%d) on %d galaxies...' % (Ndraws, len(particle_cache)))
    for ihalo, lh in enumerate(lhalo):
        if ihalo not in particle_cache:
            continue

        pdata = particle_cache[ihalo]
        Nstar_used[ihalo] = pdata['N']

        # independent, reproducible child RNG per galaxy
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))

        out = bootstrap_one_galaxy(
            pdata['mass'], pdata['x'], pdata['y'], pdata['vx'], pdata['vy'],
            rhalf_stars[ihalo].value, xbin_linear, rng, ihalo, TrackId[ihalo]
        )

        for fname in field_names:
            results[fname][ihalo, :] = out[fname]

        if ihalo % 100 == 0 or ihalo == Ngal - 1:
            print(' Group:', lh, ' | TrackId:', TrackId[ihalo], ' | N:', pdata['N'],
                  ' | f:', round(float(ihalo + 1) / Ngal, 4))

    # ======================================================================
    # Write results to hdf5
    # ======================================================================
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+bname+ext3+".hdf5"                  #Local path
    #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/" + BoxDir[0] + RunDir + bname + ext3 + ".hdf5"
    print('\n Writing to:', fn)

    output = h5.File(fn, "w")
    grp0   = output.create_group("Header")
    grp1   = output.create_group("HaloData")

    dset = grp0.create_dataset('Ndraws',   data=Ndraws,    dtype='int')
    dset = grp0.create_dataset('Seed',     data=SEED,      dtype='int')
    dset = grp0.create_dataset('Nstar_min',data=Nstar_min, dtype='float')

    dset = grp1.create_dataset('TrackId',    data=TrackId,       dtype='int')
    dset = grp1.create_dataset('Nstar_true', data=Nstar_used,    dtype='int')   # N particles entering bootstrap (-1 = none found)
    dset = grp1.create_dataset('rhalf_stars',data=rhalf_stars,   dtype='float')
    dset = grp1.create_dataset('StellarMass',data=mstar,         dtype='float')

    for fname in field_names:
        dset = grp1.create_dataset(fname + '_draws', data=results[fname], dtype='float')

    output.close()

plt.show()
sys.exit() ###################################################################
