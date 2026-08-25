import os
import sys
import warnings
import numpy                        as np
import unyt
import h5py                         as h5
import swiftsimio                   as sw
from   concurrent.futures           import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# WHY THIS VERSION IS FASTER
# --------------------------------------------------------------------------
# The original script looped over every sub-file of the snapshot, loaded
# ALL star + DM particles in that sub-file, built a full KD-tree from them,
# and then queried that tree once per halo -- for EVERY sub-file, for
# EVERY halo. Total work therefore scaled as
# (number_of_files x number_of_halos), and every byte of the (many-billion
# -particle) simulation got read into memory repeatedly per snapshot,
# regardless of how many galaxies you actually cared about.
#
# This version instead uses swiftsimio's built-in *spatial masking*, which
# reads the cell metadata already stored in the snapshot to work out
# exactly which HDF5 chunks overlap a given region, and only reads those.
# For each halo we request just the small cube (3*r50 aperture) around its
# centre, directly from the (virtual) master snapshot file -- so we read
# O(particles in the aperture) instead of O(particles in the whole box),
# and sub-files that don't overlap a halo are never touched at all. The
# per-halo loop is also embarrassingly parallel, so it's run across
# multiple worker processes.
#
# No KD-tree is needed any more (the masked read already returns a small,
# local set of particles, so a brute-force periodic distance calc on that
# subset is plenty fast) -- so the periodic_kdtree dependency is dropped.
#
# The physics/maths (aperture masses, fDM ratios, outputs) is unchanged.
# Recommend re-running this on your small local test box (L012_m6) first
# and diffing the output HDF5 against a run of your original script, to
# confirm bit-for-bit (or close to it) agreement before trusting it on
# the big production boxes.
# ==========================================================================

# --------------------------------------------------------------------------
# ---- Simulation information ----
#Fiducial_test

# --- Local test path ---
BasePath     = "/Users/23229092/Documents/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

# --- COSMA paths ---
#BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
#BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L050_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 123
#BoxDir       = ["L100_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L200_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127

#BoxDir       = ["L025_m5/"]                         ; RunDir   = "THERMAL_AGN_m5/" ; snap = 127
#BoxDir       = ["L025_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L025_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "fDM_"

# ---- analysis Information ----
Nstar_min    = 5e3  # Minimum number of stellar particles
Nstar_max    = 1e10 # Maximum number of stellar particles

# ---- parallelism ----
# Number of worker processes for the per-halo loop. Defaults to all cores
# visible to the process, or to $SLURM_CPUS_PER_TASK if set (typical on a
# batch queue -- set this to match the CPUs you request in your job script).
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def measure_halo(task):
    """
    Read only the particles within 3*r50 of a single halo (via swiftsimio
    spatial masking on the master snapshot file) and accumulate the
    stellar/DM masses within 1, 2 and 3 times r50.

    Runs in a worker process, so it re-imports what it needs and takes a
    plain tuple in / tuple out -- keeps it cheaply picklable for
    ProcessPoolExecutor.
    """
    (Swiftfile, c_phys, r50_phys, ScaleFactor,
     boxsize_comoving_val, boxsize_phys_val) = task

    import numpy      as np
    import unyt
    import swiftsimio as sw

    r1_sq = r50_phys**2
    r2_sq = (2.0 * r50_phys)**2
    r3    = 3.0 * r50_phys

    # --- Cell metadata on disk lives in the comoving frame, so the mask
    #     region has to be specified in comoving coordinates ---
    c_comov  = c_phys / ScaleFactor
    r3_comov = r3      / ScaleFactor

    region = []
    for k in range(3):
        lo = c_comov[k] - r3_comov
        hi = c_comov[k] + r3_comov
        # Handle a halo sitting near a periodic boundary: swiftsimio
        # interprets lo > hi as "wrap through the box edge".
        if lo < 0.0:
            lo = lo % boxsize_comoving_val
        if hi > boxsize_comoving_val:
            hi = hi % boxsize_comoving_val
        region.append([lo, hi] * unyt.kpc)

    mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)
    data = sw.load(Swiftfile, mask=mask)

    pos_stars  = data.stars.coordinates
    mass_stars = data.stars.masses
    pos_dm     = data.dark_matter.coordinates
    mass_dm    = data.dark_matter.masses

    pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()
    mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()
    pos_dm.convert_to_units('kpc')      ; pos_dm.convert_to_physical()
    mass_dm.convert_to_units('Msun')    ; mass_dm.convert_to_physical()

    pos_stars_val  = pos_stars.value
    mass_stars_val = mass_stars.value
    pos_dm_val     = pos_dm.value
    mass_dm_val    = mass_dm.value

    mstar_1 = mstar_2 = mstar_3 = 0.0
    mdm_1   = mdm_2   = mdm_3   = 0.0

    if len(pos_stars_val) > 0:
        dx   = pos_stars_val - c_phys
        dx  -= boxsize_phys_val * np.round(dx / boxsize_phys_val)  # periodic BC
        dr2  = np.sum(dx**2, axis=1)
        in3  = dr2 <= r3**2
        m    = mass_stars_val[in3]
        d2   = dr2[in3]
        mstar_1 = np.sum(m[d2 <= r1_sq])
        mstar_2 = np.sum(m[d2 <= r2_sq])
        mstar_3 = np.sum(m)

    if len(pos_dm_val) > 0:
        dx   = pos_dm_val - c_phys
        dx  -= boxsize_phys_val * np.round(dx / boxsize_phys_val)
        dr2  = np.sum(dx**2, axis=1)
        in3  = dr2 <= r3**2
        m    = mass_dm_val[in3]
        d2   = dr2[in3]
        mdm_1 = np.sum(m[d2 <= r1_sq])
        mdm_2 = np.sum(m[d2 <= r2_sq])
        mdm_3 = np.sum(m)

    return mstar_1, mdm_1, mstar_2, mdm_2, mstar_3, mdm_3


if __name__ == "__main__":

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)

        # Master (virtual) snapshot file. swiftsimio's spatial masking reads
        # the cell metadata from this file and pulls only the required
        # chunks, even though the underlying data physically lives across
        # many colibre_XXXX.N.hdf5 sub-files -- so we no longer need to
        # loop over sub-files by hand at all.
        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # --- Read SOAP halo data
        print('SOAP file: ', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]

        print('- snapshot        :', snap)
        print('  Number of galaxies selected: ', len(lhalo))

        rhalf_stars = soap.bound_subhalo.half_mass_radius_stars[lhalo]
        halo_centre = soap.input_halos.halo_centre[lhalo]
        TrackId     = soap.input_halos_hbtplus.track_id[lhalo]

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()

        r50_vals    = rhalf_stars.value
        centre_vals = halo_centre.value

        # --- Cheap metadata-only read (lazy loading means no particle
        #     arrays are actually touched here) ---
        meta        = sw.load(Swiftfile).metadata
        ScaleFactor = float(meta.scale_factor)

        boxsize_comoving = meta.boxsize.copy()
        boxsize_comoving.convert_to_units('kpc')
        boxsize_comoving_val = float(boxsize_comoving.value[0])   # assumes cubic box

        boxsize_phys = meta.boxsize * ScaleFactor
        boxsize_phys.convert_to_units('kpc')
        boxsize_phys_val = float(boxsize_phys.value[0])

        print('  Scale factor        :', ScaleFactor)
        print('  Box size (phys kpc) :', boxsize_phys_val)
        print('  Using', N_WORKERS, 'worker processes')

        # --- One task per halo ---
        tasks = [
            (Swiftfile, centre_vals[i], r50_vals[i], ScaleFactor,
             boxsize_comoving_val, boxsize_phys_val)
            for i in range(len(lhalo))
        ]

        mstar_1r50 = np.zeros(len(lhalo))
        mdm_1r50   = np.zeros(len(lhalo))
        mstar_2r50 = np.zeros(len(lhalo))
        mdm_2r50   = np.zeros(len(lhalo))
        mstar_3r50 = np.zeros(len(lhalo))
        mdm_3r50   = np.zeros(len(lhalo))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(measure_halo, t): i for i, t in enumerate(tasks)}
            ndone = 0
            for fut in as_completed(futures):
                i = futures[fut]
                (mstar_1r50[i], mdm_1r50[i],
                 mstar_2r50[i], mdm_2r50[i],
                 mstar_3r50[i], mdm_3r50[i]) = fut.result()
                ndone += 1
                if ndone % 100 == 0 or ndone == len(tasks):
                    print(f'  ... {ndone}/{len(tasks)} halos done')

        # --- Compute stellar-to-dark matter mass ratios (M_* / M_DM) ---
        fDM_1r50 = np.where(mdm_1r50 > 0, mstar_1r50 / mdm_1r50, np.nan)
        fDM_2r50 = np.where(mdm_2r50 > 0, mstar_2r50 / mdm_2r50, np.nan)
        fDM_3r50 = np.where(mdm_3r50 > 0, mstar_3r50 / mdm_3r50, np.nan)
        print(fDM_1r50)

        # --- Write to hdf5
        fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"                  #Local path
        #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+fname+ext3+".hdf5" #COSMA path
        print('\n Writing to:', fn)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        output = h5.File(fn, "w")
        grp0   = output.create_group("Header")
        grp1   = output.create_group("HaloData")

        grp0.create_dataset('Redshift', data=1./ScaleFactor - 1, dtype='float')

        grp1.create_dataset('TrackId',   data=TrackId,     dtype='int')
        grp1.create_dataset('r50_stars', data=rhalf_stars, dtype='float')

        grp1.create_dataset('fDM_1r50',   data=fDM_1r50,   dtype='float')
        grp1.create_dataset('mstar_1r50', data=mstar_1r50, dtype='float')
        grp1.create_dataset('mdm_1r50',   data=mdm_1r50,   dtype='float')

        grp1.create_dataset('fDM_2r50',   data=fDM_2r50,   dtype='float')
        grp1.create_dataset('mstar_2r50', data=mstar_2r50, dtype='float')
        grp1.create_dataset('mdm_2r50',   data=mdm_2r50,   dtype='float')

        grp1.create_dataset('fDM_3r50',   data=fDM_3r50,   dtype='float')
        grp1.create_dataset('mstar_3r50', data=mstar_3r50, dtype='float')
        grp1.create_dataset('mdm_3r50',   data=mdm_3r50,   dtype='float')

        output.close()

    sys.exit()
