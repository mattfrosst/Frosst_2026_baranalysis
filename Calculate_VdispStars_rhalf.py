import os
import sys
import argparse
import warnings
from   collections                  import defaultdict
import numpy                        as np
import unyt
import h5py                         as h5
import swiftsimio                   as sw
from   concurrent.futures           import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# MASS-WEIGHTED STELLAR VELOCITY DISPERSION, WITHIN r50, IN THE
# ANGULAR-MOMENTUM FRAME
# --------------------------------------------------------------------------
#
# sigma_i^2 = sum(m_j * (v_ij - <v_i>)^2) / sum(m_j),  <v_i> = sum(m_j*v_ij)/sum(m_j)
# sigma_tot^2 = sigma_x^2 + sigma_y^2 + sigma_z^2
#
# This is mass-weighted (not a plain per-particle average), since COLIBRE
# stellar particle masses vary and an unweighted average would let
# however many low-mass particles happen to land in the aperture skew
# the result relative to the galaxy's actual mass distribution.
# ==========================================================================

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

DoBound      = False   # as in the original scripts: currently a no-op placeholder
fname        = "SigmaStars_"

Nstar_min    = 5e3
Nstar_max    = 1e10

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def measure_cell_batch(batch):
    """
    Read one top-level cell's stellar positions/velocities/masses (no
    padding, no DM) and compute mass-weighted velocity dispersions for
    every halo assigned to it. Runs in a worker process.
    """
    (Swiftfile, cell_lo_comov, cell_hi_comov, halo_list) = batch
    # halo_list: list of (local_index, c_phys[3], r50_phys, velCOM[3], angJ[3])

    import numpy               as np
    import unyt
    import swiftsimio           as sw
    from   colibre_utility      import rotation_matrix_from_vectors

    region = [[cell_lo_comov[k], cell_hi_comov[k]] * unyt.kpc for k in range(3)]

    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)
    data = sw.load(Swiftfile, mask=mask)

    pos_stars  = data.stars.coordinates
    vel_stars  = data.stars.velocities
    mass_stars = data.stars.masses

    pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()
    vel_stars.convert_to_units('km/s')  ; vel_stars.convert_to_physical()
    mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()

    pos_stars_val  = pos_stars.value
    vel_stars_val  = vel_stars.value
    mass_stars_val = mass_stars.value

    n_loaded = len(pos_stars_val)

    results = []
    for (local_i, c_phys, r50_phys, velCOM, angJ) in halo_list:
        n_used = 0
        mstar_used = 0.0
        sigma_x2 = sigma_y2 = sigma_z2 = sigma_tot2 = np.nan

        if n_loaded > 0:
            dx  = pos_stars_val - c_phys
            dr2 = np.sum(dx**2, axis=1)
            sel = dr2 <= r50_phys**2
            n_used = int(np.sum(sel))

            if n_used > 0:
                w     = mass_stars_val[sel]                     # per-particle stellar mass, used as weight
                wsum  = np.sum(w)
                mstar_used = float(wsum)

                v_rel = vel_stars_val[sel] - velCOM              # centre on bulk velocity
                R     = rotation_matrix_from_vectors(angJ, [0.0, 0.0, 1.0])
                v_rot = (R @ v_rel.T).T                          # align to angular-momentum z-axis

                vx, vy, vz = v_rot[:, 0], v_rot[:, 1], v_rot[:, 2]

                # Mass-weighted mean and variance in each axis:
                #   <v_i>   = sum(m_j * v_ij) / sum(m_j)
                #   sigma_i^2 = sum(m_j * (v_ij - <v_i>)^2) / sum(m_j)
                vx_mean = np.sum(w * vx) / wsum
                vy_mean = np.sum(w * vy) / wsum
                vz_mean = np.sum(w * vz) / wsum

                sigma_x2 = np.sum(w * (vx - vx_mean)**2) / wsum
                sigma_y2 = np.sum(w * (vy - vy_mean)**2) / wsum
                sigma_z2 = np.sum(w * (vz - vz_mean)**2) / wsum
                sigma_tot2 = sigma_x2 + sigma_y2 + sigma_z2

        results.append((local_i, sigma_tot2, sigma_x2, sigma_y2, sigma_z2, n_used, mstar_used))

    return results, n_loaded


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)

        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        print('SOAP file: ', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo_all     = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]

        chunk_bounds = np.linspace(0, len(lhalo_all), args.n_chunks + 1, dtype=int)
        c0, c1 = chunk_bounds[args.chunk_id], chunk_bounds[args.chunk_id + 1]
        lhalo  = lhalo_all[c0:c1]

        print(f'- snapshot: {snap}   chunk {args.chunk_id}/{args.n_chunks}   '
              f'halos {c0}:{c1} of {len(lhalo_all)} total ({len(lhalo)} in this chunk)')

        rhalf_stars  = soap.bound_subhalo.half_mass_radius_stars[lhalo]
        halo_centre  = soap.input_halos.halo_centre[lhalo]
        TrackId      = soap.input_halos_hbtplus.track_id[lhalo]
        velCOM_stars = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[lhalo]
        angJ_stars   = soap.inclusive_sphere_50kpc.angular_momentum_stars[lhalo]

        rhalf_stars.convert_to_units('kpc')             ; rhalf_stars.convert_to_physical()
        halo_centre.convert_to_units('kpc')              ; halo_centre.convert_to_physical()
        velCOM_stars.convert_to_units('km/s')            ; velCOM_stars.convert_to_physical()
        angJ_stars.convert_to_units('Msun*kpc*km/s')     ; angJ_stars.convert_to_physical()

        r50_vals    = rhalf_stars.value
        centre_vals = halo_centre.value
        velCOM_vals = velCOM_stars.value
        angJ_vals   = angJ_stars.value

        # --- Metadata + cell grid (cheap, no particle data touched) ---
        mask0       = sw.mask(Swiftfile, spatial_only=True)
        meta        = mask0.metadata
        ScaleFactor = float(meta.scale_factor)

        cell_size_val = float(mask0.cell_size.to('kpc').value[0])

        boxsize_comoving = meta.boxsize.copy()
        boxsize_comoving.convert_to_units('kpc')
        boxsize_comoving_val = float(boxsize_comoving.value[0])

        n_cells_per_dim = int(round(boxsize_comoving_val / cell_size_val))

        print(f'  Cell size: {cell_size_val:.2f} kpc  '
              f'({n_cells_per_dim}^3 = {n_cells_per_dim**3} cells)')
        print('  Using', N_WORKERS, 'worker processes')

        # --- Assign each halo to a home cell; skip if its r50 aperture
        #     pokes outside that cell's bounds on any axis ---
        cell_groups = defaultdict(list)
        n_skipped   = 0

        for i in range(len(lhalo)):
            c_phys   = centre_vals[i]
            r50_phys = r50_vals[i]
            r_comov  = r50_phys / ScaleFactor           # aperture is 1*r50 here
            c_comov  = c_phys / ScaleFactor

            idx = np.floor(c_comov / cell_size_val).astype(int) % n_cells_per_dim
            local = c_comov - idx * cell_size_val

            safe = np.all((local - r_comov >= 0.0) & (local + r_comov <= cell_size_val))
            if safe:
                cell_groups[tuple(idx)].append(
                    (i, c_phys, r50_phys, velCOM_vals[i], angJ_vals[i])
                )
            else:
                n_skipped += 1

        n_included = len(lhalo) - n_skipped
        print(f'  {n_included}/{len(lhalo)} halos fully contained in their home cell '
              f'({n_skipped} skipped -- aperture crosses a cell boundary), '
              f'spread across {len(cell_groups)} occupied cells')

        tasks = []
        for idx, halo_list in cell_groups.items():
            cell_lo = np.array(idx) * cell_size_val
            cell_hi = cell_lo + cell_size_val
            tasks.append((Swiftfile, cell_lo, cell_hi, halo_list))

        sigma_tot2 = np.full(len(lhalo), np.nan)
        sigma_x2   = np.full(len(lhalo), np.nan)
        sigma_y2   = np.full(len(lhalo), np.nan)
        sigma_z2   = np.full(len(lhalo), np.nan)
        n_used_arr = np.zeros(len(lhalo), dtype=int)
        mstar_used_arr = np.zeros(len(lhalo), dtype=float)

        max_n_loaded = 0

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(measure_cell_batch, t): t for t in tasks}
            ndone_cells = 0
            ndone_halos = 0
            for fut in as_completed(futures):
                results, n_loaded = fut.result()
                for (local_i, st2, sx2, sy2, sz2, n_used, mstar_used) in results:
                    sigma_tot2[local_i]     = st2
                    sigma_x2[local_i]       = sx2
                    sigma_y2[local_i]       = sy2
                    sigma_z2[local_i]       = sz2
                    n_used_arr[local_i]     = n_used
                    mstar_used_arr[local_i] = mstar_used
                ndone_halos += len(results)
                max_n_loaded = max(max_n_loaded, n_loaded)

                ndone_cells += 1
                if ndone_cells % 20 == 0 or ndone_cells == len(tasks):
                    print(f'  ... {ndone_cells}/{len(tasks)} cells done, '
                          f'{ndone_halos}/{n_included} halos done  '
                          f'(max particles in a single cell read so far: {max_n_loaded})',
                          flush=True)
              
        fn = (BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+f"_chunk{args.chunk_id:04d}.hdf5") #Local path
        #fn = ("/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+fname+ext3+f"_chunk{args.chunk_id:04d}.hdf5") #COSMA path
        print('\n Writing to:', fn)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        output = h5.File(fn, "w")
        grp0   = output.create_group("Header")
        grp1   = output.create_group("HaloData")

        grp0.create_dataset('Redshift',       data=1./ScaleFactor - 1, dtype='float')
        grp0.create_dataset('ChunkId',        data=args.chunk_id,      dtype='int')
        grp0.create_dataset('NChunks',        data=args.n_chunks,      dtype='int')
        grp0.create_dataset('CellSizeKpc',    data=cell_size_val,      dtype='float')
        grp0.create_dataset('NHalosSkipped',  data=n_skipped,          dtype='int')
        grp0.create_dataset('NHalosIncluded', data=n_included,         dtype='int')

        grp1.create_dataset('TrackId',      data=TrackId,     dtype='int')
        grp1.create_dataset('r50_stars',    data=rhalf_stars, dtype='float')
        grp1.create_dataset('NstarUsed',    data=n_used_arr,  dtype='int')
        grp1.create_dataset('MstarUsed',    data=mstar_used_arr, dtype='float')

        grp1.create_dataset('sigma_tot2',   data=sigma_tot2,  dtype='float')
        grp1.create_dataset('sigma_x2',     data=sigma_x2,    dtype='float')
        grp1.create_dataset('sigma_y2',     data=sigma_y2,    dtype='float')
        grp1.create_dataset('sigma_z2',     data=sigma_z2,    dtype='float')

        output.close()

    sys.exit()
