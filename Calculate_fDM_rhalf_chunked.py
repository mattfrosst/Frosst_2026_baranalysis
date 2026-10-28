import os
import sys
import argparse
import resource
import warnings
from   collections                  import defaultdict
import numpy                        as np
import unyt
import h5py                         as h5
import swiftsimio                   as sw
from   concurrent.futures           import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# CELL-BATCHED VERSION
# --------------------------------------------------------------------------
# Diagnostics on L200_m6 showed the top-level SWIFT cell size is 3.125 Mpc
# (a 64^3 grid), while typical galaxy apertures (3*r50) are only tens of
# kpc. Two consequences:
#
#   1. A per-halo masked read was pulling in an entire ~3 Mpc cell (plus
#      padded neighbours) every time, regardless of how small the halo's
#      real aperture was -- a few hundred kpc's worth of "useful" data
#      buried inside a many-million-particle read.
#   2. Nearby halos (e.g. group/cluster members) can share the same
#      top-level cell, meaning we were often reading the *same* cell
#      from disk independently once per halo.
#
# This version:
#   - groups halos by which top-level cell they live in
#   - reads each OCCUPIED cell exactly once, with no padding (since the
#     cell's own bounds are already known exactly)
#   - computes the aperture masses for every halo assigned to that cell
#     from that one shared read
#   - SKIPS halos whose 3*r50 aperture pokes outside their home cell's
#     bounds on any axis (per Matt's instruction: ignore cross-cell
#     galaxies for now). These get NaN in the output and are counted, so
#     you can see how many were skipped and decide whether to add a
#     second pass for them later.
#
# This also naturally removes the periodic-wrap logic from the previous
# versions: since every read is strictly bounded by one cell's exact
# edges, and every included halo's aperture is guaranteed to fit inside
# those edges, there's nothing that can wrap.
# ==========================================================================

BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

DoBound      = False
fname        = "fDM_"

Nstar_min    = 5e3
Nstar_max    = 1e10

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def measure_cell_batch(batch):
    """
    Read exactly one top-level cell (no padding) and compute aperture
    masses for every halo assigned to it. Runs in a worker process.
    """
    (Swiftfile, cell_lo_comov, cell_hi_comov, halo_list) = batch
    # halo_list: list of (local_index, c_phys[3], r50_phys)

    import numpy      as np
    import unyt
    import swiftsimio as sw

    region = [[cell_lo_comov[k], cell_hi_comov[k]] * unyt.kpc for k in range(3)]

    # safe_padding isn't available on all swiftsimio versions -- fall back
    # gracefully if the installed version predates it. This is harmless
    # either way: when a snapshot has per-cell bounding-box metadata
    # (MinPositions/MaxPositions, as COLIBRE snapshots do), swiftsimio
    # ignores safe_padding and uses the true particle bounds regardless.
    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
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

    n_loaded = len(mass_stars_val) + len(mass_dm_val)

    results = []
    for (local_i, c_phys, r50_phys) in halo_list:
        r1_sq = r50_phys**2
        r2_sq = (2.0 * r50_phys)**2
        r3_sq = (3.0 * r50_phys)**2

        mstar_1 = mstar_2 = mstar_3 = 0.0
        mdm_1   = mdm_2   = mdm_3   = 0.0

        # No periodic wrap needed: the whole read is one cell's exact
        # interior, and this halo's aperture was pre-checked to fit
        # inside it.
        if len(pos_stars_val) > 0:
            dx  = pos_stars_val - c_phys
            dr2 = np.sum(dx**2, axis=1)
            in3 = dr2 <= r3_sq
            m   = mass_stars_val[in3]
            d2  = dr2[in3]
            mstar_1 = np.sum(m[d2 <= r1_sq])
            mstar_2 = np.sum(m[d2 <= r2_sq])
            mstar_3 = np.sum(m)

        if len(pos_dm_val) > 0:
            dx  = pos_dm_val - c_phys
            dr2 = np.sum(dx**2, axis=1)
            in3 = dr2 <= r3_sq
            m   = mass_dm_val[in3]
            d2  = dr2[in3]
            mdm_1 = np.sum(m[d2 <= r1_sq])
            mdm_2 = np.sum(m[d2 <= r2_sq])
            mdm_3 = np.sum(m)

        results.append((local_i, mstar_1, mdm_1, mstar_2, mdm_2, mstar_3, mdm_3))

    return results, n_loaded, rss_gb()


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

        rhalf_stars = soap.bound_subhalo.half_mass_radius_stars[lhalo]
        halo_centre = soap.input_halos.halo_centre[lhalo]
        TrackId     = soap.input_halos_hbtplus.track_id[lhalo]

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()

        r50_vals    = rhalf_stars.value
        centre_vals = halo_centre.value

        # --- Metadata + cell grid (cheap, no particle data touched) ---
        mask0       = sw.mask(Swiftfile, spatial_only=True)
        meta        = mask0.metadata
        ScaleFactor = float(meta.scale_factor)

        cell_size_val = float(mask0.cell_size.to('kpc').value[0])   # 3125 kpc

        boxsize_comoving = meta.boxsize.copy()
        boxsize_comoving.convert_to_units('kpc')
        boxsize_comoving_val = float(boxsize_comoving.value[0])

        n_cells_per_dim = int(round(boxsize_comoving_val / cell_size_val))

        print(f'  Cell size: {cell_size_val:.2f} kpc  '
              f'({n_cells_per_dim}^3 = {n_cells_per_dim**3} cells)')
        print('  Using', N_WORKERS, 'worker processes')

        # --- Assign each halo to a home cell; skip if its aperture
        #     pokes outside that cell's bounds on any axis ---
        cell_groups = defaultdict(list)   # cell_idx tuple -> list of (local_i, c_phys, r50_phys)
        n_skipped   = 0

        for i in range(len(lhalo)):
            c_phys   = centre_vals[i]
            r50_phys = r50_vals[i]
            r3_comov = (3.0 * r50_phys) / ScaleFactor
            c_comov  = c_phys / ScaleFactor

            idx = np.floor(c_comov / cell_size_val).astype(int) % n_cells_per_dim
            local = c_comov - idx * cell_size_val   # position within the cell, [0, cell_size)

            safe = np.all((local - r3_comov >= 0.0) & (local + r3_comov <= cell_size_val))
            if safe:
                cell_groups[tuple(idx)].append((i, c_phys, r50_phys))
            else:
                n_skipped += 1

        n_included = len(lhalo) - n_skipped
        print(f'  {n_included}/{len(lhalo)} halos fully contained in their home cell '
              f'({n_skipped} skipped -- aperture crosses a cell boundary), '
              f'spread across {len(cell_groups)} occupied cells')

        # --- Build one task per occupied cell ---
        tasks = []
        for idx, halo_list in cell_groups.items():
            cell_lo = np.array(idx) * cell_size_val
            cell_hi = cell_lo + cell_size_val
            tasks.append((Swiftfile, cell_lo, cell_hi, halo_list))

        mstar_1r50 = np.full(len(lhalo), np.nan) ; mdm_1r50 = np.full(len(lhalo), np.nan)
        mstar_2r50 = np.full(len(lhalo), np.nan) ; mdm_2r50 = np.full(len(lhalo), np.nan)
        mstar_3r50 = np.full(len(lhalo), np.nan) ; mdm_3r50 = np.full(len(lhalo), np.nan)

        max_n_loaded = 0
        max_rss_seen = 0.0

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(measure_cell_batch, t): t for t in tasks}
            ndone_cells = 0
            ndone_halos = 0
            for fut in as_completed(futures):
                results, n_loaded, worker_rss = fut.result()
                for (local_i, ms1, md1, ms2, md2, ms3, md3) in results:
                    mstar_1r50[local_i] = ms1 ; mdm_1r50[local_i] = md1
                    mstar_2r50[local_i] = ms2 ; mdm_2r50[local_i] = md2
                    mstar_3r50[local_i] = ms3 ; mdm_3r50[local_i] = md3
                ndone_halos += len(results)
                max_n_loaded = max(max_n_loaded, n_loaded)
                max_rss_seen = max(max_rss_seen, worker_rss)

                ndone_cells += 1
                if ndone_cells % 20 == 0 or ndone_cells == len(tasks):
                    print(f'  ... {ndone_cells}/{len(tasks)} cells done, '
                          f'{ndone_halos}/{n_included} halos done  '
                          f'(max particles in a single cell read so far: {max_n_loaded}, '
                          f'max worker RSS so far: {max_rss_seen:.1f} GB)', flush=True)

        # --- Compute stellar-to-dark matter mass ratios (M_* / M_DM) ---
        fDM_1r50 = np.where(mdm_1r50 > 0, mstar_1r50 / mdm_1r50, np.nan)
        fDM_2r50 = np.where(mdm_2r50 > 0, mstar_2r50 / mdm_2r50, np.nan)
        fDM_3r50 = np.where(mdm_3r50 > 0, mstar_3r50 / mdm_3r50, np.nan)

        fn = ("/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+fname+ext3+f"_chunk{args.chunk_id:04d}.hdf5")
        #fn = (BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3 + f"_chunk{args.chunk_id:04d}.hdf5")
        print('\n Writing to:', fn)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        output = h5.File(fn, "w")
        grp0   = output.create_group("Header")
        grp1   = output.create_group("HaloData")

        grp0.create_dataset('Redshift',        data=1./ScaleFactor - 1, dtype='float')
        grp0.create_dataset('ChunkId',         data=args.chunk_id,      dtype='int')
        grp0.create_dataset('NChunks',         data=args.n_chunks,      dtype='int')
        grp0.create_dataset('CellSizeKpc',     data=cell_size_val,      dtype='float')
        grp0.create_dataset('NHalosSkipped',   data=n_skipped,          dtype='int')
        grp0.create_dataset('NHalosIncluded',  data=n_included,         dtype='int')

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
