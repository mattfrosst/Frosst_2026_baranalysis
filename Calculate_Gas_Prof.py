import os
import sys
import argparse
import resource
import warnings
from   collections                  import defaultdict
from   concurrent.futures           import ProcessPoolExecutor, as_completed
import numpy                        as     np
import unyt
import h5py                         as     h5
import swiftsimio                   as     sw
from   scipy.spatial                import cKDTree
from   colibre_utility              import rotation_matrix_from_vectors
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
# CELL-BATCHED, as Calculate_fDM_rhalf_chunked.py:
#   * each galaxy is assigned to its home top-level SWIFT cell (3.125 Mpc for L200_m6);
#   * pass 1: every occupied cell is read once (spatial mask, no padding) and all galaxies whose
#     search region fits inside that cell are measured from that one read;
#   * pass 2: galaxies whose search region crosses a cell boundary get their own small masked
#     read of just their search region (instead of being skipped, which would bias against
#     large galaxies);
#   * galaxies whose search region crosses the box edge (~0.01%) are skipped, flagged and
#     counted;
#   * cells run in parallel worker processes (SLURM_CPUS_PER_TASK);
#   * --chunk-id / --n-chunks split the galaxies by home cell, so each array task reads its own
#     cells. Combine the chunks with merge_Gas_Prof_chunks.py.
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
QUANTITIES   = ['GasMass', 'HIMass', 'H2Mass', 'SFR']

N_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

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
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == 'darwin' else peak / 1e6   # bytes on macOS, kB on Linux

# --------------------------------------------------------------------------
# ---- Profile helpers ----
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

def profiles_from_particles(pos, weights, halo_list):
    # Measure every galaxy in halo_list from one set of particles (no periodic wrap needed:
    # each read is a region inside the box that contains the galaxies' whole search spheres).
    # halo_list entries: (local_i, centre_phys[3], r50, r_search, trans[3,3])
    results = []
    tree = cKDTree(pos) if len(pos) > 0 else None
    for (local_i, c, r50, r_search, trans) in halo_list:
        if tree is None:
            sums, counts = np.zeros((len(R_EDGES) - 1, len(QUANTITIES))), np.zeros(len(R_EDGES) - 1, int)
        else:
            idx = tree.query_ball_point(c, r_search)
            sums, counts = bin_profile(pos[idx] - c, weights[idx], trans, r50)
        results.append((local_i, sums, counts))
    return results

def load_gas_region(Swiftfile, lo_comov, hi_comov):
    # Masked read of the gas in one region [comoving kpc]; returns physical positions [kpc] and
    # weights (gas, HI, H2 mass [Msun], SFR [Msun/yr]).
    region = [[lo_comov[k], hi_comov[k]] * unyt.kpc for k in range(3)]
    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)
    data = sw.load(Swiftfile, mask=mask)
    gas  = data.gas

    pos = gas.coordinates ; pos.convert_to_units('kpc') ; pos.convert_to_physical()
    m   = gas.masses      ; m.convert_to_units('Msun')  ; m.convert_to_physical()
    m   = m.value
    X_H  = np.asarray(gas.element_mass_fractions.hydrogen)
    f_HI = np.asarray(gas.species_fractions.HI)
    f_H2 = np.asarray(gas.species_fractions.H2)
    sfr  = np.clip(gas.star_formation_rates.to_physical().to_value(unyt.Msun/unyt.yr), 0, None)
    weights = np.column_stack([m, m*X_H*f_HI, 2.0*m*X_H*f_H2, sfr])
    return pos.value, weights

def measure_region(task):
    # Worker: read one region and measure every galaxy assigned to it.
    (Swiftfile, lo_comov, hi_comov, halo_list) = task
    pos, weights = load_gas_region(Swiftfile, lo_comov, hi_comov)
    return profiles_from_particles(pos, weights, halo_list), len(pos), peak_mem_gb()

def assign_cells(centres_comov, r_search_comov, cell_size, n_cells, boxsize):
    # Home cell of each galaxy, and whether its search sphere fits inside that cell (pass 1),
    # only crosses cell boundaries (pass 2) or crosses the box edge (skipped).
    idx   = np.floor(centres_comov / cell_size).astype(int) % n_cells
    local = centres_comov - idx * cell_size
    r     = r_search_comov[:, None]
    inside   = np.all((local - r >= 0.0) & (local + r <= cell_size), axis=1)
    box_edge = np.any((centres_comov - r < 0.0) | (centres_comov + r > boxsize), axis=1)
    flat  = (idx[:, 0] * n_cells + idx[:, 1]) * n_cells + idx[:, 2]
    return idx, flat, inside, box_edge

# --------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, default=snap,
                        help=f"snapshot number to analyse (default: {snap}, set at the top of this script)")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    snap = args.snap

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)
        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # --- Read SOAP halo data (all selected galaxies; the chunk is chosen by home cell below)
        print('SOAP file: ', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
        del Nstar_subhalo ; free_column(soap.bound_subhalo, 'number_of_star_particles')
        n_gal = len(lhalo)
        print('- snapshot        :', snap)
        print('  Number of galaxies selected: ', n_gal)

        rhalf_stars = read_selected(soap.bound_subhalo, 'half_mass_radius_stars', lhalo)
        mstar       = read_selected(soap.exclusive_sphere_50kpc, 'stellar_mass', lhalo)
        angJ_stars  = read_selected(soap.inclusive_sphere_50kpc, 'angular_momentum_stars', lhalo)
        halo_centre = read_selected(soap.input_halos, 'halo_centre', lhalo)
        is_central  = read_selected(soap.input_halos, 'is_central', lhalo)
        TrackId     = read_selected(soap.input_halos_hbtplus, 'track_id', lhalo)

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        mstar.convert_to_units('Msun')      ; mstar.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()
        print(f'  Peak memory after SOAP reads: {peak_mem_gb():.1f} GB')

        r50      = rhalf_stars.value
        centres  = halo_centre.value
        angJ     = angJ_stars.value
        r_search = r50 * np.hypot(R_EDGES[-1], ZCUT_RHALF)   # 3D radius enclosing R < R_MAX, |z| < ZCUT

        # --- Cell grid (metadata only, no particle data) ---
        mask0       = sw.mask(Swiftfile, spatial_only=True)
        meta        = mask0.metadata
        ScaleFactor = float(meta.scale_factor)
        redshift    = 1.0/ScaleFactor - 1.0
        cell_size   = float(mask0.cell_size.to('kpc').value[0])                 # comoving kpc
        box_comov   = meta.boxsize.copy() ; box_comov.convert_to_units('kpc')
        box_comov   = float(box_comov.value[0])
        n_cells     = int(round(box_comov / cell_size))
        print(f'  Cell size: {cell_size:.1f} comoving kpc ({n_cells}^3 cells); using {N_WORKERS} workers')

        c_comov = centres / ScaleFactor
        r_comov = r_search / ScaleFactor
        cell_idx, cell_flat, inside, box_edge = assign_cells(c_comov, r_comov, cell_size, n_cells, box_comov)

        # --- This chunk's galaxies: chosen by home cell, so chunks read disjoint cells ---
        in_chunk = (cell_flat % args.n_chunks) == args.chunk_id
        mine     = np.where(in_chunk)[0]
        pass1    = mine[inside[mine]]
        pass2    = mine[~inside[mine] & ~box_edge[mine]]
        skipped  = mine[~inside[mine] & box_edge[mine]]
        print(f'  chunk {args.chunk_id}/{args.n_chunks}: {len(mine)} galaxies '
              f'({len(pass1)} inside their home cell, {len(pass2)} crossing a cell boundary, '
              f'{len(skipped)} skipped at the box edge)')

        def halo_entry(i):
            return (i, centres[i], r50[i], r_search[i], rotation_matrix_from_vectors(angJ[i], [0, 0, 1]))

        # --- Tasks: one per occupied cell (pass 1), one per boundary-crossing galaxy (pass 2) ---
        cell_groups = defaultdict(list)
        for i in pass1:
            cell_groups[tuple(cell_idx[i])].append(halo_entry(i))
        tasks = []
        for idx, halo_list in cell_groups.items():
            lo = np.array(idx) * cell_size
            tasks.append((Swiftfile, lo, lo + cell_size, halo_list))
        for i in pass2:
            tasks.append((Swiftfile, c_comov[i] - r_comov[i], c_comov[i] + r_comov[i], [halo_entry(i)]))
        tasks.sort(key=lambda t: -len(t[3]))   # biggest cells first, for load balancing
        print(f'  {len(cell_groups)} cell reads + {len(pass2)} single-galaxy reads')

        # --- output arrays for this chunk (NaN = not measured) ---
        n_bins = len(R_EDGES) - 1
        prof   = np.full((len(mine), n_bins, len(QUANTITIES)), np.nan)
        npart  = np.full((len(mine), n_bins), -1, dtype=np.int64)
        row_of = {g: k for k, g in enumerate(mine)}   # global index -> row in this chunk's arrays

        max_loaded, max_rss, n_done = 0, 0.0, 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(measure_region, t) for t in tasks]
            for it, fut in enumerate(as_completed(futures)):
                results, n_loaded, worker_rss = fut.result()
                for (i, sums, counts) in results:
                    prof[row_of[i]]  = sums
                    npart[row_of[i]] = counts
                n_done += len(results)
                max_loaded = max(max_loaded, n_loaded); max_rss = max(max_rss, worker_rss)
                if (it + 1) % 200 == 0 or it + 1 == len(tasks):
                    print(f'  ... {it+1}/{len(tasks)} reads done, {n_done}/{len(pass1)+len(pass2)} galaxies '
                          f'(max gas particles in one read: {max_loaded}, max worker memory: {max_rss:.1f} GB)', flush=True)

        # --- Write to hdf5 ---
        suffix = f"_chunk{args.chunk_id:04d}" if args.n_chunks > 1 else ""
        fn = OutBase+Dir+RunDir+fname+ext3+suffix+".hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with h5.File(fn, "w") as output:
            grp0 = output.create_group("Header")
            grp0.create_dataset('Redshift',        data=redshift)
            grp0.create_dataset('ZCUT_RHALF',      data=ZCUT_RHALF)
            grp0.create_dataset('ChunkId',         data=args.chunk_id)
            grp0.create_dataset('NChunks',         data=args.n_chunks)
            grp0.create_dataset('NGalTotal',       data=n_gal)
            grp0.create_dataset('CellSizeKpc',     data=cell_size)
            grp0.create_dataset('NHalosSkipped',   data=len(skipped))
            grp0.create_dataset('NHalosIncluded',  data=len(pass1) + len(pass2))
            grp0.attrs['description'] = ('Face-on gas profiles in bins of R/r50_stars within |z| < ZCUT_RHALF*r50; '
                                         'bin sums of gas, HI, H2 mass [Msun] and SFR [Msun/yr]. '
                                         'GlobalIndex is the position in the full Nstar-selected galaxy list.')

            grp1 = output.create_group("HaloData")
            grp1.create_dataset('GlobalIndex', data=mine,                         dtype='int')
            grp1.create_dataset('TrackId',     data=np.asarray(TrackId)[mine],    dtype='int')
            grp1.create_dataset('is_central',  data=np.asarray(is_central)[mine], dtype='int')
            grp1.create_dataset('StellarMass', data=mstar.value[mine],            dtype='float')
            grp1.create_dataset('r50_stars',   data=r50[mine],                    dtype='float')
            grp1.create_dataset('Skipped',     data=np.isin(mine, skipped),       dtype='bool')

            grp2 = output.create_group("Profiles")
            grp2.create_dataset('R_edges_rhalf', data=R_EDGES, dtype='float')
            for q, name in enumerate(QUANTITIES):
                grp2.create_dataset(name, data=prof[:, :, q], dtype='float')
            grp2.create_dataset('NumGasPart', data=npart, dtype='int')
        print(f' Done. Peak memory (main process): {peak_mem_gb():.1f} GB')

    sys.exit()
