import os
import sys
import argparse
import warnings
from   collections                  import defaultdict
import numpy                        as np
from   concurrent.futures           import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# CO-ROTATION RADIUS OF BARRED GALAXIES
# --------------------------------------------------------------------------
# The co-rotation radius R_CR is where the circular speed of the disc
# matches the bar pattern speed:
#
#       Vc(R_CR) = Omega_p * R_CR      <=>      Vc(R) / R = Omega_p
#
# Method:
#   1. Pattern speeds are READ from your existing bar-properties file
#      (matched to the SOAP catalogue by TrackId) -- nothing is
#      recalculated here. Only galaxies with a valid pattern speed are
#      analysed.
#   2. For each barred galaxy, build the enclosed-mass profile M(<r) of
#      ALL stars + gas + dark matter on a log-spaced radial grid, using
#      the same cell-batched spatial masking as the fDM / sigma scripts
#      (read each SWIFT top-level cell once, share it between halos).
#   3. Vc(r) = sqrt(G M(<r) / r)   (spherical approximation).
#   4. Find where Vc(r)/r drops through Omega_p (linear interpolation in
#      log r). Status codes below say what happened for each galaxy.
#
# Known simplifications (worth stating in any paper/thesis):
#   - Vc is the SPHERICAL estimate, not the in-plane radial force of the
#     flattened disc. It slightly overestimates Vc in the disc plane
#     relative to the true circular speed for a thin disc, so R_CR is
#     mildly biased high compared with a force-based calculation.
#   - Black-hole particles are not included (negligible mass on these
#     scales; add data.black_holes if you want strict completeness).
#   - Halos whose R_MAX aperture crosses a SWIFT cell boundary are
#     skipped (status 3, NaN), as in the previous cell-batched scripts.
#
# STATUS CODES (HaloData/Status)
#   0  OK: co-rotation radius found inside the grid
#   1  Vc/R < Omega_p already at R_MIN  -> R_CR is inside the radius we can
#      trust (a very fast bar, or a resolution-limited centre)
#   2  Vc/R > Omega_p all the way out to R_MAX -> R_CR lies beyond the grid
#   3  skipped: aperture crosses a cell boundary
#   4  invalid inputs / non-finite profile
# ==========================================================================

# --------------------------------------------------------------------------
# ---- Simulation information ----
BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

# Where outputs live (same place your merge scripts read from)
OutBase      = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"
fname        = "CoRotation_"

Nstar_min    = 5e3
Nstar_max    = 1e10

# --------------------------------------------------------------------------
# ---- BAR PROPERTIES FILE  *** CHECK ALL OF THESE -- I HAVE NOT SEEN IT *** ----
# I could not see how your pattern speeds are stored (the Calculate_Bar_Prof.py
# script only saves A2/Phi2 profiles), so these are PLACEHOLDERS.
def bar_file_for(Dir, RunDir, ext3):
    return OutBase + Dir + RunDir + "BarProperties_" + ext3 + "_merged.hdf5"   # <-- placeholder name

BAR_GROUP              = "HaloData"       # <-- group holding the per-halo arrays
BAR_TRACKID            = "TrackId"        # <-- dataset with HBT track ids
BAR_PATTERN_SPEED      = "PatternSpeed"   # <-- dataset with pattern speeds
BAR_PATTERN_SPEED_UNIT = "km/s/kpc"       # <-- units that dataset is stored in
BAR_FLAG               = None             # optional: name of a "is barred" dataset
                                          #   (non-zero = barred). None = use any
                                          #   finite, non-zero pattern speed.

# --------------------------------------------------------------------------
# ---- Radial grid for the rotation curve (PHYSICAL kpc) ----
R_MIN_KPC = 1.0     # innermost radius trusted; set to a few softening lengths
R_MAX_KPC = 50.0    # outermost radius; also the aperture that must fit in one cell
N_RAD     = 150     # log-spaced radii between the two

# G in kpc (km/s)^2 / Msun
G_KPC_KMS2_MSUN = 4.300917270e-6

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


# ==========================================================================
# Pure-numpy helpers (no simulation dependencies, so they can be unit-tested)
# ==========================================================================
def enclosed_mass(pos, mass, centre, edges):
    """
    Cumulative mass of the given particles within each radius in `edges`
    (ascending, physical kpc) of `centre`. The value at each edge is exact
    (it is a cumulative sum of an exact histogram), not interpolated.
    """
    rmax = edges[-1]
    sel  = np.abs(pos[:, 0] - centre[0]) <= rmax
    sel &= np.abs(pos[:, 1] - centre[1]) <= rmax
    sel &= np.abs(pos[:, 2] - centre[2]) <= rmax
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return np.zeros(len(edges))

    d = pos[idx] - centre
    r = np.sqrt(np.sum(d * d, axis=1))
    m = mass[idx]
    inside = r <= rmax
    hist, _ = np.histogram(r[inside], bins=np.concatenate(([0.0], edges)),
                           weights=m[inside])
    return np.cumsum(hist)


def circular_velocity(radii, menc):
    """Spherical circular speed in km/s: sqrt(G M(<r) / r)."""
    return np.sqrt(G_KPC_KMS2_MSUN * menc / radii)


def find_corotation(radii, vc, omega_p):
    """
    Solve Vc(R)/R = omega_p on the grid. Returns
    (R_CR [kpc or nan], status, n_crossings).
    Uses the FIRST downward crossing from small radius, with linear
    interpolation in log r. n_crossings counts every sign change of
    Vc/R - omega_p, so you can flag galaxies with a non-monotonic curve.
    """
    f = vc / radii - omega_p
    if (not np.all(np.isfinite(f))) or (not np.isfinite(omega_p)) or omega_p <= 0:
        return np.nan, 4, 0

    n_cross = int(np.sum((f[:-1] >= 0) != (f[1:] >= 0)))

    if f[0] <= 0:
        return np.nan, 1, n_cross
    neg = np.flatnonzero(f < 0)
    if neg.size == 0:
        return np.nan, 2, n_cross

    j      = neg[0]                     # first point below omega_p; f[j-1] >= 0
    lr0    = np.log(radii[j - 1])
    lr1    = np.log(radii[j])
    t      = f[j - 1] / (f[j - 1] - f[j])
    return float(np.exp(lr0 + t * (lr1 - lr0))), 0, n_cross


# ==========================================================================
# Worker: one SWIFT top-level cell -> enclosed-mass profiles of its halos
# ==========================================================================
def measure_cell_batch(batch):
    (Swiftfile, cell_lo_comov, cell_hi_comov, halo_list, edges) = batch
    # halo_list: list of (local_index, c_phys[3])

    import numpy      as np
    import unyt
    import swiftsimio as sw

    region = [[cell_lo_comov[k], cell_hi_comov[k]] * unyt.kpc for k in range(3)]

    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)
    data = sw.load(Swiftfile, mask=mask)

    species = {}
    for name, group in (("star", data.stars), ("gas", data.gas), ("dm", data.dark_matter)):
        pos  = group.coordinates
        mass = group.masses
        pos.convert_to_units('kpc')   ; pos.convert_to_physical()
        mass.convert_to_units('Msun') ; mass.convert_to_physical()
        species[name] = (pos.value, mass.value)

    n_loaded = sum(len(v[1]) for v in species.values())

    results = []
    for (local_i, c_phys) in halo_list:
        prof = {name: enclosed_mass(p, m, c_phys, edges)
                for name, (p, m) in species.items()}
        results.append((local_i, prof["star"], prof["gas"], prof["dm"]))

    return results, n_loaded


# ==========================================================================
if __name__ == "__main__":

    import unyt
    import h5py       as h5
    import swiftsimio as sw

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)

        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # ---------------- pattern speeds from the bar file ----------------
        bar_fn = bar_file_for(Dir, RunDir, ext3)
        print('Bar file  :', bar_fn)
        if not os.path.exists(bar_fn):
            sys.exit(f"Bar properties file not found: {bar_fn}\n"
                     f"Edit bar_file_for() at the top of this script.")

        with h5.File(bar_fn, 'r') as f:
            if BAR_GROUP not in f:
                sys.exit(f"Group '{BAR_GROUP}' not in bar file. Top-level groups: {list(f.keys())}")
            grp = f[BAR_GROUP]
            for needed in (BAR_TRACKID, BAR_PATTERN_SPEED):
                if needed not in grp:
                    sys.exit(f"Dataset '{needed}' not in '{BAR_GROUP}'. "
                             f"Available datasets: {list(grp.keys())}\n"
                             f"Set BAR_TRACKID / BAR_PATTERN_SPEED at the top of this script.")
            bar_track = np.asarray(grp[BAR_TRACKID][()]).astype(np.int64)
            bar_wp    = np.asarray(grp[BAR_PATTERN_SPEED][()], dtype=float)
            bar_flag  = None
            if BAR_FLAG is not None:
                bar_flag = np.asarray(grp[BAR_FLAG][()]) != 0

        bar_wp = unyt.unyt_array(bar_wp, BAR_PATTERN_SPEED_UNIT).to('km/s/kpc').value
        valid  = np.isfinite(bar_wp) & (bar_wp != 0)
        if bar_flag is not None:
            valid &= bar_flag
        wp_by_track = dict(zip(bar_track[valid].tolist(), bar_wp[valid].tolist()))
        print(f'  {len(wp_by_track)} galaxies with a valid pattern speed in the bar file')

        # ---------------- SOAP selection, restricted to barred galaxies ----------------
        print('SOAP file :', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo_sel     = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
        track_sel     = np.asarray(soap.input_halos_hbtplus.track_id[lhalo_sel]).astype(np.int64)

        is_barred = np.array([t in wp_by_track for t in track_sel.tolist()], dtype=bool)
        lhalo_all = lhalo_sel[is_barred]
        print(f'  {len(lhalo_all)} of {len(lhalo_sel)} selected galaxies are barred '
              f'({len(wp_by_track) - len(lhalo_all)} bar-file entries not found in this selection)')

        chunk_bounds = np.linspace(0, len(lhalo_all), args.n_chunks + 1, dtype=int)
        c0, c1 = chunk_bounds[args.chunk_id], chunk_bounds[args.chunk_id + 1]
        lhalo  = lhalo_all[c0:c1]
        print(f'- snapshot: {snap}   chunk {args.chunk_id}/{args.n_chunks}   '
              f'barred halos {c0}:{c1} of {len(lhalo_all)} ({len(lhalo)} in this chunk)')

        rhalf_stars = soap.bound_subhalo.half_mass_radius_stars[lhalo]
        halo_centre = soap.input_halos.halo_centre[lhalo]
        TrackId     = np.asarray(soap.input_halos_hbtplus.track_id[lhalo]).astype(np.int64)

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()

        r50_vals    = rhalf_stars.value
        centre_vals = halo_centre.value
        omega_p     = np.array([wp_by_track[t] for t in TrackId.tolist()])   # signed, km/s/kpc

        # ---------------- metadata + cell grid ----------------
        mask0       = sw.mask(Swiftfile, spatial_only=True)
        meta        = mask0.metadata
        ScaleFactor = float(meta.scale_factor)

        cell_size_val = float(mask0.cell_size.to('kpc').value[0])
        boxsize_comoving = meta.boxsize.copy()
        boxsize_comoving.convert_to_units('kpc')
        n_cells_per_dim = int(round(float(boxsize_comoving.value[0]) / cell_size_val))

        print(f'  Cell size: {cell_size_val:.2f} kpc ({n_cells_per_dim}^3 cells); '
              f'profile grid {R_MIN_KPC}-{R_MAX_KPC} kpc, {N_RAD} radii; {N_WORKERS} workers')

        edges = np.geomspace(R_MIN_KPC, R_MAX_KPC, N_RAD)

        # ---------------- assign halos to home cells ----------------
        cell_groups = defaultdict(list)
        n_skipped   = 0
        r_comov     = R_MAX_KPC / ScaleFactor

        for i in range(len(lhalo)):
            c_comov = centre_vals[i] / ScaleFactor
            idx     = np.floor(c_comov / cell_size_val).astype(int) % n_cells_per_dim
            local   = c_comov - idx * cell_size_val
            if np.all((local - r_comov >= 0.0) & (local + r_comov <= cell_size_val)):
                cell_groups[tuple(idx)].append((i, centre_vals[i]))
            else:
                n_skipped += 1

        n_included = len(lhalo) - n_skipped
        print(f'  {n_included}/{len(lhalo)} halos fully contained in their home cell '
              f'({n_skipped} skipped), across {len(cell_groups)} occupied cells')

        tasks = []
        for idx, halo_list in cell_groups.items():
            cell_lo = np.array(idx) * cell_size_val
            tasks.append((Swiftfile, cell_lo, cell_lo + cell_size_val, halo_list, edges))

        Menc_star = np.full((len(lhalo), N_RAD), np.nan)
        Menc_gas  = np.full((len(lhalo), N_RAD), np.nan)
        Menc_dm   = np.full((len(lhalo), N_RAD), np.nan)

        max_n_loaded = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(measure_cell_batch, t): t for t in tasks}
            ndone_cells = ndone_halos = 0
            for fut in as_completed(futures):
                results, n_loaded = fut.result()
                for (local_i, ms, mg, md) in results:
                    Menc_star[local_i] = ms
                    Menc_gas[local_i]  = mg
                    Menc_dm[local_i]   = md
                ndone_halos += len(results)
                max_n_loaded = max(max_n_loaded, n_loaded)
                ndone_cells += 1
                if ndone_cells % 20 == 0 or ndone_cells == len(tasks):
                    print(f'  ... {ndone_cells}/{len(tasks)} cells, {ndone_halos}/{n_included} halos '
                          f'(max particles in one cell read: {max_n_loaded})', flush=True)

        # ---------------- solve for the co-rotation radius ----------------
        Menc_tot = Menc_star + Menc_gas + Menc_dm
        Vc       = circular_velocity(edges[None, :], Menc_tot)

        R_CR      = np.full(len(lhalo), np.nan)
        status    = np.full(len(lhalo), 4, dtype=int)
        n_cross   = np.zeros(len(lhalo), dtype=int)
        skipped   = np.all(np.isnan(Menc_tot), axis=1)
        status[skipped] = 3

        for i in np.flatnonzero(~skipped):
            R_CR[i], status[i], n_cross[i] = find_corotation(edges, Vc[i], abs(omega_p[i]))

        Vc_CR = abs(omega_p) * R_CR           # = Vc at R_CR by construction, km/s

        print('\n  Status counts: ' + ', '.join(
            f'{s}: {int(np.sum(status == s))}' for s in range(5)))
        ok = status == 0
        if np.any(ok):
            print(f'  R_CR (kpc) for status 0: min={np.nanmin(R_CR[ok]):.2f}  '
                  f'median={np.nanmedian(R_CR[ok]):.2f}  max={np.nanmax(R_CR[ok]):.2f}')

        # ---------------- write ----------------
        fn = OutBase + Dir + RunDir + fname + ext3 + f"_chunk{args.chunk_id:04d}.hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        with h5.File(fn, "w") as output:
            grp0 = output.create_group("Header")
            grp1 = output.create_group("HaloData")
            grp2 = output.create_group("Profiles")

            grp0.create_dataset('Redshift',        data=1./ScaleFactor - 1, dtype='float')
            grp0.create_dataset('ChunkId',         data=args.chunk_id,      dtype='int')
            grp0.create_dataset('NChunks',         data=args.n_chunks,      dtype='int')
            grp0.create_dataset('CellSizeKpc',     data=cell_size_val,      dtype='float')
            grp0.create_dataset('NHalosSkipped',   data=n_skipped,          dtype='int')
            grp0.create_dataset('NHalosIncluded',  data=n_included,         dtype='int')
            grp0.create_dataset('RMinKpc',         data=R_MIN_KPC,          dtype='float')
            grp0.create_dataset('RMaxKpc',         data=R_MAX_KPC,          dtype='float')

            grp1.create_dataset('TrackId',         data=TrackId,            dtype='int')
            grp1.create_dataset('r50_stars',       data=r50_vals,           dtype='float')
            grp1.create_dataset('PatternSpeed_kms_per_kpc', data=omega_p,   dtype='float')
            grp1.create_dataset('R_CR_kpc',        data=R_CR,               dtype='float')
            grp1.create_dataset('R_CR_over_r50',   data=R_CR / r50_vals,    dtype='float')
            grp1.create_dataset('Vc_at_CR_kms',    data=Vc_CR,              dtype='float')
            grp1.create_dataset('Status',          data=status,             dtype='int')
            grp1.create_dataset('NCrossings',      data=n_cross,            dtype='int')

            grp2.create_dataset('RadiusKpc',       data=edges,              dtype='float')
            grp2.create_dataset('Menc_stars',      data=Menc_star,          dtype='float')
            grp2.create_dataset('Menc_gas',        data=Menc_gas,           dtype='float')
            grp2.create_dataset('Menc_dm',         data=Menc_dm,            dtype='float')
            grp2.create_dataset('Vc_kms',          data=Vc,                 dtype='float')

    sys.exit()
