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
# Pipeline position (all outputs live in OutBase + BoxDir + RunDir):
#   Calculate_Bar_Prof.py   -> Stars_Mproj_Bar_Prof_<snap>.hdf5   (A2/Phi2 profiles)
#   Calculate_Bar_Region.py -> Stars_Mproj_Bar_Region_<snap>.hdf5 (isbarred, R0/R1 bar, rhalf units)
#   Calculate_Bar_Omega.py  -> Stars_Mproj_Bar_Omega_<snap>.hdf5  (pattern speed, km/s/kpc)
#   Calculate_Bar_Rcorot.py -> CoRotation_<snap>_chunkNNNN.hdf5   (this script)
#
# Method:
#   1. Pattern speeds (and their errors) are READ from the Omega file and
#      the bar flag + bar length from the Region file, both matched to
#      the SOAP catalogue by TrackId -- nothing is recalculated here.
#      EVERY galaxy in the SOAP selection gets a row in the output (so it
#      lines up with the Prof / Region / Omega files), but only those
#      with isbarred == 1 and a finite, non-zero pattern speed are
#      analysed; the rest are NaN with Status 5 or 6 and cost no I/O.
#   2. For each barred galaxy, build the enclosed-mass profile M(<r) of
#      ALL stars + gas + dark matter on EXACTLY the radial grid of the bar
#      profiles (Calculate_Bar_Prof.py): edges 10**linspace(-1,1,41) r50,
#      i.e. 0.1-10 r50 in 0.05 dex steps, within the same 50 kpc aperture
#      (grid points beyond 50 kpc are NaN). Particles are read with
#      cell-batched spatial masking (each SWIFT top-level cell read once).
#   3. Vc(r) = sqrt(G M(<r) / r)   (spherical approximation).
#   4. Find where Vc(r)/r drops through Omega_p (ln Omega_c interpolated
#      linearly in ln r). Status codes below say what happened for each
#      galaxy. R_CR is also solved at |Omega_p| -/+ its statistical error
#      to give a range, and the rate parameter Rcal = R_CR / R_bar is
#      written out (R_bar = R1Bar_value * r50 from the Region file; it is a
#      bar-profile bin edge and so lies exactly on the grid).
#   5. Omega_c(R_bar) = Vc(R_bar)/R_bar and q = sigma_Omega / Omega_c(R_bar)
#      are written out, so every downstream script uses the same q.
#
# Known simplifications:
#   - Vc is the SPHERICAL estimate, not the in-plane radial force of the
#     flattened disc. Where the disc dominates, a thin disc has a HIGHER
#     in-plane Vc than sqrt(GM(<r)/r) (~15% near the exponential-disc
#     peak), so R_CR here is mildly biased LOW compared with a
#     force-based calculation.
#   - Black-hole particles are not included (negligible mass on these
#     scales; add data.black_holes if you want strict completeness).
#   - Halos whose aperture, min(10 r50, 50 kpc), crosses a SWIFT cell
#     boundary are skipped (status 3, NaN), as in the previous cell-batched
#     scripts. (The bar profiles use a periodic tree, so they DO include these.)
#
# STATUS CODES (returned as HaloData/Status):
#   0  OK: co-rotation radius found inside the grid
#   1  Vc/R < Omega_p already at the innermost grid point (0.1 r50)
#      -> R_CR is inside the grid (a very fast bar, or an unresolved centre)
#   2  Vc/R > Omega_p all the way out to the last grid point inside the
#      aperture, min(10 r50, 50 kpc) -> R_CR lies beyond the grid
#   3  skipped: aperture crosses a cell boundary
#   4  invalid inputs / non-finite profile
#   5  not barred (isbarred == 0, or absent from the Region file)  -> all NaN
#   6  barred, but no valid pattern speed in the Omega file (0 or NaN) -> all NaN
#      (R_bar_kpc is still filled in for these)
# ==========================================================================

STATUS_NOT_BARRED = 5
STATUS_NO_OMEGA   = 6

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
# ---- Inputs from the previous bar analysis ----
# File prefixes match bname / wname in Calculate_Bar_Omega.py and
# Calculate_Bar_Region.py. BoxDir / RunDir / snap above must match the run
# those scripts were executed on.
oname        = "Stars_Mproj_Bar_Omega_"    # Calculate_Bar_Omega.py output
wname        = "Stars_Mproj_Bar_Region_"   # Calculate_Bar_Region.py output

def omega_file_for(Dir, RunDir, ext3):
    return OutBase + Dir + RunDir + oname + ext3 + ".hdf5"

def region_file_for(Dir, RunDir, ext3):
    return OutBase + Dir + RunDir + wname + ext3 + ".hdf5"


def read_halodata(fn, names):
    """Read the given HaloData datasets from one of the bar-analysis files."""
    import h5py as h5
    if not os.path.exists(fn):
        sys.exit(f"Input file not found: {fn}\nCheck BoxDir / RunDir / snap / OutBase.")
    with h5.File(fn, 'r') as f:
        grp     = f["HaloData"]
        missing = [n for n in names if n not in grp]
        if missing:
            sys.exit(f"Datasets {missing} not in {fn}:HaloData. Available: {list(grp.keys())}")
        return {n: np.asarray(grp[n][()]) for n in names}

# --------------------------------------------------------------------------
# ---- Radial grid: IDENTICAL to Calculate_Bar_Prof.py ----
# Bin edges in units of each galaxy's stellar half-mass radius r50:
#   xbin_linear = [0, 10**linspace(-1, 1, 41)]  ->  0, 0.1 r50 ... 10 r50 (0.05 dex)
# M(<r) is evaluated at the 41 non-zero edges, so every bar-profile bin edge
# (including R0Bar / R1Bar from Calculate_Bar_Region.py) is a point on this grid.
NPROF        = 41
X_RANGE      = (-1, 1)
XBIN_LINEAR  = np.append(0, 10**np.linspace(X_RANGE[0], X_RANGE[1], NPROF))   # [r50]
RADII_R50    = XBIN_LINEAR[1:]          # radii at which M(<r) and Vc are evaluated [r50]
N_RAD        = len(RADII_R50)

# Aperture: the same 50 kpc sphere used for the bar profiles (Rcut in
# Calculate_Bar_Prof.py). Grid points beyond it are NaN, as M(<r) would be
# incomplete there (and the bar-profile bins beyond 50 kpc are empty too).
RCUT_KPC     = 50.0

# G in kpc (km/s)^2 / Msun
G_KPC_KMS2_MSUN = 4.300917270e-6

# Parallel cell reads. Each worker holds one cell's particles of one species,
# so memory scales with this number; lower it with --n-workers if a chunk is
# OOM-killed (dense cells, e.g. around clusters).
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


def near_any_halo(pos, centres, rmax, block=4_000_000):
    """
    Boolean mask: True for particles inside the cube of half-width `rmax`
    (a scalar, or one value per centre) around ANY of `centres`. Works
    through `pos` in blocks so the temporary arrays stay small
    (~block x 8 bytes) even for cells holding hundreds of millions of particles.
    """
    rmax = np.broadcast_to(np.asarray(rmax, dtype=float), (len(centres),))
    keep = np.zeros(len(pos), dtype=bool)
    for s in range(0, len(pos), block):
        p   = pos[s:s + block]
        out = keep[s:s + block]
        for c, rm in zip(centres, rmax):
            sel  = np.abs(p[:, 0] - c[0]) <= rm
            sel &= np.abs(p[:, 1] - c[1]) <= rm
            sel &= np.abs(p[:, 2] - c[2]) <= rm
            out |= sel
    return keep


def halo_grid(r50_kpc, rcut_kpc=RCUT_KPC):
    """
    This galaxy's radial grid in kpc (the bar-profile edges times r50), and
    the number of leading grid points that lie inside the aperture. Points
    beyond the aperture are left out of the measurement (NaN in the output).
    """
    radii = RADII_R50 * r50_kpc
    n_in  = int(np.searchsorted(radii, rcut_kpc * (1 + 1e-12), side='right'))
    return radii, n_in


def circular_velocity(radii, menc):
    """Spherical circular speed in km/s: sqrt(G M(<r) / r)."""
    return np.sqrt(G_KPC_KMS2_MSUN * menc / radii)


def find_corotation(radii, vc, omega_p):
    """
    Solve Vc(R)/R = omega_p on the grid. Returns
    (R_CR [kpc or nan], status, n_crossings).

    Only grid points with a finite, positive Vc are used: points beyond the
    aperture are NaN, and an innermost point can have Vc = 0 if no particle
    lies within it. Uses the FIRST downward crossing from small radius,
    interpolating ln(Vc/R) linearly in ln R (exact for a power-law rotation
    curve, which matters on the 0.05 dex bar-profile grid). n_crossings
    counts every sign change of Vc/R - omega_p, to flag non-monotonic curves.
    """
    if (not np.isfinite(omega_p)) or omega_p <= 0:
        return np.nan, 4, 0
    ok = np.isfinite(vc) & (vc > 0)
    if ok.sum() < 2:
        return np.nan, 4, 0
    r  = radii[ok]
    g  = np.log(vc[ok] / r) - np.log(omega_p)   # ln Omega_c - ln Omega_p

    n_cross = int(np.sum((g[:-1] >= 0) != (g[1:] >= 0)))

    if g[0] <= 0:
        return np.nan, 1, n_cross
    neg = np.flatnonzero(g < 0)
    if neg.size == 0:
        return np.nan, 2, n_cross

    j      = neg[0]                     # first point below omega_p; g[j-1] >= 0
    lr0    = np.log(r[j - 1])
    lr1    = np.log(r[j])
    t      = g[j - 1] / (g[j - 1] - g[j])
    return float(np.exp(lr0 + t * (lr1 - lr0))), 0, n_cross


def value_at_radius(radii, values, r):
    """Log-log interpolation of a positive profile at radius r (exact on grid points)."""
    ok = np.isfinite(values) & (values > 0)
    if (not np.isfinite(r)) or r <= 0 or ok.sum() < 2 or r < radii[ok][0] or r > radii[ok][-1]:
        return np.nan
    return float(np.exp(np.interp(np.log(r), np.log(radii[ok]), np.log(values[ok]))))


# ==========================================================================
# Worker: one SWIFT top-level cell -> enclosed-mass profiles of its halos
# ==========================================================================
def measure_cell_batch(batch):
    (Swiftfile, cell_lo_comov, cell_hi_comov, halo_list) = batch
    # halo_list: list of (local_index, c_phys[3], edges_kpc) where edges_kpc are
    # this galaxy's grid points inside the aperture (the last one sets its extent)

    import numpy      as np
    import unyt
    import swiftsimio as sw

    region = [[cell_lo_comov[k], cell_hi_comov[k]] * unyt.kpc for k in range(3)]

    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)

    # Memory: a dense top-level cell can hold >2.5e8 particles. Holding all
    # three species for the whole cell at once is what got chunks OOM-killed,
    # so load ONE species at a time, keep only the particles within each
    # halo's outermost grid radius, and free the full arrays before the next
    # species. Peak memory is then ~one species' full cell, not all three.
    centres  = np.array([c for (_, c, _) in halo_list])
    rmax     = np.array([e[-1] for (_, _, e) in halo_list])
    species  = {}
    n_loaded = 0
    for name, ptype in (("star", "stars"), ("gas", "gas"), ("dm", "dark_matter")):
        # swiftsimio caches each field on its load handle, so use a fresh
        # handle for coordinates and another for masses: only one full-cell
        # array is ever alive at a time.
        data  = sw.load(Swiftfile, mask=mask)
        pos   = getattr(data, ptype).coordinates
        pos.convert_to_units('kpc')   ; pos.convert_to_physical()
        keep  = near_any_halo(pos.value, centres, rmax)
        p_keep = pos.value[keep]
        n_loaded += len(keep)
        del pos, data

        data  = sw.load(Swiftfile, mask=mask)
        mass  = getattr(data, ptype).masses
        mass.convert_to_units('Msun') ; mass.convert_to_physical()
        m_keep = mass.value[keep]
        del mass, data, keep

        species[name] = (p_keep, m_keep)

    results = []
    for (local_i, c_phys, edges) in halo_list:
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
    parser.add_argument("--n-workers", type=int, default=N_WORKERS,
                        help="parallel cell reads (default: SLURM_CPUS_PER_TASK); "
                             "lower this if the job runs out of memory")
    args = parser.parse_args()
    N_WORKERS = max(1, args.n_workers)

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)

        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # ---------------- pattern speeds + bar region from previous analysis ----------------
        omega_fn  = omega_file_for(Dir, RunDir, ext3)
        region_fn = region_file_for(Dir, RunDir, ext3)
        print('Omega file :', omega_fn)
        print('Region file:', region_fn)

        om  = read_halodata(omega_fn,  ["TrackId", "Omega_stars_kmskpc", "Omegaerr_stars_kmskpc"])
        reg = read_halodata(region_fn, ["TrackId", "isbarred", "R1Bar_value"])

        # isbarred and R1Bar (in units of r50) from the Region file
        reg_track    = reg["TrackId"].astype(np.int64)
        barred_track = set(reg_track[reg["isbarred"] != 0].tolist())
        R1bar_by_track = dict(zip(reg_track.tolist(), reg["R1Bar_value"].astype(float).tolist()))

        # Pattern speeds (km/s/kpc, signed: + = rotates with the stellar spin)
        om_track = om["TrackId"].astype(np.int64)
        om_wp    = om["Omega_stars_kmskpc"].astype(float)
        om_err   = om["Omegaerr_stars_kmskpc"].astype(float)
        valid    = np.isfinite(om_wp) & (om_wp != 0)
        valid   &= np.array([t in barred_track for t in om_track.tolist()], dtype=bool)
        wp_by_track  = dict(zip(om_track[valid].tolist(), om_wp[valid].tolist()))
        err_by_track = dict(zip(om_track[valid].tolist(), om_err[valid].tolist()))
        print(f'  {len(barred_track)} barred in Region file; '
              f'{len(wp_by_track)} of those have a valid pattern speed in the Omega file')

        # ---------------- SOAP selection (ALL galaxies, barred or not) ----------------
        print('SOAP file :', soap_file)
        soap = sw.load(soap_file)

        Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
        lhalo_all     = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
        track_all     = np.asarray(soap.input_halos_hbtplus.track_id[lhalo_all]).astype(np.int64)

        n_wp_all = int(np.sum([t in wp_by_track for t in track_all.tolist()]))
        print(f'  {len(lhalo_all)} selected galaxies; {n_wp_all} are barred with a valid '
              f'pattern speed ({len(wp_by_track) - n_wp_all} Omega-file entries not found in this selection)')

        chunk_bounds = np.linspace(0, len(lhalo_all), args.n_chunks + 1, dtype=int)
        c0, c1 = chunk_bounds[args.chunk_id], chunk_bounds[args.chunk_id + 1]
        lhalo  = lhalo_all[c0:c1]
        print(f'- snapshot: {snap}   chunk {args.chunk_id}/{args.n_chunks}   '
              f'halos {c0}:{c1} of {len(lhalo_all)} ({len(lhalo)} in this chunk)')

        rhalf_stars = soap.bound_subhalo.half_mass_radius_stars[lhalo]
        halo_centre = soap.input_halos.halo_centre[lhalo]
        TrackId     = np.asarray(soap.input_halos_hbtplus.track_id[lhalo]).astype(np.int64)

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()

        r50_vals    = rhalf_stars.value
        centre_vals = halo_centre.value
        # NaN for galaxies without a bar / pattern speed
        is_barred   = np.array([t in barred_track for t in TrackId.tolist()], dtype=bool)
        has_wp      = np.array([t in wp_by_track  for t in TrackId.tolist()], dtype=bool)
        omega_p     = np.array([wp_by_track.get(t,  np.nan) for t in TrackId.tolist()])  # signed, km/s/kpc
        omega_err   = np.array([err_by_track.get(t, np.nan) for t in TrackId.tolist()])  # km/s/kpc
        R_bar       = np.array([R1bar_by_track.get(t, np.nan) for t in TrackId.tolist()]) * r50_vals  # kpc
        R_bar[~is_barred] = np.nan
        n_not_barred = int(np.sum(~is_barred))
        n_no_omega   = int(np.sum(is_barred & ~has_wp))
        print(f'  this chunk: {int(has_wp.sum())} to analyse, {n_not_barred} not barred, '
              f'{n_no_omega} barred without a valid pattern speed')
        n_retro     = int(np.sum(omega_p < 0))
        if n_retro:
            print(f'  WARNING: {n_retro} bars have Omega_p < 0 (counter to the stellar spin); '
                  f'|Omega_p| is used for R_CR')

        # ---------------- metadata + cell grid ----------------
        mask0       = sw.mask(Swiftfile, spatial_only=True)
        meta        = mask0.metadata
        ScaleFactor = float(meta.scale_factor)

        cell_size_val = float(mask0.cell_size.to('kpc').value[0])
        boxsize_comoving = meta.boxsize.copy()
        boxsize_comoving.convert_to_units('kpc')
        n_cells_per_dim = int(round(float(boxsize_comoving.value[0]) / cell_size_val))

        print(f'  Cell size: {cell_size_val:.2f} kpc ({n_cells_per_dim}^3 cells); '
              f'bar-profile grid {RADII_R50[0]:.1f}-{RADII_R50[-1]:.0f} r50 ({N_RAD} radii), '
              f'aperture {RCUT_KPC:.0f} kpc; {N_WORKERS} workers')

        # per-galaxy grid in kpc (bar-profile edges x r50); M(<r) and Vc are NaN beyond the aperture
        RadiusKpc = RADII_R50[None, :] * r50_vals[:, None]

        # ---------------- assign halos to home cells (barred + valid Omega only) ----------------
        # Each galaxy's aperture is its outermost grid point inside RCUT_KPC,
        # i.e. min(10 r50, 50 kpc), and that is what must fit in one cell.
        cell_groups = defaultdict(list)
        in_cell     = np.zeros(len(lhalo), dtype=bool)

        for i in np.flatnonzero(has_wp):
            radii_i, n_in = halo_grid(r50_vals[i])
            if n_in < 2:
                continue                                  # too few grid points inside the aperture
            edges_i = radii_i[:n_in]
            r_comov = edges_i[-1] / ScaleFactor
            c_comov = centre_vals[i] / ScaleFactor
            idx     = np.floor(c_comov / cell_size_val).astype(int) % n_cells_per_dim
            local   = c_comov - idx * cell_size_val
            if np.all((local - r_comov >= 0.0) & (local + r_comov <= cell_size_val)):
                cell_groups[tuple(idx)].append((i, centre_vals[i], edges_i))
                in_cell[i] = True

        n_included = int(in_cell.sum())
        n_skipped  = int(has_wp.sum()) - n_included
        print(f'  {n_included}/{int(has_wp.sum())} halos to analyse fully contained in their home cell '
              f'({n_skipped} skipped), across {len(cell_groups)} occupied cells')

        tasks = []
        for idx, halo_list in cell_groups.items():
            cell_lo = np.array(idx) * cell_size_val
            tasks.append((Swiftfile, cell_lo, cell_lo + cell_size_val, halo_list))

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
                    n_in = len(ms)                       # grid points inside the aperture
                    Menc_star[local_i, :n_in] = ms
                    Menc_gas[local_i,  :n_in] = mg
                    Menc_dm[local_i,   :n_in] = md
                ndone_halos += len(results)
                max_n_loaded = max(max_n_loaded, n_loaded)
                ndone_cells += 1
                if ndone_cells % 20 == 0 or ndone_cells == len(tasks):
                    print(f'  ... {ndone_cells}/{len(tasks)} cells, {ndone_halos}/{n_included} halos '
                          f'(max particles in one cell read: {max_n_loaded})', flush=True)

        # ---------------- solve for the co-rotation radius ----------------
        Menc_tot = Menc_star + Menc_gas + Menc_dm
        Vc       = circular_velocity(RadiusKpc, Menc_tot)

        R_CR      = np.full(len(lhalo), np.nan)
        status    = np.full(len(lhalo), 4, dtype=int)
        n_cross   = np.zeros(len(lhalo), dtype=int)
        status[~is_barred]           = STATUS_NOT_BARRED
        status[is_barred & ~has_wp]  = STATUS_NO_OMEGA
        status[has_wp & ~in_cell]    = 3

        # Range from the statistical error on Omega_p: a faster bar gives a
        # smaller R_CR, so |Omega_p| + err -> lower bound, |Omega_p| - err -> upper.
        R_CR_lo   = np.full(len(lhalo), np.nan)
        R_CR_hi   = np.full(len(lhalo), np.nan)

        # Circular speed / angular frequency at the bar end, and the pattern-speed
        # quality parameter q = sigma_Omega / Omega_c(R_bar). R1Bar_value is a
        # bar-profile bin edge, so R_bar lies exactly on this grid.
        Vc_bar    = np.full(len(lhalo), np.nan)

        for i in np.flatnonzero(in_cell):
            wp = abs(omega_p[i])
            R_CR[i], status[i], n_cross[i] = find_corotation(RadiusKpc[i], Vc[i], wp)
            R_CR_lo[i] = find_corotation(RadiusKpc[i], Vc[i], wp + omega_err[i])[0]
            R_CR_hi[i] = find_corotation(RadiusKpc[i], Vc[i], wp - omega_err[i])[0]
            Vc_bar[i]  = value_at_radius(RadiusKpc[i], Vc[i], R_bar[i])

        Vc_CR    = abs(omega_p) * R_CR        # = Vc at R_CR by construction, km/s
        Rcal     = R_CR / R_bar               # rate parameter (fast bar: 1 <= Rcal <= 1.4)
        Omc_bar  = Vc_bar / R_bar             # Omega_c(R_bar), km/s/kpc
        q_Omega  = omega_err / Omc_bar        # pattern-speed quality (see paper text)

        print('\n  Status counts: ' + ', '.join(
            f'{s}: {int(np.sum(status == s))}' for s in range(7)))
        ok = status == 0
        if np.any(ok):
            print(f'  R_CR (kpc) for status 0: min={np.nanmin(R_CR[ok]):.2f}  '
                  f'median={np.nanmedian(R_CR[ok]):.2f}  max={np.nanmax(R_CR[ok]):.2f}')
            print(f'  R_CR/R_bar for status 0: median={np.nanmedian(Rcal[ok]):.2f}  '
                  f'(fraction < 1: {np.nanmean(Rcal[ok] < 1):.2f})')

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
            grp0.create_dataset('NHalosNotBarred', data=n_not_barred,       dtype='int')
            grp0.create_dataset('NHalosNoOmega',   data=n_no_omega,         dtype='int')
            grp0.create_dataset('RMin_r50',        data=RADII_R50[0],       dtype='float')
            grp0.create_dataset('RMax_r50',        data=RADII_R50[-1],      dtype='float')
            grp0.create_dataset('RCutKpc',         data=RCUT_KPC,           dtype='float')
            grp0.attrs['OmegaFile']  = omega_fn
            grp0.attrs['RegionFile'] = region_fn

            grp1.create_dataset('TrackId',         data=TrackId,            dtype='int')
            grp1.create_dataset('r50_stars',       data=r50_vals,           dtype='float')
            grp1.create_dataset('PatternSpeed_kms_per_kpc',    data=omega_p,   dtype='float')
            grp1.create_dataset('PatternSpeedErr_kms_per_kpc', data=omega_err, dtype='float')
            grp1.create_dataset('R_bar_kpc',       data=R_bar,              dtype='float')
            grp1.create_dataset('R_CR_kpc',        data=R_CR,               dtype='float')
            grp1.create_dataset('R_CR_lo_kpc',     data=R_CR_lo,            dtype='float')
            grp1.create_dataset('R_CR_hi_kpc',     data=R_CR_hi,            dtype='float')
            grp1.create_dataset('R_CR_over_R_bar', data=Rcal,               dtype='float')
            grp1.create_dataset('R_CR_over_r50',   data=R_CR / r50_vals,    dtype='float')
            grp1.create_dataset('Vc_at_CR_kms',    data=Vc_CR,              dtype='float')
            grp1.create_dataset('Vc_at_Rbar_kms',  data=Vc_bar,             dtype='float')
            grp1.create_dataset('Omega_c_at_Rbar_kms_per_kpc', data=Omc_bar, dtype='float')
            grp1.create_dataset('q_Omega',         data=q_Omega,            dtype='float')
            grp1.create_dataset('Status',          data=status,             dtype='int')
            grp1.create_dataset('NCrossings',      data=n_cross,            dtype='int')

            grp2.create_dataset('Radius_r50',      data=RADII_R50,          dtype='float')  # same edges as the bar profiles
            grp2.create_dataset('RadiusKpc',       data=RadiusKpc,          dtype='float')  # per galaxy: Radius_r50 * r50
            grp2.create_dataset('Menc_stars',      data=Menc_star,          dtype='float')
            grp2.create_dataset('Menc_gas',        data=Menc_gas,           dtype='float')
            grp2.create_dataset('Menc_dm',         data=Menc_dm,            dtype='float')
            grp2.create_dataset('Vc_kms',          data=Vc,                 dtype='float')

    sys.exit()
