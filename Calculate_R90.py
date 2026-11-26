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
# Stellar sizes enclosing 50 and 90 per cent of the stellar mass and of the g-, r- and i-band
# luminosity, for every galaxy (for comparison with observed disc sizes, e.g. R90).
#
# For each galaxy, all star particles within a 3D sphere of R_APERTURE (default 50 kpc, physical;
# set with --aperture) of the
# SOAP halo centre are used (all particles, not only bound ones -- as Calculate_Bar_Prof.py with
# DoBound = False). For each weight (mass, L_g, L_r, L_i) we store:
#   R50_*, R90_* : projected FACE-ON radii [kpc], i.e. the cylindrical R enclosing 50 / 90 per
#                  cent of the weight in the aperture. The face-on frame is the stellar angular
#                  momentum within 50 kpc, as Calculate_Bar_Prof.py.
#   r50_*, r90_* : 3D spherical radii [kpc] enclosing 50 / 90 per cent of the weight.
# Fractions are of the total within the aperture. The face-on frame is always that of SOAP's 50 kpc
# sphere, whatever the aperture, so changing --aperture changes only which particles are counted.
# Runs with a non-default aperture are written with an '_ap<N>kpc' tag (e.g. Stellar_Sizes_127_ap30kpc.hdf5),
# so they never overwrite the default 50 kpc output.
# --bound keeps only the star particles that are bound to the galaxy's own HBT subhalo (SOAP
# membership files, GroupNr_bound == InputHalos/HaloCatalogueIndex), still within the aperture.
# This removes stars of neighbouring galaxies, which otherwise contaminate R90 (mostly satellites
# and at z >= 1). Its outputs are tagged '_bound' (e.g. Stellar_Sizes_127_bound.hdf5).
# Luminosities are the rest-frame, dust-free
# GAMA-band luminosities of the snapshot (BC03; Trayford et al. 2015), so no dust attenuation and
# no surface-brightness limit are applied.
# Sanity check: r50_Mass should be close to SOAP's bound_subhalo half_mass_radius_stars (stored
# here as r50_stars_SOAP); differences come from the aperture and from non-bound particles.
#
# CELL-BATCHED, as Calculate_Gas_Prof.py:
#   * each galaxy is assigned to its home top-level SWIFT cell;
#   * pass 1: every occupied cell is read once (spatial mask, no padding) and all galaxies whose
#     aperture fits inside that cell are measured from that one read;
#   * pass 2: galaxies whose aperture crosses a cell boundary get their own small masked read;
#   * galaxies whose aperture crosses the box edge (~0.01%) are skipped, flagged and counted;
#   * cells run in parallel worker processes (SLURM_CPUS_PER_TASK);
#   * --chunk-id / --n-chunks split the galaxies by home cell. Combine the chunks with
#     merge_R90_chunks.py.
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

fname        = "Stellar_Sizes_"
# SOAP membership files (one per snapshot sub-file, same particle order); only used with --bound.
# CHECK this path on COSMA: SOAP's Parameters/membership_filename gives where they were written.
MemberFile   = "SOAP-HBT/membership_{ext4}/membership_{ext4}.{ifile}.hdf5"

# ---- analysis Information ----
Nstar_min    = 5e3   # Minimum number of stellar particles (same selection as the bar catalogues)
Nstar_max    = 1e10  # Maximum number of stellar particles
R_APERTURE   = 50.0  # default 3D aperture [physical kpc] (as Calculate_Bar_Prof.py and SOAP's 50 kpc spheres)
FRACTIONS    = (0.5, 0.9)
WEIGHTS      = ['Mass', 'g', 'r', 'i']     # stellar mass and GAMA g, r, i luminosities

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
# ---- Size helpers ----
def enclosing_radius(r, w, fracs=FRACTIONS):
    # Radius enclosing each fraction of the total weight w, interpolating the cumulative
    # distribution linearly between particle radii. Returns NaN if there is no weight.
    tot = w.sum()
    if len(r) == 0 or not tot > 0:
        return np.full(len(fracs), np.nan)
    order = np.argsort(r)
    cum   = np.cumsum(w[order]) / tot
    return np.interp(fracs, cum, r[order])

def sizes_from_particles(pos, weights, halo_list, r_ap, grp=None):
    # Measure every galaxy in halo_list from one set of particles (no periodic wrap needed:
    # each read is a region inside the box that contains the galaxies' whole apertures).
    # halo_list entries: (local_i, centre_phys[3], trans[3,3], halo_catalogue_index)
    # grp: GroupNr_bound of each particle (None = use all particles in the aperture).
    # Returns, per galaxy, sizes (n_weights, 2 [2D, 3D], n_fracs) [kpc] and the particle number.
    nw, nf  = weights.shape[1], len(FRACTIONS)
    tree    = cKDTree(pos) if len(pos) > 0 else None
    results = []
    for (local_i, c, trans, hci) in halo_list:
        out = np.full((nw, 2, nf), np.nan)
        idx = np.asarray(tree.query_ball_point(c, r_ap) if tree is not None else [], dtype=np.int64)
        if grp is not None and len(idx) > 0:
            idx = idx[grp[idx] == hci]                  # bound to this galaxy's subhalo only
        if len(idx) > 0:
            x   = (trans @ (pos[idx] - c).T).T
            R2d = np.hypot(x[:, 0], x[:, 1])           # face-on projected radius
            r3d = np.linalg.norm(x, axis=1)
            for q in range(nw):
                out[q, 0] = enclosing_radius(R2d, weights[idx, q])
                out[q, 1] = enclosing_radius(r3d, weights[idx, q])
        results.append((local_i, out, len(idx)))
    return results

def load_star_region(Swiftfile, lo_comov, hi_comov):
    # Masked read of the stars in one region [comoving kpc]; returns physical positions [kpc] and
    # weights (mass [Msun], L_g, L_r, L_i [dimensionless AB maggies, rest frame, dust free]).
    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    try:
        region = [[lo_comov[k], hi_comov[k]] * unyt.kpc for k in range(3)]
        mask.constrain_spatial(region)
    except ValueError:
        # newer swiftsimio (>= 10) needs comoving-aware limits
        a = mask.metadata.scale_factor
        region = [sw.cosmo_array([lo_comov[k], hi_comov[k]], unyt.kpc, comoving=True,
                                 scale_factor=a, scale_exponent=1) for k in range(3)]
        mask.constrain_spatial(region)
    data  = sw.load(Swiftfile, mask=mask)
    stars = data.stars

    pos = stars.coordinates ; pos.convert_to_units('kpc') ; pos.convert_to_physical()
    m   = stars.masses      ; m.convert_to_units('Msun')  ; m.convert_to_physical()
    lum = stars.luminosities
    weights = np.column_stack([m.value,
                               np.asarray(lum.GAMA_g), np.asarray(lum.GAMA_r), np.asarray(lum.GAMA_i)])
    return pos.value, weights, np.asarray(mask.stars)

def read_membership(member_files, file_offsets, ranges):
    # GroupNr_bound of the masked stars. ranges are [start, end) indices into the snapshot's
    # concatenated (virtual-file) star order, which is sub-file 0, 1, ... in turn; the membership
    # files have the same per-file order. Returned in the order the masked read returns the stars.
    out = []
    for start, end in ranges:
        while start < end:
            f  = np.searchsorted(file_offsets, start, side='right') - 1
            e  = min(end, file_offsets[f + 1])
            with h5.File(member_files[f], 'r') as M:
                out.append(M['PartType4/GroupNr_bound'][start - file_offsets[f]:e - file_offsets[f]])
            start = e
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

def measure_region(task):
    # Worker: read one region and measure every galaxy assigned to it.
    (Swiftfile, lo_comov, hi_comov, halo_list, r_ap, member) = task
    pos, weights, ranges = load_star_region(Swiftfile, lo_comov, hi_comov)
    grp = None
    if member is not None:
        grp = read_membership(member[0], member[1], ranges)
        if len(grp) != len(pos):
            raise RuntimeError(f'membership/star count mismatch ({len(grp)} vs {len(pos)})')
    return sizes_from_particles(pos, weights, halo_list, r_ap, grp), len(pos), peak_mem_gb()

def assign_cells(centres_comov, r_search_comov, cell_size, n_cells, boxsize):
    # Home cell of each galaxy, and whether its aperture fits inside that cell (pass 1),
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
    parser.add_argument("--aperture", type=float, default=R_APERTURE,
                        help=f"3D aperture radius [physical kpc] (default: {R_APERTURE:g})")
    parser.add_argument("--bound", action="store_true",
                        help="use only star particles bound to the galaxy's own subhalo (SOAP membership)")
    args = parser.parse_args()
    snap = args.snap
    r_ap = args.aperture
    ap_tag = ("" if r_ap == R_APERTURE else f"_ap{r_ap:g}kpc") + ("_bound" if args.bound else "")
    print(f'Aperture: {r_ap:g} physical kpc; ' + ('bound particles only' if args.bound else 'all particles'))

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
        # exclusive sphere: bound particles only (the inclusive sphere tilts the frame of satellites)
        angJ_stars  = read_selected(soap.exclusive_sphere_50kpc, 'angular_momentum_stars', lhalo)
        halo_centre = read_selected(soap.input_halos, 'halo_centre', lhalo)
        is_central  = read_selected(soap.input_halos, 'is_central', lhalo)
        TrackId     = read_selected(soap.input_halos_hbtplus, 'track_id', lhalo)
        hcat_index  = np.asarray(read_selected(soap.input_halos, 'halo_catalogue_index', lhalo))

        rhalf_stars.convert_to_units('kpc') ; rhalf_stars.convert_to_physical()
        mstar.convert_to_units('Msun')      ; mstar.convert_to_physical()
        halo_centre.convert_to_units('kpc') ; halo_centre.convert_to_physical()
        print(f'  Peak memory after SOAP reads: {peak_mem_gb():.1f} GB')

        centres = halo_centre.value
        angJ    = angJ_stars.value

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

        # --- Membership files (only with --bound): list and star-count offsets of each sub-file ---
        member = None
        if args.bound:
            with h5.File(Swiftfile, 'r') as F:
                n_files = int(np.atleast_1d(F['Header'].attrs['NumFilesPerSnapshot'])[0])
            if n_files == 1:       # virtual file: count the sub-files next to it
                n_files = 0
                while os.path.exists(Swiftfile.replace('.hdf5', f'.{n_files}.hdf5')):
                    n_files += 1
                n_files = max(n_files, 1)
            member_files = [BasePath+Dir+RunDir+MemberFile.format(ext4=ext4, ifile=i) for i in range(n_files)]
            counts = []
            for mf in member_files:
                if not os.path.exists(mf):
                    sys.exit(f'Membership file not found: {mf}. Set MemberFile at the top of this script.')
                with h5.File(mf, 'r') as M:
                    counts.append(M['PartType4/GroupNr_bound'].shape[0])
            file_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            if file_offsets[-1] != int(meta.n_stars):
                sys.exit(f'Membership files hold {file_offsets[-1]} stars, snapshot has {int(meta.n_stars)}. Aborting.')
            member = (member_files, file_offsets)
            print(f'  Membership: {n_files} files, {file_offsets[-1]} stars')

        c_comov = centres / ScaleFactor
        r_comov = np.full(n_gal, r_ap / ScaleFactor)
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
            return (i, centres[i], rotation_matrix_from_vectors(angJ[i], [0, 0, 1]), hcat_index[i])

        # --- Tasks: one per occupied cell (pass 1), one per boundary-crossing galaxy (pass 2) ---
        cell_groups = defaultdict(list)
        for i in pass1:
            cell_groups[tuple(cell_idx[i])].append(halo_entry(i))
        tasks = []
        for idx, halo_list in cell_groups.items():
            lo = np.array(idx) * cell_size
            tasks.append((Swiftfile, lo, lo + cell_size, halo_list, r_ap, member))
        for i in pass2:
            tasks.append((Swiftfile, c_comov[i] - r_comov[i], c_comov[i] + r_comov[i], [halo_entry(i)], r_ap, member))
        tasks.sort(key=lambda t: -len(t[3]))   # biggest cells first, for load balancing
        print(f'  {len(cell_groups)} cell reads + {len(pass2)} single-galaxy reads')

        # --- output arrays for this chunk (NaN = not measured) ---
        sizes  = np.full((len(mine), len(WEIGHTS), 2, len(FRACTIONS)), np.nan)
        npart  = np.full(len(mine), -1, dtype=np.int64)
        row_of = {g: k for k, g in enumerate(mine)}   # global index -> row in this chunk's arrays

        max_loaded, max_rss, n_done = 0, 0.0, 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(measure_region, t) for t in tasks]
            for it, fut in enumerate(as_completed(futures)):
                results, n_loaded, worker_rss = fut.result()
                for (i, out, n) in results:
                    sizes[row_of[i]] = out
                    npart[row_of[i]] = n
                n_done += len(results)
                max_loaded = max(max_loaded, n_loaded); max_rss = max(max_rss, worker_rss)
                if (it + 1) % 200 == 0 or it + 1 == len(tasks):
                    print(f'  ... {it+1}/{len(tasks)} reads done, {n_done}/{len(pass1)+len(pass2)} galaxies '
                          f'(max star particles in one read: {max_loaded}, max worker memory: {max_rss:.1f} GB)', flush=True)

        # --- Write to hdf5 ---
        suffix = f"_chunk{args.chunk_id:04d}" if args.n_chunks > 1 else ""
        fn = OutBase+Dir+RunDir+fname+ext3+ap_tag+suffix+".hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with h5.File(fn, "w") as output:
            grp0 = output.create_group("Header")
            grp0.create_dataset('Redshift',        data=redshift)
            grp0.create_dataset('ApertureKpc',     data=r_ap)
            grp0.create_dataset('BoundOnly',       data=int(args.bound))
            grp0.create_dataset('ChunkId',         data=args.chunk_id)
            grp0.create_dataset('NChunks',         data=args.n_chunks)
            grp0.create_dataset('NGalTotal',       data=n_gal)
            grp0.create_dataset('CellSizeKpc',     data=cell_size)
            grp0.create_dataset('NHalosSkipped',   data=len(skipped))
            grp0.create_dataset('NHalosIncluded',  data=len(pass1) + len(pass2))
            grp0.attrs['description'] = ('Radii [physical kpc] enclosing 50 and 90 per cent of the stellar mass and '
                                         'the rest-frame, dust-free GAMA g, r, i luminosity of all star particles '
                                         'within a 3D aperture of ApertureKpc (if BoundOnly: only those bound to the galaxy\'s HBT subhalo; face-on frame from SOAP\'s 50 kpc sphere). R*: face-on projected (cylindrical); '
                                         'r*: 3D spherical. GlobalIndex is the position in the full Nstar-selected '
                                         'galaxy list.')

            grp1 = output.create_group("HaloData")
            grp1.create_dataset('GlobalIndex',    data=mine,                         dtype='int')
            grp1.create_dataset('TrackId',        data=np.asarray(TrackId)[mine],    dtype='int')
            grp1.create_dataset('is_central',     data=np.asarray(is_central)[mine], dtype='int')
            grp1.create_dataset('StellarMass',    data=mstar.value[mine],            dtype='float')
            grp1.create_dataset('r50_stars_SOAP', data=rhalf_stars.value[mine],      dtype='float')
            grp1.create_dataset('NumStarPart',    data=npart,                        dtype='int')
            grp1.create_dataset('Skipped',        data=np.isin(mine, skipped),       dtype='bool')

            grp2 = output.create_group("Sizes")
            for q, w in enumerate(WEIGHTS):
                for f_i, f in enumerate(FRACTIONS):
                    p = int(round(100 * f))
                    grp2.create_dataset(f'R{p}_{w}', data=sizes[:, q, 0, f_i], dtype='float')   # face-on 2D
                    grp2.create_dataset(f'r{p}_{w}', data=sizes[:, q, 1, f_i], dtype='float')   # 3D
        print(f' Done. Peak memory (main process): {peak_mem_gb():.1f} GB')

    sys.exit()
