import os
import sys
import argparse
import resource
import warnings
from   collections                  import defaultdict
from   concurrent.futures           import ProcessPoolExecutor, as_completed
import numpy                        as     np
import h5py                         as     h5
from   scipy.spatial                import cKDTree
from   colibre_utility              import rotation_matrix_from_vectors
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# --------------------------------------------------------------------------
# Face-on and edge-on maps of stars, gas, HI, H2 and star formation for every disc, in units of
# r50_stars and aligned with the bar, for stacking barred against unbarred discs
# (Frosst_2026_postanalysis/barunbar_residual_maps.py).
#
# For each galaxy (discs from the Bar_Prof table: D/T > DT_MIN, M* > MSTAR_MIN):
#   * particles are rotated face-on with the stellar angular momentum within 50 kpc (the same
#     frame as Calculate_Bar_Prof.py / Calculate_Bar_Omega.py) and scaled by r50;
#   * the frame is then turned about z so that the stellar m = 2 phase lies along +x:
#       psi = 0.5 atan2(sum m sin 2phi, sum m cos 2phi)
#     over the stars in the bar region R0bar < R < R1bar of Bar_Omega (if the galaxy has one), or in
#     PHASE_FALLBACK (in r50) otherwise. For barred discs the bar lies along x;
#   * FACE-ON map:  x-y plane, |x|,|y| < HALF, particles with |z| < ZHALF;
#     SIDE-ON map:  x-z plane (bar seen lengthwise), |x| < HALF, |z| < ZHALF, all |y| < HALF;
#     END-ON  map:  y-z plane (bar seen end-on),     |y| < HALF, |z| < ZHALF, all |x| < HALF.
#   Pixels of PIX r50. Each map holds the summed stellar, gas, HI, H2 mass [Msun] and SFR [Msun/yr]
#   per pixel (HI = m X_H f_HI, H2 = 2 m X_H f_H2, SFR clipped at 0, as in Calculate_Gas_Prof.py).
#   Divide by (PIX r50)^2 [kpc^2] for surface densities.
#
# Mass ranges: --logm-min / --logm-max map only the discs in that range of log M* (written to
# Bar_Maps_<snap>_logM<min>-<max>.hdf5), so the work can be split or restricted; the stacking script
# reads and combines every Bar_Maps_<snap>*.hdf5 file it finds (mass ranges and chunks alike).
#
# Particle reads:
#   * default (COSMA): swiftsimio masked reads, CELL-BATCHED exactly as Calculate_Gas_Prof.py
#     (one read per occupied top-level cell, own reads for galaxies crossing a cell boundary,
#     galaxies crossing the box edge skipped); --chunk-id / --n-chunks split the work by home cell.
#   * --local: plain h5py read of the whole snapshot with periodic wrapping (small test boxes,
#     e.g. L012_m6; no swiftsimio needed).
# --------------------------------------------------------------------------

# ---- Simulation information ----
# --- COSMA paths ---
BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
OutBase      = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"
BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
# --- Local test paths (used with --local) ---
LocalBase    = "/Users/matthewfrosst/Documents/COLIBRE/"
LocalOutBase = "/Users/matthewfrosst/Documents/COLIBRE/Frosst_2026_Outputs/"

fname        = "Bar_Maps_"

# ---- analysis information ----
Nstar_min    = 5e3      # same selection as the bar catalogues
Nstar_max    = 1e10
DT_MIN       = 0.6      # discs only (from the Bar_Prof table)
MSTAR_MIN    = 1e10     # [Msun]
HALF         = 3.0      # [r50] half-width of the maps in the disc plane
ZHALF        = 1.0      # [r50] half-height of the edge-on maps; face-on maps use |z| < ZHALF
PIX          = 0.1      # [r50] pixel size
PHASE_FALLBACK = (0.5, 1.5)   # [r50] annulus for the m = 2 phase of galaxies without a bar region
QUANTITIES   = ['StellarMass', 'GasMass', 'HIMass', 'H2Mass', 'SFR']

NPIX_XY      = int(round(2*HALF/PIX))
NPIX_Z       = int(round(2*ZHALF/PIX))
EDGES_XY     = np.linspace(-HALF, HALF, NPIX_XY + 1)
EDGES_Z      = np.linspace(-ZHALF, ZHALF, NPIX_Z + 1)
R_SEARCH     = np.sqrt(2*HALF**2 + ZHALF**2)      # [r50] sphere enclosing every map

N_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

MSUN_G, KPC_CM, YR_S = 1.98841e33, 3.08567758e21, 3.15576e7

def peak_mem_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == 'darwin' else peak / 1e6

# --------------------------------------------------------------------------
# ---- maps ----
def galaxy_maps(off_s, w_s, off_g, w_g, r50, trans, R0bar, R1bar):
    """
    off_*: particle offsets from the centre [physical kpc] (stars, gas); w_s: (N,) stellar mass,
    w_g: (N, 4) gas, HI, H2 mass and SFR. Returns face (nq, NXY, NXY), side (nq, NXY, NZ),
    end (nq, NXY, NZ) and the m = 2 phase psi [rad] used for the alignment.
    """
    xs = (trans @ off_s.T).T / r50
    xg = (trans @ off_g.T).T / r50
    # m = 2 phase of the stars in the bar region (or the fallback annulus), in the face-on slab
    Rs = np.hypot(xs[:, 0], xs[:, 1])
    lo, hi = (R0bar, R1bar) if R1bar > 0.1 else PHASE_FALLBACK
    sel = (Rs >= lo) & (Rs < hi) & (np.abs(xs[:, 2]) < ZHALF)
    phi = np.arctan2(xs[sel, 1], xs[sel, 0])
    psi = 0.5*np.arctan2(np.sum(w_s[sel]*np.sin(2*phi)), np.sum(w_s[sel]*np.cos(2*phi))) if sel.any() else 0.0
    c, s = np.cos(-psi), np.sin(-psi)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xs, xg = xs @ rot.T, xg @ rot.T

    W = [(xs, w_s[:, None])] + [(xg, w_g)]
    face = np.zeros((len(QUANTITIES), NPIX_XY, NPIX_XY), np.float32)
    side = np.zeros((len(QUANTITIES), NPIX_XY, NPIX_Z), np.float32)
    end  = np.zeros((len(QUANTITIES), NPIX_XY, NPIX_Z), np.float32)
    q0 = 0
    for x, w in W:
        inxy = (np.abs(x[:, 0]) < HALF) & (np.abs(x[:, 1]) < HALF)
        f = inxy & (np.abs(x[:, 2]) < ZHALF)
        for k in range(w.shape[1]):
            face[q0+k] = np.histogram2d(x[f, 0], x[f, 1], [EDGES_XY, EDGES_XY], weights=w[f, k])[0]
            side[q0+k] = np.histogram2d(x[f, 0], x[f, 2], [EDGES_XY, EDGES_Z], weights=w[f, k])[0]
            end[q0+k]  = np.histogram2d(x[f, 1], x[f, 2], [EDGES_XY, EDGES_Z], weights=w[f, k])[0]
        q0 += w.shape[1]
    return face, side, end, psi

def maps_from_particles(pos_s, w_s, pos_g, w_g, halo_list, boxsize=None):
    # halo_list entries: (local_i, centre_phys[3], r50, trans, R0bar, R1bar)
    ts = cKDTree(pos_s, boxsize=boxsize) if len(pos_s) else None
    tg = cKDTree(pos_g, boxsize=boxsize) if len(pos_g) else None
    out = []
    for (i, c, r50, trans, R0, R1) in halo_list:
        def grab(tree, pos):
            if tree is None: return np.zeros((0, 3)), np.array([], int)
            idx = np.asarray(tree.query_ball_point(c, R_SEARCH*r50), int)
            d = pos[idx] - c
            if boxsize is not None:
                d -= boxsize*np.round(d/boxsize)
            return d, idx
        ds, js = grab(ts, pos_s); dg, jg = grab(tg, pos_g)
        out.append((i, *galaxy_maps(ds, w_s[js], dg, w_g[jg], r50, trans, R0, R1)))
    return out

# --------------------------------------------------------------------------
# ---- particle reads ----
def _gas_weights(m, X_H, f_HI, f_H2, sfr):
    return np.column_stack([m, m*X_H*f_HI, 2.0*m*X_H*f_H2, np.clip(sfr, 0, None)])

def load_region_swift(Swiftfile, lo_comov, hi_comov):
    import unyt
    import swiftsimio as sw
    region = [[lo_comov[k], hi_comov[k]] * unyt.kpc for k in range(3)]
    try:
        mask = sw.mask(Swiftfile, spatial_only=True, safe_padding=False)
    except TypeError:
        mask = sw.mask(Swiftfile, spatial_only=True)
    mask.constrain_spatial(region)
    data = sw.load(Swiftfile, mask=mask)
    st, gas = data.stars, data.gas
    def phys(q, u):
        q.convert_to_units(u); q.convert_to_physical(); return q.value
    pos_s, m_s = phys(st.coordinates, 'kpc'), phys(st.masses, 'Msun')
    pos_g, m_g = phys(gas.coordinates, 'kpc'), phys(gas.masses, 'Msun')
    w_g = _gas_weights(m_g, np.asarray(gas.element_mass_fractions.hydrogen),
                       np.asarray(gas.species_fractions.HI), np.asarray(gas.species_fractions.H2),
                       gas.star_formation_rates.to_physical().to_value(unyt.Msun/unyt.yr))
    return pos_s, m_s, pos_g, w_g

def measure_region(task):
    (Swiftfile, lo, hi, halo_list) = task
    pos_s, m_s, pos_g, w_g = load_region_swift(Swiftfile, lo, hi)
    return maps_from_particles(pos_s, m_s, pos_g, w_g, halo_list), peak_mem_gb()

def load_all_h5py(Swiftfile):
    # whole snapshot, physical units, via the SWIFT unit-conversion attributes (small boxes only)
    f = h5.File(Swiftfile, 'r')
    a = float(f['Header'].attrs['Scale-factor'][0])
    box = float(f['Header'].attrs['BoxSize'][0]) * a * 1e3                  # physical kpc (internal Mpc)
    def read(name, unit_cgs, cols=None):
        d = f[name]
        fac = float(d.attrs['Conversion factor to physical CGS (including cosmological corrections)'][0])
        v = d[:] if cols is None else d[:, cols]
        return v * (fac / unit_cgs)
    names_sp = [n.decode() for n in f['SubgridScheme/NamedColumns/SpeciesFractions'][:]]
    names_el = [n.decode() for n in f['SubgridScheme/NamedColumns/ElementMassFractions'][:]]
    pos_s = read('PartType4/Coordinates', KPC_CM); m_s = read('PartType4/Masses', MSUN_G)
    pos_g = read('PartType0/Coordinates', KPC_CM); m_g = read('PartType0/Masses', MSUN_G)
    X_H  = f['PartType0/ElementMassFractions'][:, names_el.index('Hydrogen')]
    sp   = f['PartType0/SpeciesFractions']
    f_HI, f_H2 = sp[:, names_sp.index('HI')], sp[:, names_sp.index('H2')]
    sfr  = read('PartType0/StarFormationRates', MSUN_G / YR_S)
    print(f'  read {len(m_s)} star and {len(m_g)} gas particles (box {box:.0f} kpc)')
    return pos_s, m_s, pos_g, _gas_weights(m_g, X_H, f_HI, f_H2, sfr), box, 1.0/a - 1.0

# --------------------------------------------------------------------------
def read_soap(soap_file):
    # positions [physical kpc] etc. of the Nstar-selected galaxies, via h5py and the unit attributes
    s = h5.File(soap_file, 'r')
    def read(name, unit_cgs, idx):
        d = s[name]
        fac = float(d.attrs['Conversion factor to physical CGS (including cosmological corrections)'][0])
        return d[:][idx] * (fac / unit_cgs)
    N = s['BoundSubhalo/NumberOfStarParticles'][:]
    lhalo = np.where((N >= Nstar_min) & (N <= Nstar_max))[0]
    return dict(lhalo=lhalo,
                TrackId=s['InputHalos/HBTplus/TrackId'][:][lhalo],
                is_central=s['InputHalos/IsCentral'][:][lhalo],
                centre=read('InputHalos/HaloCentre', KPC_CM, lhalo),
                r50=read('BoundSubhalo/HalfMassRadiusStars', KPC_CM, lhalo),
                mstar=read('ExclusiveSphere/50kpc/StellarMass', MSUN_G, lhalo),
                angJ=s['InclusiveSphere/50kpc/AngularMomentumStars'][:][lhalo])   # direction only

def read_tables(table_dir, ext3, TrackId):
    # disc flag and bar region (R0, R1 in r50) from the bar catalogues, matched by TrackId
    p = h5.File(table_dir+'Stars_Mproj_Bar_Prof_'+ext3+'.hdf5', 'r')['HaloData']
    o = h5.File(table_dir+'Stars_Mproj_Bar_Omega_'+ext3+'.hdf5', 'r')['HaloData']
    tid = np.array(p['TrackId']); assert np.array_equal(tid, np.array(o['TrackId']))
    pos = {t: k for k, t in enumerate(tid)}
    k = np.array([pos.get(t, -1) for t in TrackId])
    ok = k >= 0
    DT, M = np.full(len(k), np.nan), np.full(len(k), np.nan)
    R0, R1 = np.zeros(len(k)), np.zeros(len(k))
    DT[ok], M[ok] = np.array(p['DT'])[k[ok]], np.array(p['StellarMass'])[k[ok]]
    R0[ok], R1[ok] = np.array(o['R0_stars_rhalf'])[k[ok]], np.array(o['R1_stars_rhalf'])[k[ok]]
    return (DT > DT_MIN) & (M > MSTAR_MIN), R0, R1, np.log10(M)

# --------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, default=snap)
    parser.add_argument("--box", type=str, default=None, help="box directory, e.g. L012_m6")
    parser.add_argument("--run", type=str, default=None, help="run directory, e.g. THERMAL_AGN_m6")
    parser.add_argument("--local", action="store_true", help="local test: h5py read of the whole snapshot")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--logm-min", type=float, default=None,
                        help="only map discs with log10 M* >= this (Bar_Prof StellarMass); output file gets a suffix")
    parser.add_argument("--logm-max", type=float, default=None, help="only map discs with log10 M* < this")
    args = parser.parse_args()
    snap = args.snap
    if args.box: BoxDir = [args.box.rstrip('/') + '/']
    if args.run: RunDir = args.run.rstrip('/') + '/'
    if args.local: BasePath, OutBase = LocalBase, LocalOutBase

    for Dir in BoxDir:
        ext4, ext3 = str(snap).zfill(4), str(snap).zfill(3)
        Swiftfile = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"
        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"
        table_dir = OutBase+Dir+RunDir

        S = read_soap(soap_file)
        disc, R0bar, R1bar, logM = read_tables(table_dir, ext3, S['TrackId'])
        mrange = ''
        if args.logm_min is not None or args.logm_max is not None:
            lo_m = -np.inf if args.logm_min is None else args.logm_min
            hi_m = np.inf if args.logm_max is None else args.logm_max
            disc &= (logM >= lo_m) & (logM < hi_m)
            mrange = '_logM' + ('min' if np.isinf(lo_m) else f'{lo_m:.2f}') + '-' + ('max' if np.isinf(hi_m) else f'{hi_m:.2f}')
            print(f'  stellar-mass range {lo_m} <= log M* < {hi_m}')
        sel = np.where(disc)[0]
        print(f'- snapshot {snap}: {len(S["lhalo"])} galaxies with Nstar >= {Nstar_min:.0f}, {len(sel)} discs to map')

        def entry(i):
            return (i, S['centre'][i], S['r50'][i], rotation_matrix_from_vectors(S['angJ'][i], [0, 0, 1]),
                    R0bar[i], R1bar[i])

        nq = len(QUANTITIES)
        face = np.zeros((len(sel), nq, NPIX_XY, NPIX_XY), np.float32)
        side = np.zeros((len(sel), nq, NPIX_XY, NPIX_Z), np.float32)
        end  = np.zeros((len(sel), nq, NPIX_XY, NPIX_Z), np.float32)
        psi  = np.full(len(sel), np.nan)
        done = np.zeros(len(sel), bool)
        row  = {g: k for k, g in enumerate(sel)}

        def store(results):
            for (i, fa, si, en, ps) in results:
                k = row[i]; face[k], side[k], end[k], psi[k] = fa, si, en, ps; done[k] = True

        if args.local:
            pos_s, m_s, pos_g, w_g, box, redshift = load_all_h5py(Swiftfile)
            store(maps_from_particles(pos_s, m_s, pos_g, w_g, [entry(i) for i in sel], boxsize=box))
            mine, skipped = sel, np.array([], int)
        else:
            import swiftsimio as sw
            mask0 = sw.mask(Swiftfile, spatial_only=True)
            meta = mask0.metadata
            a = float(meta.scale_factor); redshift = 1.0/a - 1.0
            cell = float(mask0.cell_size.to('kpc').value[0])
            boxc = meta.boxsize.copy(); boxc.convert_to_units('kpc'); boxc = float(boxc.value[0])
            ncell = int(round(boxc/cell))
            c_com = S['centre'][sel]/a; r_com = R_SEARCH*S['r50'][sel]/a
            idx = np.floor(c_com/cell).astype(int) % ncell
            loc = c_com - idx*cell
            inside = np.all((loc - r_com[:, None] >= 0) & (loc + r_com[:, None] <= cell), axis=1)
            edge = np.any((c_com - r_com[:, None] < 0) | (c_com + r_com[:, None] > boxc), axis=1)
            flat = (idx[:, 0]*ncell + idx[:, 1])*ncell + idx[:, 2]
            in_chunk = (flat % args.n_chunks) == args.chunk_id
            groups = defaultdict(list); tasks = []
            for k in np.where(in_chunk & inside)[0]:
                groups[tuple(idx[k])].append(entry(sel[k]))
            for key, hl in groups.items():
                lo = np.array(key)*cell; tasks.append((Swiftfile, lo, lo + cell, hl))
            for k in np.where(in_chunk & ~inside & ~edge)[0]:
                tasks.append((Swiftfile, c_com[k] - r_com[k], c_com[k] + r_com[k], [entry(sel[k])]))
            tasks.sort(key=lambda t: -len(t[3]))
            skipped = sel[in_chunk & ~inside & edge]
            mine = sel[in_chunk]
            print(f'  chunk {args.chunk_id}/{args.n_chunks}: {len(mine)} discs, {len(tasks)} reads, '
                  f'{len(skipped)} skipped at the box edge; {N_WORKERS} workers')
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                futs = [pool.submit(measure_region, t) for t in tasks]
                for it, fut in enumerate(as_completed(futs)):
                    res, mem = fut.result(); store(res)
                    if (it + 1) % 200 == 0 or it + 1 == len(tasks):
                        print(f'  ... {it+1}/{len(tasks)} reads, max worker memory {mem:.1f} GB', flush=True)

        keep = np.isin(sel, mine)
        suffix = f"_chunk{args.chunk_id:04d}" if args.n_chunks > 1 else ""
        fn = OutBase+Dir+RunDir+fname+ext3+mrange+suffix+".hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with h5.File(fn, "w") as out:
            g0 = out.create_group("Header")
            g0.create_dataset('LogMstarRange', data=[-np.inf if args.logm_min is None else args.logm_min,
                                                     np.inf if args.logm_max is None else args.logm_max])
            for k, v in [('Redshift', redshift), ('HALF_rhalf', HALF), ('ZHALF_rhalf', ZHALF), ('PIX_rhalf', PIX),
                         ('ChunkId', args.chunk_id), ('NChunks', args.n_chunks)]:
                g0.create_dataset(k, data=v)
            g0.create_dataset('Quantities', data=np.array(QUANTITIES, dtype='S'))
            g0.attrs['description'] = (
                'Per-disc maps in units of r50_stars, face-on frame of the 50 kpc stellar angular momentum, '
                'turned so that the stellar m=2 phase (bar region, or 0.5-1.5 r50) lies along +x. '
                'Face: x-y, |z|<ZHALF. Side: x-z. End: y-z. Pixel sums of stellar, gas, HI, H2 mass [Msun] '
                'and SFR [Msun/yr]; divide by (PIX*r50)^2 [kpc^2] for surface densities.')
            g1 = out.create_group("HaloData")
            g1.create_dataset('TrackId',    data=S['TrackId'][sel][keep])
            g1.create_dataset('is_central', data=S['is_central'][sel][keep])
            g1.create_dataset('StellarMass', data=S['mstar'][sel][keep])
            g1.create_dataset('r50_stars',  data=S['r50'][sel][keep])
            g1.create_dataset('Phi2_align_radians', data=psi[keep])
            g1.create_dataset('Mapped',     data=done[keep])
            g2 = out.create_group("Maps")
            g2.create_dataset('edges_xy_rhalf', data=EDGES_XY)
            g2.create_dataset('edges_z_rhalf',  data=EDGES_Z)
            for name, arr in [('FaceOn', face), ('SideOn', side), ('EndOn', end)]:
                g2.create_dataset(name, data=arr[keep], compression='gzip', compression_opts=4,
                                  chunks=(1,) + arr.shape[1:])
        print(f' Done: {done[keep].sum()} discs mapped. Peak memory {peak_mem_gb():.1f} GB')
