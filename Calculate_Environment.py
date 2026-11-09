import os
import sys
import argparse
import itertools
import warnings
import numpy                        as np
from   scipy.spatial                import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# ==========================================================================
# LOCAL ENVIRONMENT OF GALAXIES
# --------------------------------------------------------------------------
# Everything here is measured from the SOAP catalogue ALONE: no particle
# data is read, so there is no cell batching and no chunking, and the whole
# box runs as one job (the cost is a few kd-tree queries over the galaxy
# catalogue; minutes even for the largest boxes).
#
# Two galaxy samples are used:
#   TARGETS : Nstar_min <= Nstar <= Nstar_max, i.e. EXACTLY the selection of
#             Calculate_Bar_Prof.py / Calculate_Bar_Rcorot.py, in the same
#             order, so the rows of this file line up with theirs (match on
#             TrackId to be safe).
#   TRACERS : every subhalo (central or satellite) with stellar mass
#             >= Mstar_tracer_min. These define the environment. They go
#             deeper than the targets, so the environment of the faintest
#             target is as well sampled as that of the brightest.
# "Stellar mass" is ExclusiveSphere/50kpc/StellarMass throughout, as in
# Calculate_Gal_Table.py. A target is never counted in its own environment.
#
# All lengths are COMOVING Mpc (identical to physical at z = 0); all the
# normalised quantities are dimensionless and independent of that choice.
#
# What is measured (all in HaloData, one row per target):
#
#  1. HOST HALO  (satellites inherit the values of their central, found via
#     SOAP/HostHaloIndex; centrals use their own)
#       Host_fsub200          SOAP SO/200_crit/MassFractionSatellites: the
#                             fraction of M200 bound to satellites
#       Host_M200, Host_r200_cMpc, DistToHost_over_r200
#       Host_Nsat_R200        number of tracer satellites within R200
#       Host_Msat_star_R200   their summed stellar mass
#       Host_fsat_star_R200   Msat_star / (Msat_star + Mstar of the central)
#     NaN / -1 where there is no host (hostless satellites) or no R200.
#
#  2. FIXED-APERTURE DENSITIES, spheres of ApertureRadii (1, 5, 10 cMpc)
#       N_in_R, Mstar_in_R, Mbound_in_R        raw sums over the tracers
#       Delta_N, Delta_Mstar, Delta_Mbound     the density in the sphere over
#                             the box mean of the SAME tracers (rho / rho_bar,
#                             so 1 = mean density; subtract 1 for overdensity)
#       RhoMbound_over_RhoMatter               bound-subhalo mass density of
#                             the tracers over the cosmic mean MATTER density
#                             Omega_m rho_crit. A lower limit to the true
#                             matter density contrast: mass outside the
#                             tracers' subhalos is not counted.
#     Mbound is BoundSubhalo/TotalMass.
#
#  3. LEE et al. (2012) parameters (following Park & Choi 2009)
#       Rho20_over_RhoMean    large-scale background density: stellar mass of
#                             the N_RHO = 20 nearest tracers, each weighted by
#                             a cubic-spline kernel whose support radius is
#                             the distance to the 20th, over the mean stellar
#                             mass density of the tracers in the box.
#       Rn_over_rvir          distance to the nearest neighbour over the
#                             neighbour's virial radius. The neighbour is the
#                             nearest tracer with
#                                 Mstar_n >= NeighbourMassRatio * Mstar_target
#                             (0.631 = not more than 0.5 mag fainter, as in
#                             Lee et al.). r_vir,n follows their definition,
#                             (3 M_n / 4 pi / 200 rho_crit)^(1/3), where M_n
#                             is the neighbour's stellar mass scaled up so
#                             that the tracers carry the cosmic mean matter
#                             density (their gamma*L). It is defined in the
#                             same way for centrals and satellites.
#       Rn_over_r200          the same, but with a "physical" neighbour
#                             radius: SOAP R200crit if the neighbour is a
#                             central, else R200 of its bound mass.
#       Neighbour_TrackId etc. identify the neighbour (e.g. to look up its
#                             morphology, as Lee et al. do).
#     Differences from the observations: distances are 3D (no projection, no
#     velocity cut) and masses replace r-band luminosities.
#
#  4. r_kNN_cMpc: distance to the k-th nearest tracer, k in KNN_OUT.
# ==========================================================================

# --------------------------------------------------------------------------
# ---- Simulation information ----

# --- Local test path ---
#BasePath     = "/Users/matthewfrosst/Documents/COLIBRE/" ; SnapBase = "colibre_"
#BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#OutBase      = "/Users/matthewfrosst/Documents/COLIBRE/Frosst_2026_Outputs/"

# --- COSMA paths ---
BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
OutBase      = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"

fname        = "Environment_"

# ---- analysis Information ----
Nstar_min    = 5e3  # Minimum number of stellar particles (targets)
Nstar_max    = 1e10 # Maximum number of stellar particles (targets)

Mstar_tracer_min   = 1e9                         # [Msun] tracers of the environment
ApertureRadii      = np.array([1.0, 5.0, 10.0])  # [comoving Mpc]
N_RHO              = 20                          # neighbours in the smoothed background density
NeighbourMassRatio = 10**(-0.2)                  # neighbour no more than 0.5 mag fainter
KNN_OUT            = (3, 5, 10, 20)              # k-th neighbour distances to write out

N_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


# ==========================================================================
# Pure-numpy/scipy helpers (no simulation dependencies, so they can be
# unit-tested). Positions are comoving and wrapped into [0, box).
# ==========================================================================
def wrap(pos, box):
    """Wrap positions into [0, box), as cKDTree(boxsize=...) requires."""
    pos = np.mod(np.asarray(pos, dtype=np.float64), box)
    pos[pos >= box] -= box          # np.mod can return exactly `box` for tiny negatives
    return pos


def periodic_distance(a, b, box):
    """Distance between points a (n,3) and b (3,) or (n,3) in a periodic box."""
    d  = np.abs(a - b)
    d  = np.minimum(d, box - d)
    return np.sqrt(np.sum(d * d, axis=1))


def r200_comoving(mass, rho_crit_phys, scale_factor):
    """
    Comoving radius of the sphere whose mean (physical) density is 200 times
    the critical density rho_crit_phys [mass / physical length^3].
    """
    return np.cbrt(3.0 * mass / (4.0 * np.pi * 200.0 * rho_crit_phys)) / scale_factor


def spline_kernel(r, H):
    """
    Cubic-spline (M4) kernel in 3D, normalised to unit volume integral, with
    compact support radius H (W = 0 for r >= H). r and H broadcast.
    """
    q = r / H
    w = np.where(q <= 0.5, 1.0 - 6.0 * q**2 + 6.0 * q**3, 2.0 * (1.0 - q)**3)
    return 8.0 / (np.pi * H**3) * np.where(q < 1.0, w, 0.0)


def sum_by_key(keys, weights, query):
    """Sum `weights` over equal `keys`; return (sums, counts) for each key in `query` (0 if absent)."""
    sums = np.zeros(len(query))
    cnts = np.zeros(len(query), dtype=np.int64)
    if len(keys) == 0:
        return sums, cnts
    uniq, inv = np.unique(keys, return_inverse=True)
    s   = np.bincount(inv, weights=weights, minlength=len(uniq))
    c   = np.bincount(inv, minlength=len(uniq))
    j   = np.clip(np.searchsorted(uniq, query), 0, len(uniq) - 1)
    hit = uniq[j] == query
    sums[hit] = s[j[hit]]
    cnts[hit] = c[j[hit]]
    return sums, cnts


def knn_excluding_self(tree, pos, self_idx, k, workers=1):
    """
    The k nearest tree points to each of `pos`, nearest first, leaving out
    the tree point self_idx[i] (the query galaxy itself; -1 if it is not in
    the tree). Returns (dist, idx), both (n, k).
    """
    dist, idx = tree.query(pos, k=k + 1, workers=workers)
    is_self   = idx == self_idx[:, None]
    # stable sort moves the self entry (if any) to the last column, keeping
    # the others in distance order; the last column is then dropped
    order     = np.argsort(is_self, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)


def aperture_sums(tree, pos, radii, weights, self_idx, workers=1, batch=20000):
    """
    For each query point, the number of tree points, and the sum of each of
    the `weights` arrays (one value per tree point), within each of `radii`.
    The tree point self_idx[i] (-1 for none) is not counted.
    Returns counts (n, nR) and sums (nW, n, nR).
    """
    n      = len(pos)
    counts = np.zeros((n, len(radii)), dtype=np.int64)
    sums   = np.zeros((len(weights), n, len(radii)))
    for s in range(0, n, batch):
        p = pos[s:s + batch]
        for ir, R in enumerate(radii):
            nb    = tree.query_ball_point(p, R, workers=workers, return_sorted=False)
            lens  = np.fromiter((len(l) for l in nb), dtype=np.int64, count=len(p))
            flat  = np.fromiter(itertools.chain.from_iterable(nb), dtype=np.int64,
                                count=int(lens.sum()))
            owner = np.repeat(np.arange(len(p)), lens)
            counts[s:s + batch, ir] = lens
            for iw, w in enumerate(weights):
                sums[iw, s:s + batch, ir] = np.bincount(owner, weights=w[flat], minlength=len(p))
    own = self_idx >= 0                      # the query galaxy is always inside its own sphere
    counts[own] -= 1
    for iw, w in enumerate(weights):
        sums[iw, own] -= w[self_idx[own]][:, None]
    return counts, np.maximum(sums, 0.0)     # clip rounding residue of the self subtraction


def smoothed_density(dist, idx, mass):
    """
    Kernel-smoothed mass density from each point's k nearest neighbours
    (dist, idx from knn_excluding_self): sum_i m_i W(r_i; H), with the kernel
    support H equal to the distance to the k-th neighbour.
    """
    H = dist[:, -1][:, None]
    return np.sum(mass[idx] * spline_kernel(dist, H), axis=1)


def nearest_massive_neighbour(tree, pos_tr, mass_tr, pos, mass_min, self_idx, box,
                              dist0=None, idx0=None, workers=1, k_max=2048, verbose=False):
    """
    For each query point i, the nearest tree point (not self_idx[i]) with
    mass_tr >= mass_min[i]. Returns (dist, idx); (nan, -1) if there is none.

    Most galaxies have such a neighbour among their few nearest tracers, so:
      1. look through an existing k-nearest-neighbour list (dist0, idx0);
      2. for those left over, repeat with k four times larger, up to k_max;
      3. the few still left are the most massive galaxies, for which only a
         handful of tracers qualify: those are searched directly.
    """
    n        = len(pos)
    nb_dist  = np.full(n, np.nan)
    nb_idx   = np.full(n, -1, dtype=np.int64)
    n_tr     = len(mass_tr)

    def resolve(rows, d, ix):
        ok    = mass_tr[ix] >= mass_min[rows][:, None]
        has   = ok.any(axis=1)
        first = ok.argmax(axis=1)
        r     = rows[has]
        nb_dist[r] = d[has, first[has]]
        nb_idx[r]  = ix[has, first[has]]
        return rows[~has]

    todo = np.arange(n)
    k    = 0
    if dist0 is not None:
        k    = dist0.shape[1]
        todo = resolve(todo, dist0, idx0)
    while todo.size:
        k = max(4 * k, 8)
        if k > k_max or k + 1 >= n_tr:
            break
        step = max(1, 20_000_000 // k)       # keep the (rows x k) arrays small
        left = []
        for s in range(0, todo.size, step):
            rows  = todo[s:s + step]
            d, ix = knn_excluding_self(tree, pos[rows], self_idx[rows], k, workers)
            left.append(resolve(rows, d, ix))
        todo = np.concatenate(left)
        if verbose:
            print(f'    nearest neighbour: {todo.size} unresolved after k = {k}', flush=True)

    if todo.size:
        order    = np.argsort(-mass_tr, kind='stable')     # most massive first
        neg_sort = -mass_tr[order]
        for i in todo:
            n_cand = np.searchsorted(neg_sort, -mass_min[i], side='right')
            cand   = order[:n_cand]
            cand   = cand[cand != self_idx[i]]
            if cand.size == 0:
                continue                                   # nothing massive enough in the box
            d = periodic_distance(pos_tr[cand], pos[i], box)
            j = np.argmin(d)
            nb_dist[i], nb_idx[i] = d[j], cand[j]
    return nb_dist, nb_idx


# ==========================================================================
if __name__ == "__main__":

    import h5py       as h5
    import swiftsimio as sw

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=N_WORKERS,
                        help="threads for the kd-tree queries (default: SLURM_CPUS_PER_TASK)")
    args = parser.parse_args()
    N_WORKERS = max(1, args.n_workers)

    for idir, Dir in enumerate(BoxDir):

        ext4 = str(snap).zfill(4)
        ext3 = str(snap).zfill(3)

        soap_file = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"

        # --- Read SOAP halo data
        print('SOAP file: ', soap_file)
        soap        = sw.load(soap_file)
        ScaleFactor = float(soap.metadata.scale_factor)
        Redshift    = 1./ScaleFactor - 1
        box         = float(soap.metadata.boxsize.to('Mpc').value[0])     # comoving Mpc

        if np.max(ApertureRadii) >= 0.5 * box:
            sys.exit(f"Largest aperture ({np.max(ApertureRadii)} cMpc) must be smaller than half "
                     f"the box ({box:.1f} cMpc).")

        # --- Selections. Only these whole-catalogue arrays are needed to
        # --- decide which subhalos matter; everything else is read once and
        # --- cut down to those subhalos straight away.
        Nstar_all = np.asarray(soap.bound_subhalo.number_of_star_particles.value)
        mstar_q   = soap.exclusive_sphere_50kpc.stellar_mass
        mstar_q.convert_to_units('Msun') ; mstar_q.convert_to_physical()
        Mstar_all = np.asarray(mstar_q.value, dtype=np.float64)
        cen_all   = np.asarray(soap.input_halos.is_central.value).astype(bool)
        host_all  = np.asarray(soap.soap.host_halo_index.value).astype(np.int64)
        n_all     = len(Nstar_all)
        del mstar_q

        lhalo     = np.where((Nstar_all >= Nstar_min) & (Nstar_all <= Nstar_max))[0]   # targets
        ltracer   = np.where(Mstar_all >= Mstar_tracer_min)[0]                         # tracers

        # host (central) of every subhalo: itself for centrals, -1 if hostless
        host_all  = np.where(cen_all, np.arange(n_all), host_all)

        lhost     = host_all[np.concatenate((lhalo, ltracer))]
        need      = np.unique(np.concatenate((lhalo, ltracer, lhost[lhost >= 0])))

        print('- snapshot        :', snap)
        print(f'  Number of galaxies selected: {len(lhalo)}')
        print(f'  Number of tracers (Mstar >= {Mstar_tracer_min:.1e} Msun): {len(ltracer)}')
        print(f'  Subhalos kept: {len(need)} of {n_all}')
        if len(ltracer) <= max(N_RHO, max(KNN_OUT)) + 1:
            sys.exit("Too few tracers for the nearest-neighbour statistics; lower Mstar_tracer_min.")

        # positions in the `need` arrays
        tgt       = np.searchsorted(need, lhalo)
        trc       = np.searchsorted(need, ltracer)
        host_cat  = host_all[need]
        host      = np.where(host_cat >= 0, np.searchsorted(need, np.maximum(host_cat, 0)), -1)

        Mstar      = Mstar_all[need]
        is_central = cen_all[need]
        del Nstar_all, Mstar_all, cen_all, host_all

        # --- remaining SOAP properties, for the kept subhalos only
        centre    = soap.input_halos.halo_centre[need]
        Mbound    = soap.bound_subhalo.total_mass[need]
        M200      = soap.spherical_overdensity_200_crit.total_mass[need]
        r200      = soap.spherical_overdensity_200_crit.soradius[need]
        fsub      = soap.spherical_overdensity_200_crit.mass_fraction_satellites[need]
        TrackId   = np.asarray(soap.input_halos_hbtplus.track_id[need].value).astype(np.int64)

        # --- convert Units (lengths: comoving Mpc; masses: Msun)
        centre.convert_to_units('Mpc')  ; centre.convert_to_comoving()
        r200.convert_to_units('Mpc')    ; r200.convert_to_comoving()
        Mbound.convert_to_units('Msun') ; Mbound.convert_to_physical()
        M200.convert_to_units('Msun')   ; M200.convert_to_physical()

        pos       = wrap(centre.value, box)
        Mbound    = np.asarray(Mbound.value, dtype=np.float64)
        M200      = np.asarray(M200.value,   dtype=np.float64)
        r200      = np.asarray(r200.value,   dtype=np.float64)
        fsub      = np.asarray(fsub.value,   dtype=np.float64)

        # --- cosmology: critical density (physical) and mean matter density (comoving)
        cosmo         = soap.metadata.cosmology
        rho_crit      = float(cosmo.critical_density(Redshift).to('Msun/Mpc**3').value)   # physical
        rho_matter    = float(cosmo.Om0 * cosmo.critical_density0.to('Msun/Mpc**3').value) # comoving

        # --- tracer sample and its box-mean densities (comoving)
        pos_tr        = pos[trc]
        mstar_tr      = Mstar[trc]
        mbound_tr     = Mbound[trc]
        volume        = box**3
        nbar          = len(trc) / volume
        rhobar_star   = mstar_tr.sum()  / volume
        rhobar_bound  = mbound_tr.sum() / volume
        print(f'  Box {box:.1f} cMpc; tracer means: n = {nbar:.3e} cMpc^-3, '
              f'rho_star = {rhobar_star:.3e}, rho_bound = {rhobar_bound:.3e} Msun cMpc^-3 '
              f'(rho_bound / rho_matter = {rhobar_bound / rho_matter:.2f}); {N_WORKERS} workers')

        tree          = cKDTree(pos_tr, boxsize=box)

        # index of each target in the tracer arrays (-1 if it is not a tracer)
        tr_of         = np.full(len(need), -1, dtype=np.int64)
        tr_of[trc]    = np.arange(len(trc))
        self_idx      = tr_of[tgt]
        pos_tg        = pos[tgt]
        mstar_tg      = Mstar[tgt]
        ntg           = len(tgt)

        # ---------------- 1. host halo ----------------
        h_tg          = host[tgt]
        has_host      = h_tg >= 0
        hh            = np.maximum(h_tg, 0)
        Host_TrackId  = np.where(has_host, TrackId[hh], -1)
        Host_M200     = np.where(has_host, M200[hh], np.nan)
        Host_r200     = np.where(has_host, r200[hh], np.nan)
        Host_fsub     = np.where(has_host, fsub[hh], np.nan)
        no_so         = ~(Host_r200 > 0)                    # SO not found (or no host)
        Host_M200[no_so] = Host_r200[no_so] = Host_fsub[no_so] = np.nan
        DistToHost    = np.where(has_host, periodic_distance(pos_tg, pos[hh], box), np.nan)

        # tracer satellites within R200 of their central, summed per central
        h_tr          = host[trc]
        sat           = (h_tr >= 0) & ~is_central[trc]
        hs            = h_tr[sat]
        inside        = periodic_distance(pos_tr[sat], pos[hs], box) <= r200[hs]
        Msat, Nsat    = sum_by_key(hs[inside], mstar_tr[sat][inside], hh)
        Host_Nsat     = np.where(no_so, -1, Nsat)
        Host_Msat     = np.where(no_so, np.nan, Msat)
        Host_fsat     = Host_Msat / (Host_Msat + Mstar[hh])
        print(f'  Host halo: {int(np.sum(~no_so))}/{ntg} targets have a host with R200 '
              f'({int(np.sum(~has_host))} hostless)', flush=True)

        # ---------------- 2. fixed-aperture densities ----------------
        N_in_R, (Mstar_in_R, Mbound_in_R) = aperture_sums(
            tree, pos_tg, ApertureRadii, (mstar_tr, mbound_tr), self_idx, workers=N_WORKERS)
        Vsphere       = 4.0 / 3.0 * np.pi * ApertureRadii**3
        Delta_N       = N_in_R      / Vsphere / nbar
        Delta_Mstar   = Mstar_in_R  / Vsphere / rhobar_star
        Delta_Mbound  = Mbound_in_R / Vsphere / rhobar_bound
        Rho_Mb_matter = Mbound_in_R / Vsphere / rho_matter
        for ir, R in enumerate(ApertureRadii):
            print(f'  R = {R:4.1f} cMpc: median N = {np.median(N_in_R[:, ir]):.0f}, '
                  f'median rho_star/mean = {np.median(Delta_Mstar[:, ir]):.2f}, '
                  f'empty spheres: {int(np.sum(N_in_R[:, ir] == 0))}', flush=True)

        # ---------------- 3. Lee et al. (2012) parameters ----------------
        k_nn          = max(N_RHO, max(KNN_OUT))
        nn_dist, nn_idx = knn_excluding_self(tree, pos_tg, self_idx, k_nn, workers=N_WORKERS)

        Rho20         = smoothed_density(nn_dist[:, :N_RHO], nn_idx[:, :N_RHO], mstar_tr)
        Rho20_norm    = Rho20 / rhobar_star
        r_kNN         = nn_dist[:, np.array(KNN_OUT) - 1]

        Rn, nb        = nearest_massive_neighbour(
            tree, pos_tr, mstar_tr, pos_tg, NeighbourMassRatio * mstar_tg, self_idx, box,
            dist0=nn_dist, idx0=nn_idx, workers=N_WORKERS, verbose=True)
        has_nb        = nb >= 0
        nbk           = trc[np.maximum(nb, 0)]              # neighbour's position in the `need` arrays

        # Lee et al.-style virial radius: stellar mass scaled so that the
        # tracers carry the cosmic mean matter density
        rvir_lee      = r200_comoving(Mstar * rho_matter / rhobar_star, rho_crit, ScaleFactor)
        # "physical" radius: SOAP R200crit for centrals, R200 of the bound mass otherwise
        r200_phys     = np.where(is_central & (r200 > 0), r200,
                                 r200_comoving(Mbound, rho_crit, ScaleFactor))

        Nb_TrackId    = np.where(has_nb, TrackId[nbk], -1)
        Nb_Mstar      = np.where(has_nb, Mstar[nbk], np.nan)
        Nb_central    = np.where(has_nb, is_central[nbk], -1)
        Nb_rvir       = np.where(has_nb, rvir_lee[nbk], np.nan)
        Nb_r200       = np.where(has_nb, r200_phys[nbk], np.nan)
        Rn_over_rvir  = Rn / Nb_rvir
        Rn_over_r200  = Rn / Nb_r200

        print(f'  rho20/mean: median = {np.median(Rho20_norm):.2f}  '
              f'(16-84%: {np.percentile(Rho20_norm, 16):.2f}-{np.percentile(Rho20_norm, 84):.2f})')
        print(f'  Rn/rvir   : median = {np.nanmedian(Rn_over_rvir):.2f}  '
              f'(fraction < 1: {np.nanmean(Rn_over_rvir[has_nb] < 1):.2f}); '
              f'{int(np.sum(~has_nb))} targets have no neighbour massive enough')

        # ---------------- write ----------------
        fn = OutBase + Dir + RunDir + fname + ext3 + ".hdf5"
        print('\n Writing to:', fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        with h5.File(fn, "w") as output:
            grp0 = output.create_group("Header")
            grp1 = output.create_group("HaloData")

            grp0.create_dataset('Redshift',             data=Redshift,           dtype='float')
            grp0.create_dataset('BoxSize_cMpc',         data=box,                dtype='float')
            grp0.create_dataset('MstarTracerMin',       data=Mstar_tracer_min,   dtype='float')
            grp0.create_dataset('NTracers',             data=len(trc),           dtype='int')
            grp0.create_dataset('ApertureRadii_cMpc',   data=ApertureRadii,      dtype='float')
            grp0.create_dataset('KNN',                  data=np.array(KNN_OUT),  dtype='int')
            grp0.create_dataset('NRho',                 data=N_RHO,              dtype='int')
            grp0.create_dataset('NeighbourMassRatio',   data=NeighbourMassRatio, dtype='float')
            # box means of the tracers and cosmic densities, Msun / cMpc^3 (number: cMpc^-3)
            grp0.create_dataset('MeanNumberDensity',    data=nbar,               dtype='float')
            grp0.create_dataset('MeanMstarDensity',     data=rhobar_star,        dtype='float')
            grp0.create_dataset('MeanMboundDensity',    data=rhobar_bound,       dtype='float')
            grp0.create_dataset('MeanMatterDensity',    data=rho_matter,         dtype='float')
            grp0.create_dataset('CriticalDensity_phys', data=rho_crit,           dtype='float')  # Msun / pMpc^3
            grp0.attrs['SOAPFile'] = soap_file

            grp1.create_dataset('TrackId',              data=TrackId[tgt],       dtype='int')
            grp1.create_dataset('is_central',           data=is_central[tgt],    dtype='int')
            grp1.create_dataset('is_tracer',            data=self_idx >= 0,      dtype='int')
            grp1.create_dataset('StellarMass',          data=mstar_tg,           dtype='float')

            # --- 1. host halo ---
            grp1.create_dataset('Host_TrackId',         data=Host_TrackId,       dtype='int')
            grp1.create_dataset('Host_M200',            data=Host_M200,          dtype='float')
            grp1.create_dataset('Host_r200_cMpc',       data=Host_r200,          dtype='float')
            grp1.create_dataset('Host_fsub200',         data=Host_fsub,          dtype='float')
            grp1.create_dataset('DistToHost_over_r200', data=DistToHost / Host_r200, dtype='float')
            grp1.create_dataset('Host_Nsat_R200',       data=Host_Nsat,          dtype='int')
            grp1.create_dataset('Host_Msat_star_R200',  data=Host_Msat,          dtype='float')
            grp1.create_dataset('Host_fsat_star_R200',  data=Host_fsat,          dtype='float')

            # --- 2. fixed apertures: (galaxy x ApertureRadii_cMpc) ---
            grp1.create_dataset('N_in_R',               data=N_in_R,             dtype='int')
            grp1.create_dataset('Mstar_in_R',           data=Mstar_in_R,         dtype='float')
            grp1.create_dataset('Mbound_in_R',          data=Mbound_in_R,        dtype='float')
            grp1.create_dataset('Delta_N',              data=Delta_N,            dtype='float')
            grp1.create_dataset('Delta_Mstar',          data=Delta_Mstar,        dtype='float')
            grp1.create_dataset('Delta_Mbound',         data=Delta_Mbound,       dtype='float')
            grp1.create_dataset('RhoMbound_over_RhoMatter', data=Rho_Mb_matter,  dtype='float')

            # --- 3. Lee et al. (2012) ---
            grp1.create_dataset('Rho20_over_RhoMean',   data=Rho20_norm,         dtype='float')
            grp1.create_dataset('Rn_cMpc',              data=Rn,                 dtype='float')
            grp1.create_dataset('Rn_over_rvir',         data=Rn_over_rvir,       dtype='float')
            grp1.create_dataset('Rn_over_r200',         data=Rn_over_r200,       dtype='float')
            grp1.create_dataset('Neighbour_TrackId',    data=Nb_TrackId,         dtype='int')
            grp1.create_dataset('Neighbour_StellarMass',data=Nb_Mstar,           dtype='float')
            grp1.create_dataset('Neighbour_is_central', data=Nb_central,         dtype='int')
            grp1.create_dataset('Neighbour_rvir_cMpc',  data=Nb_rvir,            dtype='float')
            grp1.create_dataset('Neighbour_r200_cMpc',  data=Nb_r200,            dtype='float')

            # --- 4. k-th nearest tracer: (galaxy x KNN) ---
            grp1.create_dataset('r_kNN_cMpc',           data=r_kNN,              dtype='float')

    sys.exit()
