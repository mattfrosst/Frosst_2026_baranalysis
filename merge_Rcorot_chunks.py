"""
Merge the per-chunk CoRotation output files (from Calculate_Bar_Rcorot.py)
into a single HDF5 file.
"""
import os
import re
import glob
import sys
import numpy as np
import h5py as h5

# --- Adjust these to match your run (same values as Calculate_Bar_Rcorot.py) ---
BoxDir   = "L200_m6/"
RunDir   = "THERMAL_AGN_m6/"
fname    = "CoRotation_"
snap     = 127
BasePath = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/" #COSMA path

TAG      = "_merged"   # merged file is <fname><snap><TAG>.hdf5 (e.g. --tag _barbins)


def set_paths(snap_number, tag=TAG):
    """Chunk pattern and merged-file name for one snapshot."""
    global ext3, OutputsDir, chunk_glob, merged_fn
    ext3       = str(snap_number).zfill(3)
    OutputsDir = BasePath + BoxDir + RunDir
    chunk_glob = OutputsDir + fname + ext3 + "_chunk*.hdf5"
    merged_fn  = OutputsDir + fname + ext3 + tag + ".hdf5"


set_paths(snap)

DATASETS_HALO = [
    'TrackId', 'r50_stars',
    'PatternSpeed_kms_per_kpc', 'PatternSpeedErr_kms_per_kpc',
    'R_bar_kpc',
    'R_CR_kpc', 'R_CR_lo_kpc', 'R_CR_hi_kpc',
    'R_CR_over_R_bar', 'R_CR_over_r50',
    'Vc_at_CR_kms', 'Vc_at_Rbar_kms', 'Omega_c_at_Rbar_kms_per_kpc', 'q_Omega',
    'Status', 'NCrossings',
]

# 2-D (halo x radius) profiles, stacked along the halo axis. RadiusKpc is
# per galaxy (the bar-profile grid Radius_r50 times that galaxy's r50).
DATASETS_PROF = ['RadiusKpc', 'Menc_stars', 'Menc_gas', 'Menc_dm', 'Vc_kms']

# Header values that must be identical in every chunk
HEADER_FLOATS = ['Redshift', 'CellSizeKpc', 'RMin_r50', 'RMax_r50', 'RCutKpc']
HEADER_ATTRS  = ['OmegaFile', 'RegionFile']

STATUS_LABELS = {
    0: 'R_CR found',
    1: 'R_CR < 0.1 r50 (inside the grid)',
    2: 'R_CR > min(10 r50, 50 kpc) (beyond the grid)',
    3: 'skipped (crosses cell boundary)',
    4: 'invalid inputs',
    5: 'not barred',
    6: 'barred, no valid pattern speed',
}

# Per-chunk halo counts in the Header, summed over chunks. Together they
# must add up to the number of rows merged.
HEADER_COUNTS = ['NHalosIncluded', 'NHalosSkipped', 'NHalosNotBarred', 'NHalosNoOmega']


def find_chunk_files():
    paths = sorted(glob.glob(chunk_glob))
    if not paths:
        sys.exit(f"No chunk files found matching: {chunk_glob}")

    pattern = re.compile(r"_chunk(\d+)\.hdf5$")
    by_id = {}
    for p in paths:
        m = pattern.search(p)
        if not m:
            sys.exit(f"Could not parse a chunk id out of filename: {p}")
        cid = int(m.group(1))
        if cid in by_id:
            sys.exit(f"Duplicate chunk id {cid} found ({by_id[cid]} and {p}) "
                      f"-- refusing to guess which is correct.")
        by_id[cid] = p
    return by_id


def attr_str(value):
    """h5py may return string attributes as bytes; normalise to str."""
    return value.decode() if isinstance(value, bytes) else str(value)


def main():
    if os.path.exists(merged_fn):
        sys.exit(f"Merged output already exists: {merged_fn}\n"
                  f"Refusing to overwrite it. Move/rename/delete it yourself "
                  f"first if you want to regenerate the merge.")

    by_id = find_chunk_files()

    # --- Reference values from chunk 0 (or the first file found) ---
    first_path = by_id.get(0, next(iter(by_id.values())))
    with h5.File(first_path, 'r') as f:
        hdr                = f['Header']
        if 'Radius_r50' not in f['Profiles'] or 'RCutKpc' not in hdr:
            sys.exit(f"{first_path} was made by an older Calculate_Bar_Rcorot.py "
                      f"(fixed 1-50 kpc grid, not the bar-profile grid). "
                      f"Re-run all chunks before merging.")
        n_chunks_expected  = int(hdr['NChunks'][()])
        header_ref         = {k: float(hdr[k][()]) for k in HEADER_FLOATS}
        attrs_ref          = {k: attr_str(hdr.attrs[k]) for k in HEADER_ATTRS}
        radius_ref         = f['Profiles']['Radius_r50'][()]   # the shared bar-profile grid

    missing = [cid for cid in range(n_chunks_expected) if cid not in by_id]
    if missing:
        sys.exit(f"Missing chunk file(s) for chunk id(s): {missing}. "
                  f"Found {len(by_id)}/{n_chunks_expected} chunks. Aborting "
                  f"-- re-run the missing chunks before merging.")

    extra = [cid for cid in by_id if cid >= n_chunks_expected]
    if extra:
        sys.exit(f"Found chunk file(s) with id(s) {extra}, but the run's "
                  f"header says NChunks={n_chunks_expected}. This looks "
                  f"like output from a different run mixed in -- aborting.")

    print(f'Found all {n_chunks_expected} expected chunk files. Merging...')

    # --- Read each chunk (read-only), validate, and collect arrays ---
    per_chunk_halo = []
    per_chunk_prof = []
    totals         = {k: 0 for k in HEADER_COUNTS}

    for cid in range(n_chunks_expected):
        path = by_id[cid]
        with h5.File(path, 'r') as f:
            hdr = f['Header']
            if int(hdr['NChunks'][()]) != n_chunks_expected:
                sys.exit(f"Chunk {cid} ({path}) has NChunks="
                          f"{int(hdr['NChunks'][()])}, expected "
                          f"{n_chunks_expected}. Refusing to merge "
                          f"inconsistent chunk sets.")
            if int(hdr['ChunkId'][()]) != cid:
                sys.exit(f"File {path} has ChunkId={int(hdr['ChunkId'][()])} "
                          f"in its own header, but was matched to chunk id "
                          f"{cid} by its filename. Aborting -- filenames "
                          f"and internal headers disagree.")
            for k, ref in header_ref.items():
                here = float(hdr[k][()])
                if abs(here - ref) > 1e-6 * max(1.0, abs(ref)):
                    sys.exit(f"Chunk {cid} has {k}={here}, expected {ref}. "
                              f"Aborting -- chunks don't look like they're "
                              f"from the same run/settings.")
            for k, ref in attrs_ref.items():
                here = attr_str(hdr.attrs[k])
                if here != ref:
                    sys.exit(f"Chunk {cid} was made from {k}={here}, but chunk "
                              f"0 used {ref}. Aborting -- chunks were built from "
                              f"different bar-analysis inputs.")

            for k in HEADER_COUNTS:
                if k not in hdr:
                    sys.exit(f"Chunk {cid} ({path}) has no Header/{k}. It was "
                              f"probably made by an older Calculate_Bar_Rcorot.py "
                              f"(barred galaxies only) -- re-run it before merging.")
                totals[k] += int(hdr[k][()])

            prof = f['Profiles']
            if not np.array_equal(prof['Radius_r50'][()], radius_ref):
                sys.exit(f"Chunk {cid} has a different radial grid (Radius_r50) "
                          f"from chunk 0. Aborting -- profiles can't be stacked.")

            grp  = f['HaloData']
            data = {name: grp[name][()] for name in DATASETS_HALO}
            pdat = {name: prof[name][()] for name in DATASETS_PROF}

            n_here = len(data['TrackId'])
            for name, arr in data.items():
                if len(arr) != n_here:
                    sys.exit(f"Chunk {cid}: HaloData/{name} has {len(arr)} "
                              f"entries but TrackId has {n_here}. Aborting.")
            for name, arr in pdat.items():
                if arr.shape != (n_here, len(radius_ref)):
                    sys.exit(f"Chunk {cid}: Profiles/{name} has shape {arr.shape}, "
                              f"expected ({n_here}, {len(radius_ref)}). Aborting.")

            per_chunk_halo.append(data)
            per_chunk_prof.append(pdat)

        print(f'  chunk {cid}: {n_here} halos read from {path}')

    # --- Concatenate in chunk-id order ---
    merged_halo = {
        name: np.concatenate([d[name] for d in per_chunk_halo])
        for name in DATASETS_HALO
    }
    merged_prof = {
        name: np.concatenate([d[name] for d in per_chunk_prof], axis=0)
        for name in DATASETS_PROF
    }
    n_total = len(merged_halo['TrackId'])

    # Chunks are disjoint slices of one halo list, so a repeated TrackId
    # means chunk files from different runs were mixed together.
    uniq, counts = np.unique(merged_halo['TrackId'], return_counts=True)
    if np.any(counts > 1):
        sys.exit(f"{int(np.sum(counts > 1))} TrackId(s) appear in more than one "
                  f"chunk (e.g. {uniq[counts > 1][:5].tolist()}). Aborting.")

    if n_total != sum(totals.values()):
        sys.exit(f"Halo count mismatch: {n_total} halos merged, but chunk headers "
                  f"sum to {sum(totals.values())} ({totals}). Aborting.")

    print(f'\nTotal halos after merge: {n_total} ('
          + ', '.join(f'{k}: {v}' for k, v in totals.items()) + ')')
    status = merged_halo['Status']
    for s, label in STATUS_LABELS.items():
        print(f'  Status {s} ({label}): {int(np.sum(status == s))}')

    # --- Write the new merged file (mode 'x' = create, error if exists,
    #     so this can never silently clobber anything) ---
    with h5.File(merged_fn, 'x') as out:
        grp0 = out.create_group("Header")
        for k, v in header_ref.items():
            grp0.create_dataset(k, data=v, dtype='float')
        grp0.create_dataset('NChunksMerged',  data=n_chunks_expected, dtype='int')
        grp0.create_dataset('NHalosTotal',    data=n_total,           dtype='int')
        for k, v in totals.items():
            grp0.create_dataset(k, data=v, dtype='int')
        for k, v in attrs_ref.items():
            grp0.attrs[k] = v

        grp1 = out.create_group("HaloData")
        for name, arr in merged_halo.items():
            grp1.create_dataset(name, data=arr, dtype=arr.dtype)

        grp2 = out.create_group("Profiles")
        grp2.create_dataset('Radius_r50', data=radius_ref, dtype=radius_ref.dtype)
        for name, arr in merged_prof.items():
            grp2.create_dataset(name, data=arr, dtype=arr.dtype)

    print(f'\nWrote merged file: {merged_fn}')
    print('Source chunk files were opened read-only and were not modified.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, default=snap, help=f"snapshot number (default: {snap})")
    parser.add_argument("--tag", default=TAG,
                        help=f"suffix for the merged file name (default: {TAG})")
    args = parser.parse_args()
    set_paths(args.snap, args.tag)
    main()
