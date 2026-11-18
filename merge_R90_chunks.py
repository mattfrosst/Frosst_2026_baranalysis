"""
Merge the per-chunk Calculate_R90.py outputs (Stellar_Sizes_<snap>_chunk*.hdf5) into one file,
Stellar_Sizes_<snap>.hdf5, with galaxies in the full Nstar-selected order (as a single-chunk run).

Safety design (as merge_fDM_chunks.py):
  - Chunk files are opened strictly read-only.
  - The merged file is opened with mode 'x' (create, fail if it already exists), so it is never
    overwritten. Its name has no "_chunk" in it, so it can't collide with a chunk file.
  - Before writing, every expected chunk id (0 .. NChunks-1) must be present with consistent
    headers, and every galaxy (GlobalIndex 0 .. NGalTotal-1) must appear exactly once.

Usage:  python merge_R90_chunks.py --snap 127
"""
import os
import re
import glob
import sys
import argparse
import numpy as np
import h5py as h5

# --- Adjust these to match your run ---
BoxDir   = "L200_m6/"
RunDir   = "THERMAL_AGN_m6/"
fname    = "Stellar_Sizes_"
BasePath = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/" #COSMA path

HALO_SETS = ['TrackId', 'is_central', 'StellarMass', 'r50_stars_SOAP', 'NumStarPart', 'Skipped']
SIZE_SETS = [f'{d}{p}_{w}' for w in ('Mass', 'g', 'r', 'i') for p in (50, 90) for d in ('R', 'r')]


def find_chunk_files(chunk_glob):
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
            sys.exit(f"Duplicate chunk id {cid} found ({by_id[cid]} and {p}) -- refusing to guess which is correct.")
        by_id[cid] = p
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, required=True)
    parser.add_argument("--aperture", type=float, default=50.0,
                        help="aperture [physical kpc] the chunks were run with (default: 50, no filename tag)")
    args = parser.parse_args()

    ext3       = str(args.snap).zfill(3) + ("" if args.aperture == 50.0 else f"_ap{args.aperture:g}kpc")
    OutputsDir = BasePath + BoxDir + RunDir
    chunk_glob = OutputsDir + fname + ext3 + "_chunk*.hdf5"
    merged_fn  = OutputsDir + fname + ext3 + ".hdf5"

    if os.path.exists(merged_fn):
        sys.exit(f"Merged output already exists: {merged_fn}\n"
                 f"Refusing to overwrite it. Move/rename/delete it yourself first if you want to regenerate the merge.")

    by_id = find_chunk_files(chunk_glob)
    with h5.File(next(iter(by_id.values())), 'r') as f:
        hdr = f['Header']
        n_chunks  = int(hdr['NChunks'][()])
        n_gal     = int(hdr['NGalTotal'][()])
        redshift  = float(hdr['Redshift'][()])
        aperture  = float(hdr['ApertureKpc'][()])
        cell_size = float(hdr['CellSizeKpc'][()])
        desc      = hdr.attrs['description']
    if abs(aperture - args.aperture) > 1e-9:
        sys.exit(f"Chunks have ApertureKpc = {aperture:g}, but --aperture {args.aperture:g} was given. Aborting.")

    missing = [c for c in range(n_chunks) if c not in by_id]
    if missing:
        sys.exit(f"Missing chunk file(s) for chunk id(s): {missing}. Found {len(by_id)}/{n_chunks}. "
                 f"Re-run the missing chunks before merging.")
    extra = [c for c in by_id if c >= n_chunks]
    if extra:
        sys.exit(f"Found chunk id(s) {extra}, but the header says NChunks={n_chunks} -- looks like a different run mixed in.")
    print(f'Found all {n_chunks} chunk files for snapshot {args.snap}. Merging {n_gal} galaxies...')

    halo  = {}
    sizes = {name: np.full(n_gal, np.nan) for name in SIZE_SETS}
    seen  = np.zeros(n_gal, dtype=int)
    n_skipped = n_included = 0

    for cid in range(n_chunks):
        path = by_id[cid]
        with h5.File(path, 'r') as f:
            hdr = f['Header']
            checks = [('NChunks', n_chunks), ('ChunkId', cid), ('NGalTotal', n_gal)]
            for key, expected in checks:
                if int(hdr[key][()]) != expected:
                    sys.exit(f"{path}: header {key}={int(hdr[key][()])}, expected {expected}. Aborting.")
            if abs(float(hdr['Redshift'][()]) - redshift) > 1e-6 or abs(float(hdr['ApertureKpc'][()]) - aperture) > 1e-9:
                sys.exit(f"{path}: redshift or ApertureKpc differs from the other chunks. Aborting.")
            n_skipped  += int(hdr['NHalosSkipped'][()])
            n_included += int(hdr['NHalosIncluded'][()])

            gi = f['HaloData']['GlobalIndex'][()]
            seen[gi] += 1
            for name in HALO_SETS:
                arr = f['HaloData'][name][()]
                if name not in halo:
                    halo[name] = np.zeros(n_gal, dtype=arr.dtype)
                halo[name][gi] = arr
            for name in SIZE_SETS:
                sizes[name][gi] = f['Sizes'][name][()]
        print(f'  chunk {cid}: {len(gi)} galaxies read from {path}')

    if np.any(seen != 1):
        sys.exit(f"Galaxy coverage problem: {np.sum(seen == 0)} galaxies in no chunk, "
                 f"{np.sum(seen > 1)} in more than one. Aborting -- nothing written.")

    with h5.File(merged_fn, 'x') as out:
        grp0 = out.create_group("Header")
        grp0.create_dataset('Redshift',       data=redshift)
        grp0.create_dataset('ApertureKpc',    data=aperture)
        grp0.create_dataset('CellSizeKpc',    data=cell_size)
        grp0.create_dataset('NChunksMerged',  data=n_chunks)
        grp0.create_dataset('NGalTotal',      data=n_gal)
        grp0.create_dataset('NHalosSkipped',  data=n_skipped)
        grp0.create_dataset('NHalosIncluded', data=n_included)
        grp0.attrs['description'] = desc
        grp1 = out.create_group("HaloData")
        for name, arr in halo.items():
            grp1.create_dataset(name, data=arr, dtype=arr.dtype)
        grp2 = out.create_group("Sizes")
        for name, arr in sizes.items():
            grp2.create_dataset(name, data=arr, dtype='float')

    print(f'\nWrote merged file: {merged_fn}  ({n_included} measured, {n_skipped} skipped at the box edge)')
    print('Source chunk files were opened read-only and were not modified.')


if __name__ == "__main__":
    main()
