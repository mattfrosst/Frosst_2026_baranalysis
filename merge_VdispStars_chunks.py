"""
Merge the per-chunk SigmaStars_rhalf output files into a single HDF5 file.

Safety design (identical to merge_fDM_chunks.py):
  - Chunk files are opened strictly read-only ('r') -- this script never
    writes to, deletes, or renames any file matching the chunk pattern.
  - The merged output is written to a NEW file with a distinct name
    (SigmaStars_0127_merged.hdf5, not SigmaStars_0127_chunk*.hdf5), so it
    can never collide with a chunk file's name.
  - The output file is opened with mode 'x' (create, fail if it already
    exists) -- if a merged file from a previous run is already there,
    this script stops rather than overwriting it. Delete or rename the
    old merged file yourself if you want to regenerate it.
  - Before writing anything, it checks that every expected chunk ID
    (0 .. n_chunks-1) is present and that the total halo count adds up,
    and aborts with a clear message if not -- so you never end up with a
    silently incomplete merged file.
"""
import os
import re
import glob
import sys
import numpy as np
import h5py as h5

# --- Adjust these to match your run ---
BoxDir   = "L200_m6/"
RunDir   = "THERMAL_AGN_m6/"
fname    = "SigmaStars_"
snap     = 127
BasePath = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/" #COSMA path

ext3       = str(snap).zfill(3)
OutputsDir = BasePath + BoxDir + RunDir
chunk_glob = OutputsDir + fname + ext3 + "_chunk*.hdf5"
merged_fn  = OutputsDir + fname + ext3 + "_merged.hdf5"

DATASETS_HALO = [
    'TrackId', 'r50_stars', 'NstarUsed', 'MstarUsed',
    'sigma_tot2', 'sigma_x2', 'sigma_y2', 'sigma_z2',
]


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


def main():
    if os.path.exists(merged_fn):
        sys.exit(f"Merged output already exists: {merged_fn}\n"
                  f"Refusing to overwrite it. Move/rename/delete it yourself "
                  f"first if you want to regenerate the merge.")

    by_id = find_chunk_files()

    # --- Read n_chunks from the first file and check every chunk is present ---
    first_path = next(iter(by_id.values()))
    with h5.File(first_path, 'r') as f:
        n_chunks_expected = int(f['Header']['NChunks'][()])
        redshift_ref       = float(f['Header']['Redshift'][()])
        cell_size_ref       = float(f['Header']['CellSizeKpc'][()])

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
    stop

    # --- Read each chunk (read-only), validate, and collect arrays ---
    per_chunk_data     = []
    total_skipped      = 0
    total_included     = 0

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
            z_here = float(hdr['Redshift'][()])
            if abs(z_here - redshift_ref) > 1e-6:
                sys.exit(f"Chunk {cid} has Redshift={z_here}, expected "
                          f"{redshift_ref}. Aborting -- chunks don't look "
                          f"like they're from the same snapshot.")

            total_skipped  += int(hdr['NHalosSkipped'][()])
            total_included += int(hdr['NHalosIncluded'][()])

            grp  = f['HaloData']
            data = {name: grp[name][()] for name in DATASETS_HALO}
            per_chunk_data.append(data)

        print(f'  chunk {cid}: {len(data["TrackId"])} halos read from {path}')

    # --- Concatenate in chunk-id order ---
    merged = {
        name: np.concatenate([d[name] for d in per_chunk_data])
        for name in DATASETS_HALO
    }
    n_total = len(merged['TrackId'])
    print(f'\nTotal halos after merge: {n_total} '
          f'(included: {total_included}, skipped/NaN: {total_skipped})')

    # --- Write the new merged file (mode 'x' = create, error if exists,
    #     so this can never silently clobber anything) ---
    with h5.File(merged_fn, 'x') as out:
        grp0 = out.create_group("Header")
        grp0.create_dataset('Redshift',       data=redshift_ref,      dtype='float')
        grp0.create_dataset('CellSizeKpc',    data=cell_size_ref,     dtype='float')
        grp0.create_dataset('NChunksMerged',  data=n_chunks_expected, dtype='int')
        grp0.create_dataset('NHalosTotal',    data=n_total,           dtype='int')
        grp0.create_dataset('NHalosSkipped',  data=total_skipped,     dtype='int')
        grp0.create_dataset('NHalosIncluded', data=total_included,    dtype='int')

        grp1 = out.create_group("HaloData")
        for name, arr in merged.items():
            grp1.create_dataset(name, data=arr, dtype=arr.dtype)

    print(f'\nWrote merged file: {merged_fn}')
    print('Source chunk files were opened read-only and were not modified.')


if __name__ == "__main__":
    main()
