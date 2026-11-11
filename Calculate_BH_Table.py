import numpy                        as     np
import h5py                         as     h5
import os
import sys
import argparse

# --------------------------------------------------------------------------
# Black hole properties of the selected galaxies, read from the SOAP catalogue.
#
# Writes BH_Table_<snap>.hdf5 with one row per galaxy, in the SAME order as
# Gal_Table_<snap>.hdf5 and the Stars_Mproj_Bar_* files (same Nstar selection on the same
# SOAP catalogue), so the tables can be used side by side. TrackId is written so the
# post-analysis can assert that.
#
# The SOAP file is read with h5py rather than swiftsimio: each column is read once, cut to
# the selected galaxies and dropped, and the units come from the SOAP dataset attributes.
# All outputs are physical: Msun, Msun/yr, erg, kpc.
#
#   python Calculate_BH_Table.py --snap 127
#   python Calculate_BH_Table.py --snap 127 --box L012_m6 \
#          --base /path/to/COLIBRE/ --out /path/to/Frosst_2026_Outputs/      (local test)
# --------------------------------------------------------------------------

# --- COSMA paths (defaults) ---
BasePath     = "/cosma8/data/dp004/colibre/Runs/"
OutBase      = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"
fname        = "BH_Table_"

# ---- analysis Information (must match Calculate_Gal_Table.py) ----
Nstar_min    = 5e3  # Minimum number of stellar particles
Nstar_max    = 1e10 # Maximum number of stellar particles

# ---- unit conversions from CGS ----
Msun_g       = 1.98841e33
yr_s         = 3.15576e7
kpc_cm       = 3.0856775814913673e21
CGS_TO       = {'Msun': 1./Msun_g, 'Msun/yr': yr_s/Msun_g, 'erg': 1., 'kpc': 1./kpc_cm, 'km/s': 1e-5, None: None}

# ---- which black hole columns to read: (SOAP name, output name, output unit) ----
# The same names exist in BoundSubhalo and in every ExclusiveSphere aperture.
BH_COLUMNS = [
    ('MostMassiveBlackHoleMass',                   'BHMass',                'Msun'),     # subgrid mass
    ('MostMassiveBlackHoleAccretionRate',          'BHAccretionRate',       'Msun/yr'),  # instantaneous
    ('MostMassiveBlackHoleAveragedAccretionRate',  'BHAccretionRate_avg',   'Msun/yr'),  # [:,0] 100 Myr, [:,1] 10 Myr
    ('MostMassiveBlackHoleTotalAccretedMass',      'BHTotalAccretedMass',   'Msun'),
    ('MostMassiveBlackHoleInjectedThermalEnergy',  'BHInjectedThermalEnergy', 'erg'),    # cumulative
    ('MostMassiveBlackHoleNumberOfAGNEvents',      'BHNumberOfAGNEvents',   None),
    ('MostMassiveBlackHoleLastEventScalefactor',   'BHLastEventScalefactor', None),
    ('MostMassiveBlackHoleFormationScalefactor',   'BHFormationScalefactor', None),
    ('MostMassiveBlackHoleGWMassLoss',             'BHGWMassLoss',          'Msun'),
    ('MostMassiveBlackHoleID',                     'BHID',                  None),
    ('BlackHolesSubgridMass',                      'BHSubgridMass_total',   'Msun'),     # all BHs
    ('BlackHolesDynamicalMass',                    'BHDynamicalMass_total', 'Msun'),
    ('BlackHolesTotalInjectedThermalEnergy',       'BHInjectedThermalEnergy_total', 'erg'),
    ('NumberOfBlackHoleParticles',                 'NumBHPart',             None),
]
# Only present in the hybrid (jet) AGN runs; read if they exist
BH_COLUMNS_OPTIONAL = [
    ('MostMassiveBlackHoleInjectedJetEnergy',      'BHInjectedJetEnergy',   'erg'),
    ('MostMassiveBlackHoleSpin',                   'BHSpin',                None),
    ('MostMassiveBlackHoleAccretionMode',          'BHAccretionMode',       None),
    ('MostMassiveBlackHoleNumberOfAGNJetEvents',   'BHNumberOfAGNJetEvents', None),
]
# Small apertures: is the gas-rich centre feeding / being heated by the central BH?
APERTURES = ['1kpc', '3kpc']
APERTURE_COLUMNS = ['MostMassiveBlackHoleMass', 'MostMassiveBlackHoleID', 'NumberOfBlackHoleParticles']

def read_selected(soap, path, idx, unit=None):
    # Read one SOAP column, keep the selected galaxies, convert to physical `unit`
    dset = soap[path]
    data = dset[:][idx]
    if unit is not None:
        cgs  = float(np.ravel(dset.attrs['Conversion factor to physical CGS (including cosmological corrections)'])[0])
        data = data.astype('float64') * cgs * CGS_TO[unit]
    return data

# ---- Runtime arguments ----
parser = argparse.ArgumentParser()
parser.add_argument("--snap", type=int, default=127, help="snapshot number to analyse (default: 127)")
parser.add_argument("--box",  type=str, default="L200_m6", help="box and resolution, e.g. L200_m6 (default)")
parser.add_argument("--agn",  type=str, default="THERMAL_AGN", help="AGN model prefix of the run directory")
parser.add_argument("--base", type=str, default=BasePath, help="directory holding the simulation boxes")
parser.add_argument("--out",  type=str, default=OutBase,  help="output base directory")
args   = parser.parse_args()

snap   = args.snap
Dir    = args.box + "/"
RunDir = args.agn + "_" + args.box.split("_")[-1] + "/"
ext4   = str(snap).zfill(4)
ext3   = str(snap).zfill(3)

soap_file = args.base+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"
print('SOAP file: ', soap_file)

with h5.File(soap_file, 'r') as soap:

    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (same selection as Gal_Table)
    Nstar_subhalo = soap['BoundSubhalo/NumberOfStarParticles'][:]
    lhalo         = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
    del Nstar_subhalo

    ScaleFactor   = float(np.ravel(soap['Cosmology'].attrs['Scale-factor'])[0])
    print('- snapshot        :', snap)
    print('  Number of galaxies selected: ', len(lhalo))

    out = {}
    out['TrackId']    = read_selected(soap, 'InputHalos/HBTplus/TrackId', lhalo)
    out['is_central'] = read_selected(soap, 'InputHalos/IsCentral',       lhalo)

    # --- Bound subhalo: most massive black hole and totals
    for sname, oname, unit in BH_COLUMNS:
        out[oname] = read_selected(soap, 'BoundSubhalo/'+sname, lhalo, unit)
    for sname, oname, unit in BH_COLUMNS_OPTIONAL:
        if 'BoundSubhalo/'+sname in soap:
            out[oname] = read_selected(soap, 'BoundSubhalo/'+sname, lhalo, unit)

    # --- Distance of the most massive BH from the halo centre [physical kpc]
    #     (a large offset flags a BH that is not the central one, e.g. brought in by a satellite)
    dpos = (read_selected(soap, 'BoundSubhalo/MostMassiveBlackHolePosition', lhalo, 'kpc') -
            read_selected(soap, 'InputHalos/HaloCentre',                     lhalo, 'kpc'))
    box  = float(np.ravel(soap['Header'].attrs['BoxSize'])[0]) * 1e3 * ScaleFactor   # Mpc -> physical kpc
    dpos = (dpos + 0.5*box) % box - 0.5*box                                          # periodic wrap
    out['BHOffset_kpc'] = np.linalg.norm(dpos, axis=1)

    # --- Small apertures
    for ap in APERTURES:
        for sname in APERTURE_COLUMNS:
            path = 'ExclusiveSphere/'+ap+'/'+sname
            if path in soap:
                unit = 'Msun' if sname.endswith('Mass') else None
                out[{'MostMassiveBlackHoleMass': 'BHMass', 'MostMassiveBlackHoleID': 'BHID',
                     'NumberOfBlackHoleParticles': 'NumBHPart'}[sname]+'_'+ap] = read_selected(soap, path, lhalo, unit)

    # --- Galaxies without a black hole: SOAP stores zeros; flag them explicitly
    out['hasBH'] = (out['NumBHPart'] > 0).astype('int')

    # --- Header: subgrid AGN parameters and cosmology, for Eddington ratios / lookback times
    header = {'Redshift': 1./ScaleFactor - 1.}
    params = soap['SWIFT/Parameters'].attrs
    for key, name in [('COLIBREAGN:radiative_efficiency', 'RadiativeEfficiency'),
                      ('COLIBREAGN:coupling_efficiency',  'CouplingEfficiency'),
                      ('COLIBREAGN:AGN_delta_T_K',        'AGN_delta_T_K'),
                      ('COLIBREAGN:subgrid_seed_mass_Msun', 'SeedMass_Msun')]:
        if key in params:
            val = params[key]
            header[name] = float(val.decode() if isinstance(val, bytes) else val)
    for key in ['h', 'Omega_m', 'Omega_lambda', 'Omega_b']:
        header[key] = float(np.ravel(soap['Cosmology'].attrs[key])[0])

# --- Write to hdf5
fn = args.out+Dir+RunDir+fname+ext3+".hdf5"
print('\n Writing to:', fn)
os.makedirs(os.path.dirname(fn), exist_ok=True)

with h5.File(fn, "w") as output:
    grp0 = output.create_group("Header")
    grp1 = output.create_group("HaloData")
    for key, val in header.items():
        grp0.create_dataset(key, data=val, dtype='float')
    for key, val in out.items():
        isint = np.issubdtype(np.asarray(val).dtype, np.integer)
        grp1.create_dataset(key, data=val, dtype='int' if isint else 'float')

print('  has a black hole: %d / %d' % (out['hasBH'].sum(), len(lhalo)))
sys.exit() ###################################################################
