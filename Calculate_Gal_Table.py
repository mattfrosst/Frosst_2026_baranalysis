from   Frosst_2026_fourieranalysis  import *
from   periodic_kdtree              import PeriodicCKDTree
from   colibre_utility              import *
import numpy                        as     np
import scipy                        as     scipy
import unyt                         as     unyt
import h5py                         as     h5
import swiftsimio                   as     sw
import scipy.optimize               as     sp_opt
import os
import sys
import time 
import itertools
import warnings
import matplotlib.pylab             as plt
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# --------------------------------------------------------------------------
# ---- Simulation information ----
#Fiducial_test

# --- Local test path ---
BasePath     = "/Users/23229092/Documents/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

# --- COSMA paths ---
#BasePath     = "/cosma8/data/dp004/colibre/Runs/"   ; SnapBase = "colibre_"
#BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L050_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 123
#BoxDir       = ["L050_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127
#BoxDir       = ["L100_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L100_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/"  ; snap = 127
#BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 93#95#99#101#103#105#107#109#111#113#115#117#119#121#123#125#58#62#66#72#80#88#97#112#56#60#64#68#76#84#92#102#127
#BoxDir       = ["L200_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/"  ; snap = 127

#BoxDir       = ["L025_m5/"]                         ; RunDir   = "THERMAL_AGN_m5/" ; snap = 127
#BoxDir       = ["L025_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L025_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "Gal_Table__"

# ---- analysis Information ----
Nstar_min    = 5e3  # Minimum number of stellar particles
Nstar_max    = 1e10 # Maximum number of stellar particles

for     idir,  Dir  in enumerate(BoxDir):

    ext4         = str(snap).zfill(4)
    ext3         = str(snap).zfill(3)

    Swiftfile    = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"

    soap_file    = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"
            
    # --- Read SOAP halo data 
    print('SOAP file: ', soap_file)
    soap         = sw.load(soap_file)

    # --- Read the selection data (Stellar and DM resolution)
    Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
            
    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (Using SOAP catalogs)
    lhalo        = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]

    print('- snapshot        :',snap)
    print('  Number of galaxies selected: ',len(lhalo)) 

    ScaleFactor  = soap.metadata.scale_factor
    
    # --- Get some properties of the galaxies from SOAP
    # --- Bound subhalo: galaxy properties
    rhalf_stars       = soap.bound_subhalo.half_mass_radius_stars[lhalo]
    rhalf_gas         = soap.bound_subhalo.half_mass_radius_gas[lhalo]
    nstar             = soap.bound_subhalo.number_of_star_particles[lhalo]
    ngas              = soap.bound_subhalo.number_of_gas_particles[lhalo]
    mean_stellar_age  = soap.bound_subhalo.mass_weighted_mean_stellar_age[lhalo]
    kappa_co_allstars = soap.bound_subhalo.kappa_corot_stars[lhalo]

    # --- Exclusive sphere 50kpc: galaxy properties
    mstar             = soap.exclusive_sphere_50kpc.stellar_mass[lhalo]
    mgas              = soap.exclusive_sphere_50kpc.gas_mass[lhalo]
    sf_mgas           = soap.exclusive_sphere_50kpc.star_forming_gas_mass[lhalo]
    mHI               = soap.exclusive_sphere_50kpc.atomic_hydrogen_mass[lhalo]
    mH2               = soap.exclusive_sphere_50kpc.molecular_hydrogen_mass[lhalo]
    DT                = soap.exclusive_sphere_50kpc.disc_to_total_stellar_mass_fraction[lhalo]
    SFR               = soap.exclusive_sphere_50kpc.star_formation_rate[lhalo]
    kappa_co_stars    = soap.exclusive_sphere_50kpc.kappa_corot_stars[lhalo]

    # -- luminosities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_g           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,1]
    Lstar_r           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,2]
    Lstar_z           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,4]
    Lstar_Y           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,5]
    
    # -- Inclusive sphere 50kpc: angular momentum
    angJ_stars        = soap.inclusive_sphere_50kpc.angular_momentum_stars[lhalo]
    angJ_gas          = soap.inclusive_sphere_50kpc.angular_momentum_gas[lhalo]
    angJ_baryons      = soap.inclusive_sphere_50kpc.angular_momentum_baryons[lhalo]

    # --- Exclusive sphere 30kpc: galaxy properties
    mstar_30kpc             = soap.exclusive_sphere_30kpc.stellar_mass[lhalo]
    mgas_30kpc              = soap.exclusive_sphere_30kpc.gas_mass[lhalo]
    sf_mgas_30kpc           = soap.exclusive_sphere_30kpc.star_forming_gas_mass[lhalo]
    mHI_30kpc               = soap.exclusive_sphere_30kpc.atomic_hydrogen_mass[lhalo]
    mH2_30kpc               = soap.exclusive_sphere_30kpc.molecular_hydrogen_mass[lhalo]
    DT_30kpc                = soap.exclusive_sphere_30kpc.disc_to_total_stellar_mass_fraction[lhalo]
    SFR_30kpc               = soap.exclusive_sphere_30kpc.star_formation_rate[lhalo]
    kappa_co_stars_30kpc    = soap.exclusive_sphere_30kpc.kappa_corot_stars[lhalo]

    # -- luminosities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_g_30kpc           = soap.inclusive_sphere_30kpc.stellar_luminosity[lhalo,1]
    Lstar_r_30kpc           = soap.inclusive_sphere_30kpc.stellar_luminosity[lhalo,2]
    Lstar_z_30kpc           = soap.inclusive_sphere_30kpc.stellar_luminosity[lhalo,4]
    Lstar_Y_30kpc           = soap.inclusive_sphere_30kpc.stellar_luminosity[lhalo,5]

    # -- Inclusive sphere 30kpc: angular momentum
    angJ_stars_30kpc        = soap.inclusive_sphere_30kpc.angular_momentum_stars[lhalo]
    angJ_gas_30kpc          = soap.inclusive_sphere_30kpc.angular_momentum_gas[lhalo]
    angJ_baryons_30kpc      = soap.inclusive_sphere_30kpc.angular_momentum_baryons[lhalo]
    
    # --- Exclusive sphere 10kpc: galaxy properties
    mstar_10kpc             = soap.exclusive_sphere_10kpc.stellar_mass[lhalo]
    mgas_10kpc              = soap.exclusive_sphere_10kpc.gas_mass[lhalo]
    sf_mgas_10kpc           = soap.exclusive_sphere_10kpc.star_forming_gas_mass[lhalo]
    mHI_10kpc               = soap.exclusive_sphere_10kpc.atomic_hydrogen_mass[lhalo]
    mH2_10kpc               = soap.exclusive_sphere_10kpc.molecular_hydrogen_mass[lhalo]
    DT_10kpc                = soap.exclusive_sphere_10kpc.disc_to_total_stellar_mass_fraction[lhalo]
    SFR_10kpc               = soap.exclusive_sphere_10kpc.star_formation_rate[lhalo]
    kappa_co_stars_10kpc    = soap.exclusive_sphere_10kpc.kappa_corot_stars[lhalo]

    # -- luminosities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_g_10kpc           = soap.inclusive_sphere_10kpc.stellar_luminosity[lhalo,1]
    Lstar_r_10kpc           = soap.inclusive_sphere_10kpc.stellar_luminosity[lhalo,2]
    Lstar_z_10kpc           = soap.inclusive_sphere_10kpc.stellar_luminosity[lhalo,4]
    Lstar_Y_10kpc           = soap.inclusive_sphere_10kpc.stellar_luminosity[lhalo,5]

    # -- Inclusive sphere 30kpc: angular momentum
    angJ_stars_10kpc        = soap.inclusive_sphere_10kpc.angular_momentum_stars[lhalo]
    angJ_gas_10kpc          = soap.inclusive_sphere_10kpc.angular_momentum_gas[lhalo]
    angJ_baryons_10kpc      = soap.inclusive_sphere_10kpc.angular_momentum_baryons[lhalo]
    
    # --- Exclusive sphere 3kpc: galaxy properties
    mstar_3kpc             = soap.exclusive_sphere_3kpc.stellar_mass[lhalo]
    mgas_3kpc              = soap.exclusive_sphere_3kpc.gas_mass[lhalo]
    sf_mgas_3kpc           = soap.exclusive_sphere_3kpc.star_forming_gas_mass[lhalo]
    mHI_3kpc               = soap.exclusive_sphere_3kpc.atomic_hydrogen_mass[lhalo]
    mH2_3kpc               = soap.exclusive_sphere_3kpc.molecular_hydrogen_mass[lhalo]
    DT_3kpc                = soap.exclusive_sphere_3kpc.disc_to_total_stellar_mass_fraction[lhalo]
    SFR_3kpc               = soap.exclusive_sphere_3kpc.star_formation_rate[lhalo]
    kappa_co_stars_3kpc    = soap.exclusive_sphere_3kpc.kappa_corot_stars[lhalo]

    # -- luminosities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_g_3kpc           = soap.inclusive_sphere_3kpc.stellar_luminosity[lhalo,1]
    Lstar_r_3kpc           = soap.inclusive_sphere_3kpc.stellar_luminosity[lhalo,2]
    Lstar_z_3kpc           = soap.inclusive_sphere_3kpc.stellar_luminosity[lhalo,4]
    Lstar_Y_3kpc           = soap.inclusive_sphere_3kpc.stellar_luminosity[lhalo,5]

    # -- Inclusive sphere 30kpc: angular momentum
    angJ_stars_3kpc        = soap.inclusive_sphere_3kpc.angular_momentum_stars[lhalo]
    angJ_gas_3kpc          = soap.inclusive_sphere_3kpc.angular_momentum_gas[lhalo]
    angJ_baryons_3kpc      = soap.inclusive_sphere_3kpc.angular_momentum_baryons[lhalo]
    
    # --- In spherical r200crit: halo properties
    N200              = soap.spherical_overdensity_200_crit.number_of_dark_matter_particles[lhalo]
    M200              = soap.spherical_overdensity_200_crit.total_mass[lhalo]
    r200              = soap.spherical_overdensity_200_crit.soradius[lhalo]
    angJ_DM           = soap.spherical_overdensity_200_crit.angular_momentum_dark_matter[lhalo]
    velCOM_stars      = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[lhalo]
    fsub              = soap.spherical_overdensity_200_crit.mass_fraction_satellites[lhalo]
    concentration     = soap.spherical_overdensity_200_crit.concentration[lhalo]
    spin              = soap.spherical_overdensity_200_crit.spin_parameter[lhalo]

    # --- HBT halo properties
    is_central        = soap.input_halos.is_central[lhalo]
    halo_centre       = soap.input_halos.halo_centre[lhalo]
    TrackId           = soap.input_halos_hbtplus.track_id[lhalo]
    
    # --- convert Units
    rhalf_stars.convert_to_units('kpc')            ; rhalf_stars.convert_to_physical()
    rhalf_gas.convert_to_units('kpc')              ; rhalf_gas.convert_to_physical()
    mean_stellar_age.convert_to_units('Gyr')       ; mean_stellar_age.convert_to_physical()

    M200.convert_to_units('Msun')                  ; M200.convert_to_physical()
    r200.convert_to_units('kpc')                   ; r200.convert_to_physical()
    angJ_DM.convert_to_units('Msun*kpc*km/s')      ; angJ_DM.convert_to_physical()
    velCOM_stars.convert_to_units('km/s')          ; velCOM_stars.convert_to_physical()
    halo_centre.convert_to_units('kpc')            ; halo_centre.convert_to_physical()

    # -- 50kpc --
    mstar.convert_to_units('Msun')                 ; mstar.convert_to_physical()
    mgas.convert_to_units('Msun')                  ; mgas.convert_to_physical()
    sf_mgas.convert_to_units('Msun')               ; sf_mgas.convert_to_physical()
    mHI.convert_to_units('Msun')                   ; mHI.convert_to_physical()
    mH2.convert_to_units('Msun')                   ; mH2.convert_to_physical()
    SFR.convert_to_units('Msun/yr')                ; SFR.convert_to_physical()

    angJ_stars.convert_to_units('Msun*kpc*km/s')   ; angJ_stars.convert_to_physical()
    angJ_gas.convert_to_units('Msun*kpc*km/s')     ; angJ_gas.convert_to_physical()
    angJ_baryons.convert_to_units('Msun*kpc*km/s') ; angJ_baryons.convert_to_physical()

    # -- 30kpc --
    mstar_30kpc.convert_to_units('Msun')           ; mstar_30kpc.convert_to_physical()
    mgas_30kpc.convert_to_units('Msun')            ; mgas_30kpc.convert_to_physical()
    sf_mgas_30kpc.convert_to_units('Msun')         ; sf_mgas_30kpc.convert_to_physical()
    mHI_30kpc.convert_to_units('Msun')             ; mHI_30kpc.convert_to_physical()
    mH2_30kpc.convert_to_units('Msun')             ; mH2_30kpc.convert_to_physical()
    SFR_30kpc.convert_to_units('Msun/yr')          ; SFR_30kpc.convert_to_physical()

    angJ_stars_30kpc.convert_to_units('Msun*kpc*km/s')   ; angJ_stars_30kpc.convert_to_physical()
    angJ_gas_30kpc.convert_to_units('Msun*kpc*km/s')     ; angJ_gas_30kpc.convert_to_physical()
    angJ_baryons_30kpc.convert_to_units('Msun*kpc*km/s') ; angJ_baryons_30kpc.convert_to_physical()

    # -- 10kpc --
    mstar_10kpc.convert_to_units('Msun')           ; mstar_10kpc.convert_to_physical()
    mgas_10kpc.convert_to_units('Msun')            ; mgas_10kpc.convert_to_physical()
    sf_mgas_10kpc.convert_to_units('Msun')         ; sf_mgas_10kpc.convert_to_physical()
    mHI_10kpc.convert_to_units('Msun')             ; mHI_10kpc.convert_to_physical()
    mH2_10kpc.convert_to_units('Msun')             ; mH2_10kpc.convert_to_physical()
    SFR_10kpc.convert_to_units('Msun/yr')          ; SFR_10kpc.convert_to_physical()

    angJ_stars_10kpc.convert_to_units('Msun*kpc*km/s')   ; angJ_stars_10kpc.convert_to_physical()
    angJ_gas_10kpc.convert_to_units('Msun*kpc*km/s')     ; angJ_gas_10kpc.convert_to_physical()
    angJ_baryons_10kpc.convert_to_units('Msun*kpc*km/s') ; angJ_baryons_10kpc.convert_to_physical()

    # -- 3kpc --
    mstar_3kpc.convert_to_units('Msun')           ; mstar_3kpc.convert_to_physical()
    mgas_3kpc.convert_to_units('Msun')            ; mgas_3kpc.convert_to_physical()
    sf_mgas_3kpc.convert_to_units('Msun')         ; sf_mgas_3kpc.convert_to_physical()
    mHI_3kpc.convert_to_units('Msun')             ; mHI_3kpc.convert_to_physical()
    mH2_3kpc.convert_to_units('Msun')             ; mH2_3kpc.convert_to_physical()
    SFR_3kpc.convert_to_units('Msun/yr')          ; SFR_3kpc.convert_to_physical()

    angJ_stars_3kpc.convert_to_units('Msun*kpc*km/s')   ; angJ_stars_3kpc.convert_to_physical()
    angJ_gas_3kpc.convert_to_units('Msun*kpc*km/s')     ; angJ_gas_3kpc.convert_to_physical()
    angJ_baryons_3kpc.convert_to_units('Msun*kpc*km/s') ; angJ_baryons_3kpc.convert_to_physical()
    

    # --- get the total angular momentum
    angtot_stars   = np.linalg.norm(angJ_stars,  axis=1)
    angtot_baryons = np.linalg.norm(angJ_baryons,axis=1)
    angtot_gas     = np.linalg.norm(angJ_gas,    axis=1)
    angtot_DM      = np.linalg.norm(angJ_DM,     axis=1)

    angtot_stars_30kpc   = np.linalg.norm(angJ_stars_30kpc,  axis=1)
    angtot_baryons_30kpc = np.linalg.norm(angJ_baryons_30kpc,axis=1)
    angtot_gas_30kpc     = np.linalg.norm(angJ_gas_30kpc,    axis=1)

    angtot_stars_10kpc   = np.linalg.norm(angJ_stars_10kpc,  axis=1)
    angtot_baryons_10kpc = np.linalg.norm(angJ_baryons_10kpc,axis=1)
    angtot_gas_10kpc     = np.linalg.norm(angJ_gas_10kpc,    axis=1)
    
    angtot_stars_3kpc   = np.linalg.norm(angJ_stars_3kpc,  axis=1)
    angtot_baryons_3kpc = np.linalg.norm(angJ_baryons_3kpc,axis=1)
    angtot_gas_3kpc     = np.linalg.norm(angJ_gas_3kpc,    axis=1)
    
    # --- Write to hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"                  #Local path
    #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+fname+ext3+".hdf5" #COSMA path
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w")
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData_50kpc")
    grp2    = output.create_group("HaloData_30kpc")
    grp3    = output.create_group("HaloData_10kpc")
    grp4    = output.create_group("HaloData_3kpc")

    dset    = grp0.create_dataset('Redshift',       data = 1./ScaleFactor - 1,     dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,                dtype = 'int')
    dset    = grp1.create_dataset('is_central',     data = is_central,             dtype = 'int')
    dset    = grp1.create_dataset('NumStellarPart', data = nstar,                  dtype = 'int')
    dset    = grp1.create_dataset('NumGasPart',     data = ngas,                   dtype = 'int')
    dset    = grp1.create_dataset('N200_DM',        data = N200,                   dtype = 'int')
    dset    = grp1.create_dataset('M200',           data = M200,                   dtype = 'float')
    Mset    = grp1.create_dataset('r200',           data = r200,                   dtype = 'float')
    dset    = grp1.create_dataset('concentration',  data = concentration,          dtype = 'float')
    dset    = grp1.create_dataset('SpinParameter',  data = spin,                   dtype = 'float')
    dset    = grp1.create_dataset('MeanStellarAge', data = mean_stellar_age,       dtype = 'float')
    dset    = grp1.create_dataset('fsub200',        data = fsub,                   dtype = 'float')
    dset    = grp1.create_dataset('r50_stars',      data = rhalf_stars,            dtype = 'float')
    dset    = grp1.create_dataset('r50_gas',        data = rhalf_gas,              dtype = 'float')
    dset    = grp1.create_dataset('KappaCorotAllStars', data = kappa_co_allstars,   dtype = 'float')

    # --- ES within 50kpc ---
    dset    = grp1.create_dataset('StellarMass',    data = mstar,                  dtype = 'float')
    dset    = grp1.create_dataset('GasMass',        data = mgas,                   dtype = 'float')
    dset    = grp1.create_dataset('StarFormingGasMass', data = sf_mgas,            dtype = 'float')
    dset    = grp1.create_dataset('H2Mass',         data = mH2,                    dtype = 'float')
    dset    = grp1.create_dataset('HIMass',         data = mHI,                    dtype = 'float')
    dset    = grp1.create_dataset('DT',             data = DT,                     dtype = 'float')
    dset    = grp1.create_dataset('SFR_50kpc',      data = SFR,                    dtype = 'float')
    dset    = grp1.create_dataset('KappaCorotStars',data = kappa_co_stars,         dtype = 'float')
    dset    = grp1.create_dataset('Lstar_gband',    data = Lstar_g,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_rband',    data = Lstar_r,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_zband',    data = Lstar_z,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_Yband',    data = Lstar_Y,                dtype = 'float')
    dset    = grp1.create_dataset('angJ_stars',     data = angJ_stars,             dtype = 'float')
    dset    = grp1.create_dataset('angJ_gas',       data = angJ_gas,               dtype = 'float')
    dset    = grp1.create_dataset('angJ_baryons',   data = angJ_baryons,           dtype = 'float')
    dset    = grp1.create_dataset('angJ_DM',        data = angJ_DM,                dtype = 'float')
    dset    = grp1.create_dataset('angtot_stars',   data = angtot_stars,           dtype = 'float')
    dset    = grp1.create_dataset('angtot_gas',     data = angtot_gas,             dtype = 'float')
    dset    = grp1.create_dataset('angtot_baryons', data = angtot_baryons,         dtype = 'float')
    dset    = grp1.create_dataset('angtot_DM',      data = angtot_DM,              dtype = 'float')
    
    # --- ES within 30kpc ---
    dset    = grp2.create_dataset('StellarMass',    data = mstar_30kpc,            dtype = 'float')
    dset    = grp2.create_dataset('GasMass',        data = mgas_30kpc,             dtype = 'float')
    dset    = grp2.create_dataset('StarFormingGasMass', data = sf_mgas_30kpc,      dtype = 'float')
    dset    = grp2.create_dataset('H2Mass',         data = mH2_30kpc,              dtype = 'float')
    dset    = grp2.create_dataset('HIMass',         data = mHI_30kpc,              dtype = 'float')
    dset    = grp2.create_dataset('DT',             data = DT_30kpc,               dtype = 'float')
    dset    = grp2.create_dataset('SFR',            data = SFR_30kpc,              dtype = 'float')
    dset    = grp2.create_dataset('KappaCorotStars',data = kappa_co_stars_30kpc,   dtype = 'float')
    dset    = grp2.create_dataset('Lstar_gband',    data = Lstar_g_30kpc,          dtype = 'float')
    dset    = grp2.create_dataset('Lstar_rband',    data = Lstar_r_30kpc,          dtype = 'float')
    dset    = grp2.create_dataset('Lstar_zband',    data = Lstar_z_30kpc,          dtype = 'float')
    dset    = grp2.create_dataset('Lstar_Yband',    data = Lstar_Y_30kpc,          dtype = 'float')
    dset    = grp2.create_dataset('angJ_stars',     data = angJ_stars_30kpc,       dtype = 'float')
    dset    = grp2.create_dataset('angJ_gas',       data = angJ_gas_30kpc,         dtype = 'float')
    dset    = grp2.create_dataset('angJ_baryons',   data = angJ_baryons_30kpc,     dtype = 'float')
    dset    = grp2.create_dataset('angtot_stars',   data = angtot_stars_30kpc,     dtype = 'float')
    dset    = grp2.create_dataset('angtot_gas',     data = angtot_gas_30kpc,       dtype = 'float')
    dset    = grp2.create_dataset('angtot_baryons', data = angtot_baryons_30kpc,   dtype = 'float')
    
    # --- ES within 10kpc ---
    dset    = grp3.create_dataset('StellarMass',    data = mstar_10kpc,            dtype = 'float')
    dset    = grp3.create_dataset('GasMass',        data = mgas_10kpc,             dtype = 'float')
    dset    = grp3.create_dataset('StarFormingGasMass', data = sf_mgas_10kpc,      dtype = 'float')
    dset    = grp3.create_dataset('H2Mass',         data = mH2_10kpc,              dtype = 'float')
    dset    = grp3.create_dataset('HIMass',         data = mHI_10kpc,              dtype = 'float')
    dset    = grp3.create_dataset('DT',             data = DT_10kpc,               dtype = 'float')
    dset    = grp3.create_dataset('SFR',            data = SFR_10kpc,              dtype = 'float')
    dset    = grp3.create_dataset('KappaCorotStars',data = kappa_co_stars_10kpc,   dtype = 'float')
    dset    = grp3.create_dataset('Lstar_gband',    data = Lstar_g_10kpc,          dtype = 'float')
    dset    = grp3.create_dataset('Lstar_rband',    data = Lstar_r_10kpc,          dtype = 'float')
    dset    = grp3.create_dataset('Lstar_zband',    data = Lstar_z_10kpc,          dtype = 'float')
    dset    = grp3.create_dataset('Lstar_Yband',    data = Lstar_Y_10kpc,          dtype = 'float')
    dset    = grp3.create_dataset('angJ_stars',     data = angJ_stars_10kpc,       dtype = 'float')
    dset    = grp3.create_dataset('angJ_gas',       data = angJ_gas_10kpc,         dtype = 'float')
    dset    = grp3.create_dataset('angJ_baryons',   data = angJ_baryons_10kpc,     dtype = 'float')
    dset    = grp3.create_dataset('angtot_stars',   data = angtot_stars_10kpc,     dtype = 'float')
    dset    = grp3.create_dataset('angtot_gas',     data = angtot_gas_10kpc,       dtype = 'float')
    dset    = grp3.create_dataset('angtot_baryons', data = angtot_baryons_10kpc,   dtype = 'float')

    # --- ES within 3kpc ---
    dset    = grp4.create_dataset('StellarMass',    data = mstar_3kpc,            dtype = 'float')
    dset    = grp4.create_dataset('GasMass',        data = mgas_3kpc,             dtype = 'float')
    dset    = grp4.create_dataset('StarFormingGasMass', data = sf_mgas_3kpc,      dtype = 'float')
    dset    = grp4.create_dataset('H2Mass',         data = mH2_3kpc,              dtype = 'float')
    dset    = grp4.create_dataset('HIMass',         data = mHI_3kpc,              dtype = 'float')
    dset    = grp4.create_dataset('DT',             data = DT_3kpc,               dtype = 'float')
    dset    = grp4.create_dataset('SFR',            data = SFR_3kpc,              dtype = 'float')
    dset    = grp4.create_dataset('KappaCorotStars',data = kappa_co_stars_3kpc,   dtype = 'float')
    dset    = grp4.create_dataset('Lstar_gband',    data = Lstar_g_3kpc,          dtype = 'float')
    dset    = grp4.create_dataset('Lstar_rband',    data = Lstar_r_3kpc,          dtype = 'float')
    dset    = grp4.create_dataset('Lstar_zband',    data = Lstar_z_3kpc,          dtype = 'float')
    dset    = grp4.create_dataset('Lstar_Yband',    data = Lstar_Y_3kpc,          dtype = 'float')
    dset    = grp4.create_dataset('angJ_stars',     data = angJ_stars_3kpc,       dtype = 'float')
    dset    = grp4.create_dataset('angJ_gas',       data = angJ_gas_3kpc,         dtype = 'float')
    dset    = grp4.create_dataset('angJ_baryons',   data = angJ_baryons_3kpc,     dtype = 'float')
    dset    = grp4.create_dataset('angtot_stars',   data = angtot_stars_3kpc,     dtype = 'float')
    dset    = grp4.create_dataset('angtot_gas',     data = angtot_gas_3kpc,       dtype = 'float')
    dset    = grp4.create_dataset('angtot_baryons', data = angtot_baryons_3kpc,   dtype = 'float')
    
    output.close()


plt.show()    
sys.exit() ###################################################################

