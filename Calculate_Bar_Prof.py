from   Frosst_2026_fourieranalysis  import *
from   periodic_kdtree              import PeriodicCKDTree
from   bar_profiles                 import *
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
BasePath     = "/Users/23229092/Documents/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L0200N3008/"]                       ; RunDir   = "THERMAL_AGN_m6/"          ; snap = 127
#BoxDir       = ["L0400N3008/"]                       ; RunDir   = "THERMAL_AGN_m7/"          ; snap = 127
#BoxDir       = ["L0100N1504/"]                       ; RunDir   = "Thermal_non_equilibrium/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "Stars_Mproj_Bar_Prof_"

# ---- analysis Information ----
Nstar_min    = 5e3  # Minimum number of stellar particles
Nstar_max    = 1e10 # Maximum number of stellar particles

for     idir,  Dir  in enumerate(BoxDir):

    ext4         = str(snap).zfill(4)
    ext3         = str(snap).zfill(3)

    Swiftfile    = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"

    soap_file    = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"
    #soap_file    = BasePath+Dir+RunDir+"SOAP/halo_properties_"+ext4+".hdf5"
        
    # --- Read SOAP halo data 
    print('SOAP file: ', soap_file)
    soap         = sw.load(soap_file)

    # --- Read the selection data (Stellar and DM resolution)
    #NDM_subhalo   = soap.bound_subhalo.number_of_dark_matter_particles
    Nstar_subhalo = soap.bound_subhalo.number_of_star_particles
            
    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (Using SOAP catalogs)
    lhalo        = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]

    print('- snapshot        :',snap)
    print('  Number of galaxies selected: ',len(lhalo)) 
    
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
    mHI               = soap.exclusive_sphere_50kpc.atomic_hydrogen_mass[lhalo]
    mH2               = soap.exclusive_sphere_50kpc.molecular_hydrogen_mass[lhalo]
    DT                = soap.exclusive_sphere_50kpc.disc_to_total_stellar_mass_fraction[lhalo]
    SFR               = soap.exclusive_sphere_50kpc.star_formation_rate[lhalo]

    # --- In spherical r200crit: halo properties
    N200              = soap.spherical_overdensity_200_crit.number_of_dark_matter_particles[lhalo]
    M200              = soap.spherical_overdensity_200_crit.total_mass[lhalo]
    r200              = soap.spherical_overdensity_200_crit.soradius[lhalo]
    angJ_DM           = soap.spherical_overdensity_200_crit.angular_momentum_dark_matter[lhalo]
    velCOM_stars      = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[lhalo]
    fsub              = soap.spherical_overdensity_200_crit.mass_fraction_satellites[lhalo]

    # -- Inclusive sphere 50kpc: angular momentum
    angJ_stars        = soap.inclusive_sphere_50kpc.angular_momentum_stars[lhalo]
    angJ_gas          = soap.inclusive_sphere_50kpc.angular_momentum_gas[lhalo]
    angJ_baryons      = soap.inclusive_sphere_50kpc.angular_momentum_baryons[lhalo]

    # --- HBT halo properties
    is_central        = soap.input_halos.is_central[lhalo]
    halo_centre       = soap.input_halos.halo_centre[lhalo]
    TrackId           = soap.input_halos_hbtplus.track_id[lhalo]
    
    # -- luminosities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_r           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,2]
    Lstar_z           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,4]
    Lstar_Y           = soap.inclusive_sphere_50kpc.stellar_luminosity[lhalo,5]

    # --- convert Units
    velCOM_stars.convert_to_units('km/s')          ; velCOM_stars.convert_to_physical()
    rhalf_stars.convert_to_units('kpc')            ; rhalf_stars.convert_to_physical()
    r200.convert_to_units('kpc')                   ; r200.convert_to_physical()
    halo_centre.convert_to_units('kpc')            ; halo_centre.convert_to_physical()
    rhalf_gas.convert_to_units('kpc')              ; rhalf_gas.convert_to_physical()
    mstar.convert_to_units('Msun')                 ; mstar.convert_to_physical()
    mgas.convert_to_units('Msun')                  ; mgas.convert_to_physical()
    mHI.convert_to_units('Msun')                   ; mH2.convert_to_physical()
    mean_stellar_age.convert_to_units('Gyr')       ; mean_stellar_age.convert_to_physical()
    M200.convert_to_units('Msun')                  ; M200.convert_to_physical()    
    angJ_stars.convert_to_units('Msun*kpc*km/s')   ; angJ_stars.convert_to_physical()
    angJ_gas.convert_to_units('Msun*kpc*km/s')     ; angJ_gas.convert_to_physical()
    angJ_baryons.convert_to_units('Msun*kpc*km/s') ; angJ_baryons.convert_to_physical()
    angJ_DM.convert_to_units('Msun*kpc*km/s')      ; angJ_DM.convert_to_physical()

    # --- get the total angular momentum
    angtot_stars   = np.linalg.norm(angJ_stars,  axis=1)
    angtot_baryons = np.linalg.norm(angJ_baryons,axis=1)
    angtot_gas     = np.linalg.norm(angJ_gas,    axis=1)
    angtot_DM      = np.linalg.norm(angJ_DM,     axis=1)
    
    ang_spec_stars   = angtot_stars   / mstar
    ang_spec_gas     = angtot_gas     / mgas
    ang_spec_baryons = angtot_baryons / (mgas+mstar)
    ang_spec_DM      = angtot_DM      / M200

    # --- define arrays for output; n.b. we will normalise the positions by the half mass radius
    Nprof          = 41
    x_range        = [-1,1] # i.e., bins from 0.01Rhalf to 10Rhalf
    xbin           = np.linspace(x_range[0],x_range[1],Nprof)
    dx             = xbin[1] - xbin[0] # Bins are equally spaced in x
    xbin_linear    = np.append(0,10**xbin[:])
    print('xbin:', xbin, 'dx: ', dx, 'xbin_linear: ', xbin_linear)

    # --- number of stars per bin (nB), the number of stars analyised, and the surf mass density
    nB_stars       = np.zeros((len(lhalo),Nprof));
    nstar_count    = np.zeros(len(lhalo),dtype=int);
    Sd0_prof_stars = np.zeros((len(lhalo),Nprof));

    # --- mass weighted bin edges (left, right -> R0, R1) and middle (Rm)
    R0_prof_stars = np.zeros((len(lhalo),Nprof));
    Rm_prof_stars = np.zeros((len(lhalo),Nprof));
    R1_prof_stars = np.zeros((len(lhalo),Nprof));

    # --- mass weighted 2nd fourier amplitude and error
    A2_prof_stars    = np.zeros((len(lhalo),Nprof));
    A2err_prof_stars = np.zeros((len(lhalo),Nprof));

    # --- mass weighted 2nd fourier position angle and error
    Phi2_prof_stars    = np.zeros((len(lhalo),Nprof));
    Phi2err_prof_stars = np.zeros((len(lhalo),Nprof));

    # --- How many files do we need to look at?
    total_files  = 0 
    for root, _, filenames in os.walk(BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/"):
        total_files += len(filenames) - 1
    print('total files:', total_files)
    
    for ifile in range(total_files):
        Swiftfile    = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+'.'+str(ifile)+".hdf5"

        print('SWIFT file:',Swiftfile)
        data         = sw.load(Swiftfile)
        meta_data    = data.metadata
        ScaleFactor  = meta_data.scale_factor
        boxsize      = meta_data.boxsize * ScaleFactor
        cosmic_time  = meta_data.time
        cosmic_time.convert_to_units('yr')
        boxsize.convert_to_units('kpc')
    
        # --- read the relevant stellar particle data
        pos_stars     = data.stars.coordinates           ; print(' ...read stellar coordinates')
        mass_stars    = data.stars.masses                ; print(' ...read stellar masses')
        vel_stars     = data.stars.velocities            ; print(' ...read stellar velocities')
        lum_stars_z   = data.stars.luminosities.GAMA_z   ; print(' ...read GAMA z-band Luminosities')
        lum_stars_Y   = data.stars.luminosities.GAMA_Y   ; print(' ...read GAMA Y-band Luminosities')
        if DoBound:
            IDs_stars = data.stars.particle_ids          ; print(' ...read stellar IDs')
            
        pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()                                                                                 
        mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()
        vel_stars.convert_to_units('km/s') ; vel_stars.convert_to_physical()
        
        # Build relevant KDE trees
        print(' \n Building particle tree...')
        star_tree = PeriodicCKDTree(boxsize, pos_stars, leafsize=100) ; print(' Star particle tree done... \n')    
    
        # --- Calculate the various sizes for each galaxy of interest
        for ihalo, lh in enumerate(lhalo):

            # -------------------------------------------------
            # --- Compute Bar Profiles for KDTree particles ---
            # -------------------------------------------------
            Rcut        = 50.
            lstar_tree  = np.asarray(star_tree.query_ball_point(halo_centre[ihalo,:], Rcut))
                            
            if len(lstar_tree) > 0:
                nstar_count[ihalo] = +len(lstar_tree)
            
                pos_tree   = pos_stars[lstar_tree,:].value - halo_centre[ihalo,:].value
                vel_tree   = vel_stars[lstar_tree,:].value - velCOM_stars[ihalo,:].value

                # --- align galaxy with z-component of AM within 50kpc
                trans = rotation_matrix_from_vectors(angJ_stars[ihalo,:], [0,0,1])   # Transformation Matrix
                pos_tree = (trans @ pos_tree.T).T / rhalf_stars[ihalo].value         # One BLAS call, then norm by stellar half mass radius
                vel_tree = (trans @ vel_tree.T).T / rhalf_stars[ihalo].value         # One BLAS call, then norm by stellar half mass radius

                bar_tool  = FourierMethodFast(mass_stars[lstar_tree], pos_tree[:,0], pos_tree[:,1], vel_tree[:,0], vel_tree[:,1])
                binData   = bar_tool.analyseBins(xbin_linear)
                
                nB_stars[ihalo, :]           = binData[:,0]
                R0_prof_stars[ihalo, :]      = binData[:,1]
                Rm_prof_stars[ihalo, :]      = binData[:,2]
                R1_prof_stars[ihalo, :]      = binData[:,3]
                Sd0_prof_stars[ihalo, :]     = binData[:,4]
                A2_prof_stars[ihalo, :]      = binData[:,5]
                A2err_prof_stars[ihalo, :]   = binData[:,6]
                Phi2_prof_stars[ihalo, :]    = binData[:,7]
                Phi2err_prof_stars[ihalo, :] = binData[:,8]

        fracs     = round(float(ihalo+1)/len(lhalo),4)
        if ihalo % 100  ==0:
            print(' Group:',lh,' | f:',fracs)
        elif ihalo   ==len(lhalo)-1:
            print(' Group:',lh,' | f:',fracs)
        
    # --- Write to hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w")
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")
    grp2    = output.create_group("Profiles")

    dset    = grp0.create_dataset('Redshift',       data = 1./ScaleFactor - 1,     dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,                dtype = 'int')
    dset    = grp1.create_dataset('is_central',     data = is_central,             dtype = 'int')
    dset    = grp1.create_dataset('NumStellarPart', data = nstar,                  dtype = 'int')
    dset    = grp1.create_dataset('NumGasPart',     data = ngas,                   dtype = 'int')
    dset    = grp1.create_dataset('N200_DM',        data = N200,                   dtype = 'int')
    dset    = grp1.create_dataset('M200',           data = M200,                   dtype = 'float')
    Mset    = grp1.create_dataset('r200',           data = r200,                   dtype = 'float')
    dset    = grp1.create_dataset('MeanStellarAge', data = mean_stellar_age,       dtype = 'float')
    dset    = grp1.create_dataset('DT',             data = DT,                     dtype = 'float')
    dset    = grp1.create_dataset('fsub200',        data = fsub,                   dtype = 'float')
    dset    = grp1.create_dataset('Lstar_rband',    data = Lstar_r,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_zband',    data = Lstar_z,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_Yband',    data = Lstar_Y,                dtype = 'float')
    dset    = grp1.create_dataset('StellarMass',    data = mstar,                  dtype = 'float')
    dset    = grp1.create_dataset('GasMass',        data = mgas,                   dtype = 'float')
    dset    = grp1.create_dataset('H2Mass',         data = mH2,                    dtype = 'float')
    dset    = grp1.create_dataset('HIMass',         data = mHI,                    dtype = 'float')
    dset    = grp1.create_dataset('r50_stars',      data = rhalf_stars,            dtype = 'float')
    dset    = grp1.create_dataset('angJ_stars',     data = angJ_stars,             dtype = 'float')
    dset    = grp1.create_dataset('angJ_gas',       data = angJ_gas,               dtype = 'float')
    dset    = grp1.create_dataset('angJ_DM',        data = angJ_DM,                dtype = 'float')
    dset    = grp1.create_dataset('angtot_stars',   data = angtot_stars,           dtype = 'float')
    dset    = grp1.create_dataset('angtot_gas',     data = angtot_gas,             dtype = 'float')
    dset    = grp1.create_dataset('angtot_DM',      data = angtot_DM,              dtype = 'float')
    dset    = grp1.create_dataset('SFR_50kpc',      data = SFR,                    dtype = 'float')
    
    dset    = grp2.create_dataset('xbin',           data = xbin,                   dtype='float')
    dset    = grp2.create_dataset('xbin_linear',    data = xbin_linear,            dtype='float')

    dset    = grp2.create_dataset('nB_stars',           data = nB_stars,           dtype='float')
    dset    = grp2.create_dataset('R0_prof_stars',      data = R0_prof_stars,      dtype='float')
    dset    = grp2.create_dataset('Rm_prof_stars',      data = Rm_prof_stars,      dtype='float')
    dset    = grp2.create_dataset('R1_prof_stars',      data = R1_prof_stars,      dtype='float')
    dset    = grp2.create_dataset('Sd0_prof_stars',     data = Sd0_prof_stars,     dtype='float')
    dset    = grp2.create_dataset('A2_prof_stars',      data = A2_prof_stars,      dtype='float')
    dset    = grp2.create_dataset('A2err_prof_stars',   data = A2err_prof_stars,   dtype='float')
    dset    = grp2.create_dataset('Phi2_prof_stars',    data = Phi2_prof_stars,    dtype='float')
    dset    = grp2.create_dataset('Phi2err_prof_stars', data = Phi2err_prof_stars, dtype='float')
    
    output.close()


plt.show()    
sys.exit() ###################################################################

