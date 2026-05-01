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
#BoxDir       = ["L050_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L100_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "Stars_Mproj_Bar_Prof_"
rname        = "Stars_Mproj_Bar_Region_"
bname        = "Stars_Mproj_Bar_Omega_"

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
    soap          = sw.load(soap_file)

    # --- Read the selection data (Stellar and DM resolution)
    #NDM_subhalo   = soap.bound_subhalo.number_of_dark_matter_particles
    Nstar_subhalo = soap.bound_subhalo.number_of_star_particles

    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (Using SOAP catalogs)
    lhalo         = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
    
    # --- Get some properties of the galaxies from SOAP
    # --- Bound subhalo: galaxy properties
    rhalf_stars       = soap.bound_subhalo.half_mass_radius_stars[lhalo]
    nstar             = soap.bound_subhalo.number_of_star_particles[lhalo]

    # --- In spherical r200crit: halo properties
    velCOM_stars      = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[lhalo]

    # -- Inclusive sphere 50kpc: angular momentum
    angJ_stars        = soap.inclusive_sphere_50kpc.angular_momentum_stars[lhalo]

    # --- HBT halo properties
    halo_centre       = soap.input_halos.halo_centre[lhalo]
    TrackId           = soap.input_halos_hbtplus.track_id[lhalo]

    # --- convert Units
    velCOM_stars.convert_to_units('km/s')          ; velCOM_stars.convert_to_physical()
    rhalf_stars.convert_to_units('kpc')            ; rhalf_stars.convert_to_physical()
    halo_centre.convert_to_units('kpc')            ; halo_centre.convert_to_physical()
    angJ_stars.convert_to_units('Msun*kpc*km/s')   ; angJ_stars.convert_to_physical()

    # --- Read the selection data, i.e., only galaxies with a bar identified.
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+rname+ext3+".hdf5"
    print('\n Reading:',fn)
    data     = h5.File(fn, "r")
    HaloData = data["HaloData"];

    # --- Read required bar region properties
    TrackId_analysed  = HaloData["TrackId"];
    R0bar_analysed    = HaloData["R0Bar_value"]
    R1bar_analysed    = HaloData["R1Bar_value"]
    maxA2_analysed    = HaloData["maxA2_value"]
    
    print('- snapshot        :',snap)
    print('  Number of galaxies selected: ',len(lhalo)) 
    
    # --- define arrays for output; n.b. we will normalise the positions by the half mass radius
    Nprof          = 41
    x_range        = [-1,1] # i.e., bins from 0.01Rhalf to 10Rhalf
    xbin           = np.linspace(x_range[0],x_range[1],Nprof)
    dx             = xbin[1] - xbin[0] # Bins are equally spaced in x
    xbin_linear    = np.append(0,10**xbin[:])
    print('xbin:', xbin, 'dx: ', dx, 'xbin_linear: ', xbin_linear)

    # --- number of stars in bar region bins (nB)
    nBar_stars       = np.zeros((len(lhalo)));

    # --- maximum bar strength within the bar region
    maxA2_stars      = np.zeros((len(lhalo)));
    
    # --- mass weighted bin edges (left, right -> R0, R1) and middle (Rm)
    R0_stars = np.zeros((len(lhalo)));
    Rm_stars = np.zeros((len(lhalo)));
    R1_stars = np.zeros((len(lhalo)));

    # --- mass weighted 2nd fourier phase angle and error in bar region
    Phi2_stars    = np.zeros((len(lhalo)));
    Phi2err_stars = np.zeros((len(lhalo)));

    # --- mass weighted 2nd fourier position pattern speed and error in bar region
    Omega_stars    = np.zeros((len(lhalo)));
    Omegaerr_stars = np.zeros((len(lhalo)));

    # --- statistical correlation between ψ and Ω in bar region
    Cov_stars = np.zeros((len(lhalo)));
    
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
            print('Looking at galaxy ID:', TrackId[ihalo])
            # -------------------------------------------------
            # --- Compute Bar Profiles for KDTree particles ---
            # -------------------------------------------------
            Rcut        = 50.
            lstar_tree  = np.asarray(star_tree.query_ball_point(halo_centre[ihalo,:], Rcut))

            if (len(lstar_tree) > 0):
                if (R1bar_analysed[ihalo] > 0.1):
                    pos_tree   = pos_stars[lstar_tree,:].value - halo_centre[ihalo,:].value
                    vel_tree   = vel_stars[lstar_tree,:].value - velCOM_stars[ihalo,:].value

                    # --- align galaxy with z-component of AM within 50kpc
                    trans = rotation_matrix_from_vectors(angJ_stars[ihalo,:], [0,0,1])   # Transformation Matrix
                    pos_tree = (trans @ pos_tree.T).T / rhalf_stars[ihalo].value         # One BLAS call, then norm by stellar half mass radius, units: rhalf
                    vel_tree = (trans @ vel_tree.T).T                                    # One BLAS call, units: km/s 
                    
                    # Setup FourierMethodFast class
                    bar_tool  = FourierMethodFast(mass_stars[lstar_tree], pos_tree[:,0], pos_tree[:,1], vel_tree[:,0], vel_tree[:,1])
                    
                    # Fourier analysis on bar region
                    binOmega = bar_tool.analyseOmega(xbin_linear, R0bar_analysed[ihalo], R1bar_analysed[ihalo], tophat=True)
                    print('analyseOmega', lh, binOmega)

                    # --- Save data.                                               # UNITS (notes)
                    nBar_stars[ihalo]     = binOmega[0]                            # dimensionless (particle count)
                    maxA2_stars[ihalo]    = maxA2_analysed[ihalo]                  # dimensionless (A2 is a ratio)
                    R0_stars[ihalo]       = binOmega[1]                            # rhalf
                    Rm_stars[ihalo]       = binOmega[2]                            # rhalf
                    R1_stars[ihalo]       = binOmega[3]                            # rhalf
                    Phi2_stars[ihalo]     = binOmega[4]                            # radians
                    Phi2err_stars[ihalo]  = binOmega[5]                            # radians
                    Omega_stars[ihalo]    = binOmega[6] / rhalf_stars[ihalo].value # km/s/kpc
                    Omegaerr_stars[ihalo] = binOmega[7] / rhalf_stars[ihalo].value # km/s/kpc
                    Cov_stars[ihalo]      = binOmega[8]                            # correlation coefficient

        fracs     = round(float(ihalo+1)/len(lhalo),4)
        if ihalo % 100  ==0:
            print(' Group:',lh,' | f:',fracs)
        elif ihalo   ==len(lhalo)-1:
            print(' Group:',lh,' | f:',fracs)

    print('nBar:',  nBar_stars)
    print('A2max:', maxA2_stars)
    print('Rbar:',  R1_stars)
    print('PAbar:', Phi2_stars)
    print('Obar:',  Omega_stars)
    
    # --- Write to hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+bname+ext3+".hdf5"                  #Local path
    #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+bname+ext3+".hdf5" #COSMA path
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w")
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")

    dset    = grp0.create_dataset('Redshift',       data = 1./ScaleFactor - 1,            dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,                       dtype = 'int')
    dset    = grp1.create_dataset('nBar_stars_dimless',    data = nBar_stars,             dtype='float')
    dset    = grp1.create_dataset('maxA2_stars_dimless',   data = maxA2_stars,            dtype='float')
    dset    = grp1.create_dataset('R0_stars_rhalf',        data = R0_stars,               dtype='float')
    dset    = grp1.create_dataset('Rm_stars_rhalf',        data = Rm_stars,               dtype='float')
    dset    = grp1.create_dataset('R1_stars_rhalf',        data = R1_stars,               dtype='float')
    dset    = grp1.create_dataset('Phi2_stars_radians',    data = Phi2_stars,             dtype='float')
    dset    = grp1.create_dataset('Phi2err_stars_radians', data = Phi2err_stars,          dtype='float')
    dset    = grp1.create_dataset('Omega_stars_kmskpc',    data = Omega_stars,            dtype='float')
    dset    = grp1.create_dataset('Omegaerr_stars_kmskpc', data = Omegaerr_stars,         dtype='float')
    dset    = grp1.create_dataset('Cov_stars_dimless',     data = Cov_stars,              dtype='float')
    
    output.close()


plt.show()    
sys.exit() ###################################################################

