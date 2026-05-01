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
BasePath     = "/Users/23229092/Documents/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L012_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L0200N3008/"]                       ; RunDir   = "THERMAL_AGN_m6/"          ; snap = 127
#BoxDir       = ["L0400N3008/"]                       ; RunDir   = "THERMAL_AGN_m7/"          ; snap = 127
#BoxDir       = ["L0100N1504/"]                       ; RunDir   = "Thermal_non_equilibrium/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
rname        = "Stars_Mproj_Bar_Prof_"
wname        = "Stars_Mproj_Bar_Region_"

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
    Nstar_subhalo = soap.bound_subhalo.number_of_star_particles

    # --- find all halos with Nstar_max >= Nstar >= Nstar_min (Using SOAP catalogs)
    lhalo         = np.where((Nstar_subhalo >= Nstar_min) & (Nstar_subhalo <= Nstar_max))[0]
    
    # --- Read pre-calculated profiles from hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+rname+ext3+".hdf5"
    print('\n Reading:',fn)
    data  = h5.File(fn, "r")
    Header   = data["Header"];
    HaloData = data["HaloData"];
    Profiles = data["Profiles"];

    # --- File information
    TrackId  = HaloData["TrackId"];
    Redshift = Header["Redshift"];

    # Determine size of profile arrays
    nB = Profiles['nB_stars']

    # --- Read required profiles
    R0_prof   = Profiles["R0_prof_stars"]
    R1_prof   = Profiles["R1_prof_stars"]
    A2_prof   = Profiles["A2_prof_stars"]
    Phi2_prof = Profiles["Phi2_prof_stars"]

    # --- Write bar region data
    nBar_galaxies     = np.zeros((len(lhalo)));
    isbarred_galaxies = np.zeros((len(lhalo)));
    maxA2_galaxies    = np.zeros((len(lhalo)));
    b0_galaxies   = np.zeros((len(lhalo))); b1_galaxies   = np.zeros((len(lhalo)))
    R0_galaxies   = np.zeros((len(lhalo))); R1_galaxies   = np.zeros((len(lhalo)))
    
    # ---------------------------
    #    Find the bar region
    # ---------------------------
    print("Analizing ", len(lhalo), " galaxies... ")
    for ihalo, lh in enumerate(lhalo):
        if np.any(nB[ihalo]):
            nBar, b0, b1, R0Bar, R1Bar, maxA2Bar, isbarred = findBarRegion(nB[ihalo], R0_prof[ihalo], R1_prof[ihalo], A2_prof[ihalo], Phi2_prof[ihalo],
                                                                           minA2Bar=0.2, maxDPhi2=15.0, minDexBar=0.15, minNumBar=200)
            print(ihalo, "Inner and outer index: ", b0, b1)
            print(ihalo, "Inner and outer Rbar: ", R0Bar, R1Bar)
            print(ihalo, "A2max: ", maxA2Bar)

            nBar_galaxies[ihalo]     = nBar;
            b0_galaxies[ihalo]       = b0;
            b1_galaxies[ihalo]       = b1;
            R0_galaxies[ihalo]       = R0Bar;
            R1_galaxies[ihalo]       = R1Bar;
            maxA2_galaxies[ihalo]    = maxA2Bar;
            isbarred_galaxies[ihalo] = isbarred;

    # -----------------------------------------------
    #    Write the bar region properties to hdf5
    # -----------------------------------------------

    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+wname+ext3+".hdf5"                                # Local path
    #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+wname+ext3+".hdf5" #COSMA path
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w") # Might want to just write this back to the original file? Risky.
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")

    dset    = grp0.create_dataset('Redshift',       data = Redshift,          dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,           dtype = 'int')
    dset    = grp1.create_dataset('nBar',           data = nBar_galaxies,     dtype = 'int')
    dset    = grp1.create_dataset('isbarred',       data = isbarred_galaxies, dtype = 'int')
    dset    = grp1.create_dataset('R0Bar_index',    data = b0_galaxies,       dtype = 'int')
    dset    = grp1.create_dataset('R1Bar_index',    data = b1_galaxies,       dtype = 'int')
    dset    = grp1.create_dataset('R0Bar_value',    data = R0_galaxies,       dtype = 'float')
    dset    = grp1.create_dataset('R1Bar_value',    data = R1_galaxies,       dtype = 'float')
    dset    = grp1.create_dataset('maxA2_value',    data = maxA2_galaxies,    dtype = 'float')

    output.close()

plt.show()
sys.exit() ###################################################################
