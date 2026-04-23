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
rname        = "Stars_Mproj_Bar_Prof_"
wname        = "Stars_Mproj_Bar_Region_"

for     idir,  Dir  in enumerate(BoxDir):

    ext4         = str(snap).zfill(4)
    ext3         = str(snap).zfill(3)

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
    nGal = nB.shape[0]; nBin = nB.shape[1]

    # --- Read required profiles
    R0_prof   = Profiles["R0_prof_stars"]
    R1_prof   = Profiles["R1_prof_stars"]
    A2_prof   = Profiles["A2_prof_stars"]
    Phi2_prof = Profiles["Phi2_prof_stars"]

    # --- Write bar region data
    nBar_galaxies = np.zeros((nGal));
    b0_galaxies   = np.zeros((nGal)); b1_galaxies = np.zeros((nGal))
    R0_galaxies   = np.zeros((nGal)); R1_galaxies = np.zeros((nGal))

    # ---------------------------
    #    Find the bar region
    # ---------------------------
    print("Analizing ", nGal, " galaxies with ", nBin, "bins...")
    for i in range(nGal):
        nBar, b0, b1, R0_bar, R1_bar = findBarRegion(nB[i], R0_prof[i], R1_prof[i], A2_prof[i], Phi2_prof[i],
                                                     minA2Bar=0.2, maxDPhi2=15.0, minDexBar=0.15, minNumBar=200)
        print("Inner and outer index: ", b0, b1)
        print("Inner and outer Rbar: ", R0_bar, R1_bar)

        nbar_galaxies[i] = nBar;
        b0_galaxies[i] = b0;
        b1_galaxies[i] = b1;

        R0_galaxies[i] = R0_bar;
        R1_galaxies[i] = R1_bar;

    # -----------------------------------------------
    #    Write the bar region properties to hdf5
    # -----------------------------------------------

    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+wname+ext3+".hdf5"
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w") # Might want to just write this back to the original file? Risky.
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")

    dset    = grp0.create_dataset('Redshift',       data = Redshift,       dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,        dtype = 'int')
    dset    = grp1.create_dataset('nBar',           data = nBar_galaxies,  dtype = 'int')
    dset    = grp1.create_dataset('R0_index',       data = b0_galaxies,    dtype = 'int')
    dset    = grp1.create_dataset('R1_index',       data = b0_galaxies,    dtype = 'int')
    dset    = grp1.create_dataset('R0_value',       data = R0_galaxies,    dtype = 'float')
    dset    = grp1.create_dataset('R1_value',       data = R1_galaxies,    dtype = 'float')

    output.close()

plt.show()
sys.exit() ###################################################################
