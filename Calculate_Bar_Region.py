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

    # --- Read pre-calculated profiles from hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"
    print('\n Writing to:',fn)
    data  = h5.File(fn, "r")
    profiles = data["Profiles"]

    # Determine size of profile arrays
    nB = profiles['nB_stars']
    print(nB, nB.shape)
    nGal = nB.shape[0]; nBin = nB.shape[1]
    print(nGal, nBin)

    # --- Read required profiles
    R0_prof   = profiles['R0_prof_stars']
    R1_prof   = profiles['R1_prof_stars']
    A2_prof   = profiles['A2_prof_stars']
    Phi2_prof = profiles['Phi2_prof_stars']

    # ---------------------------
    #    Find the bar region
    # ---------------------------
    for i in range(nGal):
        print('profiles for gal i:', R0_prof[i], R1_prof[i], A2_prof[i], Phi2_prof[i])
        b0, b1    = findBarRegion(R0_prof[i], R1_prof[i], A2_prof[i], Phi2_prof[i],
                                  minA2Bar=0.2, maxDPsi=15.0, minDexBar=0.15, minNumBar=200)
        print("Inner and outer index: ", b0, b1)
        print("Inner and outer Rbar: ", R0_prof[b0], R1_prof[b1])

sys.exit() ###################################################################

