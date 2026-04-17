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
    stop

    binData = np.zeros((nGal, nBin, 7))

    binData[:, :, 0] = nB;
    binData[:, :, 1] = profiles['R0_prof_stars']
    binData[:, :, 2] = profiles['Rm_prof_stars']
    binData[:, :, 3] = profiles['R1_prof_stars']
    binData[:, :, 4] = profiles['A2_prof_stars']
    binData[:, :, 5] = profiles['A2err_prof_stars']
    binData[:, :, 6] = profiles['Phi2_prof_stars']
    binData[:, :, 7] = profiles['Phi2err_prof_stars']

    # ---------------------------
    #    Find the bar region
    # ---------------------------

    #bar_tool  = FourierMethodFast(mass_stars[lstar_tree], pos_tree[:,0], pos_tree[:,1], vel_tree[:,0], vel_tree[:,1])
    #binData   = bar_tool.analyseBins(xbin_linear)

    #Need to reconstruct binData... Annoying - find out what we actually need here and pass indiv arrays.
    b0, b1    = bar_tool.findBarRegion(binData, minA2Bar=0.2, maxDPsi=15.0, minDexBar=0.15, minNumBar=200)

plt.show()    
sys.exit() ###################################################################

