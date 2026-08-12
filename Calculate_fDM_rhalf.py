from   periodic_kdtree              import PeriodicCKDTree
from   colibre_utility              import *
import numpy                        as     np
import unyt                         as     unyt
import h5py                         as     h5
import swiftsimio                   as     sw
import os
import sys
import warnings
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
#BoxDir       = ["L100_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L200_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L200_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127

#BoxDir       = ["L025_m5/"]                         ; RunDir   = "THERMAL_AGN_m5/" ; snap = 127
#BoxDir       = ["L025_m6/"]                         ; RunDir   = "THERMAL_AGN_m6/" ; snap = 127
#BoxDir       = ["L025_m7/"]                         ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "fDM_"

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
    
    # --- Get some properties of the galaxies from SOAP
    rhalf_stars       = soap.bound_subhalo.half_mass_radius_stars[lhalo]
    halo_centre       = soap.input_halos.halo_centre[lhalo]
    TrackId           = soap.input_halos_hbtplus.track_id[lhalo]

    # --- convert Units
    rhalf_stars.convert_to_units('kpc')            ; rhalf_stars.convert_to_physical()
    halo_centre.convert_to_units('kpc')            ; halo_centre.convert_to_physical()

    # --- Extract raw np values upfront to avoid unyt overhead in loops ---
    r50_vals    = rhalf_stars.value
    centre_vals = halo_centre.value
    
    # --- stellar-to-DM mass fraction within integer mults of r50 ---
    mstar_1r50 = np.zeros((len(lhalo)));
    mdm_1r50   = np.zeros((len(lhalo)));
    mstar_2r50 = np.zeros((len(lhalo)));
    mdm_2r50   = np.zeros((len(lhalo)));
    mstar_3r50 = np.zeros((len(lhalo)));
    mdm_3r50   = np.zeros((len(lhalo)));

    # --- How many files do we need to look at?
    total_files  = 0 
    for root, _, filenames in os.walk(BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/"):
        #total_files += len(filenames) - 1
        total_files += sum(1 for f in filenames if not f.endswith('.old'))
    total_files -= 1  # subtract the master hdf5 file
    print('total files:', total_files)
    
    for ifile in range(total_files):
        Swiftfile    = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+'.'+str(ifile)+".hdf5"

        print('SWIFT file:',Swiftfile)
        data         = sw.load(Swiftfile)
        meta_data    = data.metadata
        ScaleFactor  = meta_data.scale_factor
        boxsize      = meta_data.boxsize * ScaleFactor
        boxsize.convert_to_units('kpc')
        boxsize_val  = boxsize.value
        
        # --- read the relevant stellar particle data
        pos_stars  = data.stars.coordinates           ; print(' ...read stellar coordinates')
        mass_stars = data.stars.masses                ; print(' ...read stellar masses')
        pos_dm     = data.dark_matter.coordinates     ; print(' ...read DM coordinates')
        mass_dm    = data.dark_matter.masses          ; print(' ...read DM masses')
        
        pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()                                                                                 
        mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()
        pos_dm.convert_to_units('kpc')      ; pos_dm.convert_to_physical()                                                                                 
        mass_dm.convert_to_units('Msun')    ; mass_dm.convert_to_physical()

        # --- Extract np arrays
        pos_stars_val  = pos_stars.value
        mass_stars_val = mass_stars.value
        pos_dm_val     = pos_dm.value
        mass_dm_val    = mass_dm.value
        
        # Build relevant KDE trees
        print(' \n Building particle tree...')
        star_tree = PeriodicCKDTree(boxsize_val, pos_stars, leafsize=100) ; print(' Star particle tree done... \n')
        dm_tree = PeriodicCKDTree(boxsize_val, pos_dm, leafsize=100) ; print(' DM particle tree done... \n')    
    
        # --- Calculate the various sizes for each galaxy of interest
        for ihalo, lh in enumerate(lhalo):
            # -------------------------------------------------
            # --- Compute mass w/in r50 for KDTree particles ---
            # -------------------------------------------------
            c_i   = centre_vals[ihalo]
            r50_i = r50_vals[ihalo]

            r1_sq = r50_i**2
            r2_sq = (2.0 * r50_i)**2
            r3    = 3.0 * r50_i

            # --- Stellar Particles ---
            idx_star = star_tree.query_ball_point(c_i, r3)
            if len(idx_star) > 0:
                dx = pos_stars_val[idx_star] - c_i
                dx -= boxsize_val * np.round(dx / boxsize_val)  # Periodic boundary correction
                dr2 = np.sum(dx**2, axis=1)

                m_pts = mass_stars_val[idx_star]
                mstar_1r50[ihalo] += np.sum(m_pts[dr2 <= r1_sq])
                mstar_2r50[ihalo] += np.sum(m_pts[dr2 <= r2_sq])
                mstar_3r50[ihalo] += np.sum(m_pts)  # All queried particles fall within 3*r50

            # --- Dark Matter Particles ---
            idx_dm = dm_tree.query_ball_point(c_i, r3)
            if len(idx_dm) > 0:
                dx = pos_dm_val[idx_dm] - c_i
                dx -= boxsize_val * np.round(dx / boxsize_val)  # Periodic boundary correction
                dr2 = np.sum(dx**2, axis=1)

                m_pts = mass_dm_val[idx_dm]
                mdm_1r50[ihalo] += np.sum(m_pts[dr2 <= r1_sq])
                mdm_2r50[ihalo] += np.sum(m_pts[dr2 <= r2_sq])
                mdm_3r50[ihalo] += np.sum(m_pts)

    # --- Compute stellar-to-dark matter mass ratios (M_* / M_DM) ---
    fDM_r50  = np.where(mdm_1r50 > 0, mstar_1r50 / mdm_1r50, 0.0)
    fDM_2r50 = np.where(mdm_2r50 > 0, mstar_2r50 / mdm_2r50, 0.0)
    fDM_3r50 = np.where(mdm_3r50 > 0, mstar_3r50 / mdm_3r50, 0.0)
    print(fDM_r50)
    
    # --- Write to hdf5
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"                  #Local path
    #fn = "/cosma8/data/do019/dc-fros1/Frosst_2026_Outputs/"+BoxDir[0]+RunDir+fname+ext3+".hdf5" #COSMA path
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w")
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")
    
    dset    = grp0.create_dataset('Redshift',       data = 1./ScaleFactor - 1,     dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId,                dtype = 'int')
    dset    = grp1.create_dataset('r50_stars',      data = rhalf_stars,            dtype = 'float')
    dset    = grp1.create_dataset('fDM_r50',        data = fDM_r50,                dtype = 'float')
    dset    = grp1.create_dataset('fDM_2r50',       data = fDM_2r50,               dtype = 'float')
    dset    = grp1.create_dataset('fDM_3r50',       data = fDM_3r50,               dtype = 'float')

    output.close()

sys.exit() ###################################################################

