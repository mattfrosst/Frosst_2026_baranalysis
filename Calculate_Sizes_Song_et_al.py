from   HBTReader         import HBTReader
from   scipy.spatial     import KDTree, cKDTree
from   periodic_kdtree   import PeriodicCKDTree
from   GalaxyProfiles    import *
from   colibre_functions import *
from   halo_shape        import *
import numpy             as     np
import scipy             as     scipy
import unyt              as     unyt
import h5py              as     h5
import swiftsimio        as     sw
import matplotlib.pyplot as     plt 
import rotation          as     rot
import profiling         as     prof
import scipy.optimize    as     sp_opt
import os
import sys
import time 
import itertools
import warnings 
import match_searchsorted as mi
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)

# --------------------------------------------------------------------------
"""
plt.clf() ; plt.ion()
fig  = plt.figure(1, figsize=(1,1))
fig.set_size_inches(8, 4, forward=True)

pos = [0.08,0.15,0.40,0.82] ; ax1  = fig.add_axes(pos)
pos = [0.58,0.15,0.40,0.82] ; ax2  = fig.add_axes(pos)
ax1.tick_params(axis="both",direction="in", pad=5)
ax1.set_xlim(-2,2)  ; ax1.set_ylim(5.0,12.0)
ax2.set_xlim(-2,2)  ; ax2.set_ylim(5.0,12.)
for ax in [ax1,ax2]: ax.set_xlabel(r'$\log\,\,\,r\,\,{\rm [kpc]}$',fontsize=15)
ax1.set_ylabel(r'$\log \,\,M(r),\,\,L(r)$',            fontsize=15)
ax2.set_ylabel(r'$\log \,\,\Sigma_\star,\,\,\Sigma_L$',fontsize=15)
"""


# ---- Simulation information ----
BasePath     = "/Volumes/RAID5/Simulations/COLIBRE/" ; SnapBase = "colibre_"
BoxDir       = ["L0025N0188/"]                       ; RunDir   = "THERMAL_AGN_m7/" ; snap = 127
#BoxDir       = ["L0200N3008/"]                       ; RunDir   = "THERMAL_AGN_m6/"          ; snap = 127
#BoxDir       = ["L0400N3008/"]                       ; RunDir   = "THERMAL_AGN_m7/"          ; snap = 127
#BoxDir       = ["L0100N1504/"]                       ; RunDir   = "Thermal_non_equilibrium/" ; snap = 127

DoBound      = False # Use only bound particles (True) or all particles within an aperture (False)?
fname        = "Stars_Mproj_Sigma_Prof_z_Y_"

# ---- analysis Information ----
Nstar_min    = 1e3  # Minimum number of stellar particles
Nstar_max    = 1e10 # Minimum number of stellar particles

for     idir,  Dir  in enumerate(BoxDir):

    ext4         = str(snap).zfill(4)
    ext3         = str(snap).zfill(3)

    HBTdir       = BasePath+Dir+RunDir+"HBT-HERONS/"
    #HBTdir       = BasePath+Dir+RunDir+"HBTplus/"

    Swiftfile    = BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/colibre_"+ext4+".hdf5"

    soap_file    = BasePath+Dir+RunDir+"SOAP-HBT/halo_properties_"+ext4+".hdf5"
    #soap_file    = BasePath+Dir+RunDir+"SOAP/halo_properties_"+ext4+".hdf5"
   
    # --- Initialize the HBT data
    print('HBT directory:',HBTdir)
    reader       = HBTReader(HBTdir) 
    f            = h5.File(HBTdir+ext3+"/SubSnap_"+ext3+".0.hdf5", "r")

    # --- Read the HBT halo data
    subhalos     = reader.LoadSubhalos(snap)
    subhalos     = reader.LoadSubhalos(snap,selection=['NboundType','TrackId'])
        
    # --- Read SOAP halo data 
    print('SOAP file: ', soap_file)
    soap         = sw.load(soap_file)

    # --- find all halos with NDM > NDM_min (Using HBT catalogs)
    lhalo        = np.where((subhalos['NboundType'][:,4] >= Nstar_min) & (subhalos['NboundType'][:,4] <= Nstar_max))[0]
    #if DoBound:
    #    hbt_ids  = reader.LoadParticles(snap)[lhalo]

    print('- snapshot        :',snap)
    print('  Number of galaxies selected: ',len(lhalo)) 

    #hbt_index        = soap.input_halos.halo_catalogue_index         # Index of galaxy in HBTplus catalog
    TrackId          = soap.input_halos_hbtplus.track_id
    # --- find matching halos in HBT and SOAP catalogues 
    #soap_match       = mi.match(lhalo,hbt_index)
    soap_match          = mi.match(subhalos['TrackId'][lhalo],TrackId)

    # --- Get some properties of the galaxies from SOAP
    rhalf_stars      = soap.bound_subhalo.half_mass_radius_stars[soap_match]     
    rhalf_gas        = soap.bound_subhalo.half_mass_radius_gas[soap_match]     
    nstar            = soap.bound_subhalo.number_of_star_particles[soap_match]        
    ngas             = soap.bound_subhalo.number_of_gas_particles[soap_match]         
    mstar            = soap.exclusive_sphere_50kpc.stellar_mass[soap_match]                 
    mgas             = soap.exclusive_sphere_50kpc.gas_mass[soap_match]                     
    mHI              = soap.exclusive_sphere_50kpc.atomic_hydrogen_mass[soap_match]         
    mH2              = soap.exclusive_sphere_50kpc.molecular_hydrogen_mass[soap_match]      
    mean_stellar_age = soap.bound_subhalo.mass_weighted_mean_stellar_age[soap_match]     
    kappa_co_all     = soap.bound_subhalo.kappa_corot_stars[soap_match]                     
    DT               = soap.exclusive_sphere_50kpc.disc_to_total_stellar_mass_fraction[soap_match]          
    N200             = soap.spherical_overdensity_200_crit.number_of_dark_matter_particles[soap_match]
    M200             = soap.spherical_overdensity_200_crit.total_mass[soap_match]
    r200             = soap.spherical_overdensity_200_crit.soradius[soap_match]
    angJ_stars       = soap.inclusive_sphere_50kpc.angular_momentum_stars[soap_match]
    angJ_gas         = soap.inclusive_sphere_50kpc.angular_momentum_gas[soap_match]
    angJ_baryons     = soap.inclusive_sphere_50kpc.angular_momentum_baryons[soap_match]
    angJ_DM          = soap.spherical_overdensity_200_crit.angular_momentum_dark_matter[soap_match]
    velCOM_stars     = soap.spherical_overdensity_500_crit.stellar_centre_of_mass_velocity[soap_match]
    is_central       = soap.input_halos.is_central[soap_match]
    halo_centre      = soap.input_halos.halo_centre[soap_match]
    SFR              = soap.exclusive_sphere_50kpc.star_formation_rate[soap_match]
    fsub             = soap.spherical_overdensity_200_crit.mass_fraction_satellites[soap_match]
    
    # -- luminoxsities are in the following order: u, g, r, i, z, Y, J, H, K
    Lstar_r          = soap.inclusive_sphere_50kpc.stellar_luminosity[soap_match,2]
    Lstar_z          = soap.inclusive_sphere_50kpc.stellar_luminosity[soap_match,4]
    Lstar_Y          = soap.inclusive_sphere_50kpc.stellar_luminosity[soap_match,5]

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

    # --- define arrays for output
    Nprof          = 51
    x_range        = [-2,3]
    xbin           = np.linspace(x_range[0],x_range[1],Nprof)
    dx             = xbin[1] - xbin[0] # Bins are equally spaced in x
    xbin_min       = np.append(0,xbin[:-1])

    Mproj_stars    = zeros((len(lhalo),Nprof))  ; Sigma_stars     = zeros((len(lhalo),Nprof))
    L_z_proj_stars = zeros((len(lhalo),Nprof))  ; Sigma_L_z_stars = zeros((len(lhalo),Nprof))
    L_Y_proj_stars = zeros((len(lhalo),Nprof))  ; Sigma_L_Y_stars = zeros((len(lhalo),Nprof))
    nstar_count    = zeros(len(lhalo),dtype=int)
    
    total_files  = 0 
    for root, _, filenames in os.walk(BasePath+Dir+RunDir+"snapshots/"+SnapBase+ext4+"/"):
        total_files += len(filenames) - 1

    for ifile in arange(total_files):
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
        lum_stars_z   = data.stars.luminosities.GAMA_z   ; print(' ...read GAMA z-band Luminosities')
        lum_stars_Y   = data.stars.luminosities.GAMA_Y   ; print(' ...read GAMA Y-band Luminosities')
        if DoBound:
            IDs_stars = data.stars.particle_ids          ; print(' ...read stellar IDs')
            
        pos_stars.convert_to_units('kpc')   ; pos_stars.convert_to_physical()                                                                                 
        mass_stars.convert_to_units('Msun') ; mass_stars.convert_to_physical()                                                                                
            
        # Build relevant KDE trees
        print(' \n Building particle tree...')
        star_tree     = PeriodicCKDTree(boxsize, pos_stars, leafsize=100) ; print(' Star particle tree done... \n')    
    
        # --- Calculate the various sizes for each galaxy of interest
        for ihalo, lh in enumerate(lhalo):

            # ------------------------------------------
            # --- Compute sizes for KDTree particles ---
            # ------------------------------------------
            Rcut        = 50.

            l_star      = asarray(star_tree.query_ball_point(halo_centre[ihalo,:], Rcut))   

            # --- reset coordinates to COP, CMV
            #if len(l_star)==0:
            #    print(' Warning - found no tree particles for halo',lh,len(l_star))
            #if len(l_star) > 0:
            #    if DoBound:
            #        list_match_hbt = mi.match(hbt_ids[ihalo],IDs_stars[l_star].value)
            #        lstar_tree     = l_tree[list_match_hbt[list_match_hbt>=0]]
            #    else:
            #        lstar_tree     = l_star

            lstar_tree     = l_star
                
            if len(lstar_tree) > 0:
                nstar_count[ihalo] = +len(l_star)
            
                pos_tree   = pos_stars[lstar_tree,:].value - halo_centre[ihalo,:].value
                # --- wrap-around if needed
                R_tree     = sqrt(pos_tree[:,0]**2 + pos_tree[:,1]**2)
            
                id_bins           = np.digitize(np.log10(R_tree),xbin,right=True)
                for i in range(Nprof):
                    c_mask        = id_bins <=i
                    d_mask        = id_bins ==i        
                    Mproj_stars[ihalo,i]     += np.sum(mass_stars[lstar_tree[c_mask]].value)
                    Sigma_stars[ihalo,i]     += np.sum(mass_stars[lstar_tree[d_mask]].value)

                    L_z_proj_stars[ihalo,i]  += np.sum(lum_stars_z[lstar_tree[c_mask]].value)
                    Sigma_L_z_stars[ihalo,i] += np.sum(lum_stars_z[lstar_tree[d_mask]].value)

                    L_Y_proj_stars[ihalo,i]  += np.sum(lum_stars_Y[lstar_tree[c_mask]].value)
                    Sigma_L_Y_stars[ihalo,i] += np.sum(lum_stars_Y[lstar_tree[d_mask]].value)
                    
                    
        fracs     = round(float(ihalo+1)/len(lhalo),4)
        if ihalo % 100  ==0:
            print(' Group:',lh,' | f:',fracs)
        elif ihalo   ==len(lhalo)-1:
            print(' Group:',lh,' | f:',fracs)

    # Normalize the profiles
    dA               = np.pi * ((10.**xbin)**2 - (10.**xbin_min)**2)
    Sigma_stars     /= dA
    Sigma_L_z_stars /= dA
    Sigma_L_Y_stars /= dA
    xbin_cum         = 10.**xbin
    xbin             = 10.**(xbin - 0.5*dx)
        
    #lc = np.where((is_central == 1) & (fsub < 0.05))[0]
    #for ihalo, lh in enumerate(lhalo): 
    #    ax2.plot(np.log10(xbin),    np.log10(Sigma_stars[ihalo,:]),color='orangered', linestyle='-',linewidth=0.5)
    #    ax1.plot(np.log10(xbin_cum),np.log10(Mproj_stars[ihalo,:]),color='r',         linestyle='-',linewidth=0.5)
    #    ax2.plot(np.log10(xbin),    np.log10(Sigma_L_z_stars[ihalo,:]),color='dodgerblue', linestyle='-',linewidth=0.5)
    #    ax1.plot(np.log10(xbin_cum),np.log10(L_z_proj_stars[ihalo,:]),color='b',         linestyle='-',linewidth=0.5)
    #    ax2.plot(np.log10(xbin),    np.log10(Sigma_L_Y_stars[ihalo,:]),color='olivedrab', linestyle='-',linewidth=0.5)
    #    ax1.plot(np.log10(xbin_cum),np.log10(L_Y_proj_stars[ihalo,:]),color='g',         linestyle='-',linewidth=0.5)
    #    ax1.axhline(np.log10(Lstar_z[ihalo]),color='b')
    #    ax1.axhline(np.log10(mstar[ihalo]),color='r')
        
    fn = BasePath+Dir[:-1]+"_OutPuts/"+RunDir+fname+ext3+".hdf5"
    print('\n Writing to:',fn)

    output  = h5.File(fn, "w")
    grp0    = output.create_group("Header")
    grp1    = output.create_group("HaloData")
    grp2    = output.create_group("Profiles")

    dset    = grp0.create_dataset('Redshift',       data = 1./ScaleFactor - 1,  dtype = 'float')

    dset    = grp1.create_dataset('TrackId',        data = TrackId[soap_match], dtype = 'int')
    dset    = grp1.create_dataset('is_central',     data = is_central,          dtype = 'int')
    dset    = grp1.create_dataset('NumStellarPart', data = nstar,               dtype = 'int')
    dset    = grp1.create_dataset('NumGasPart',     data = ngas,                dtype = 'int')
    dset    = grp1.create_dataset('N200_DM',        data = N200,                dtype = 'int')
    dset    = grp1.create_dataset('M200',           data = M200,                dtype = 'float')
    Mset    = grp1.create_dataset('r200',           data = r200,                dtype = 'float')
    dset    = grp1.create_dataset('MeanStellarAge', data = mean_stellar_age,    dtype = 'float')
    dset    = grp1.create_dataset('DT',             data = DT,                  dtype = 'float')
    dset    = grp1.create_dataset('fsub200',        data = fsub,                dtype = 'float')
    dset    = grp1.create_dataset('Lstar_rband',    data = Lstar_r,             dtype = 'float')
    dset    = grp1.create_dataset('Lstar_zband',    data = Lstar_z,             dtype = 'float')
    dset    = grp1.create_dataset('Lstar_Yband',    data = Lstar_Y,             dtype = 'float')
    dset    = grp1.create_dataset('StellarMass',    data = mstar,               dtype = 'float')
    dset    = grp1.create_dataset('GasMass',        data = mgas,                dtype = 'float')
    dset    = grp1.create_dataset('H2Mass',         data = mH2,                 dtype = 'float')
    dset    = grp1.create_dataset('HIMass',         data = mHI,                 dtype = 'float')
    dset    = grp1.create_dataset('r50_stars',      data = rhalf_stars,         dtype = 'float')
    dset    = grp1.create_dataset('angJ_stars',     data = angJ_stars,          dtype = 'float')
    dset    = grp1.create_dataset('angJ_gas',       data = angJ_gas,            dtype = 'float')
    dset    = grp1.create_dataset('angJ_DM',        data = angJ_DM,             dtype = 'float')
    dset    = grp1.create_dataset('angtot_stars',   data = angtot_stars,        dtype = 'float')
    dset    = grp1.create_dataset('angtot_gas',     data = angtot_gas,          dtype = 'float')
    dset    = grp1.create_dataset('angtot_DM',      data = angtot_DM,           dtype = 'float')
    dset    = grp1.create_dataset('SFR_50kpc',      data = SFR,                 dtype = 'float')
    
    dset    = grp2.create_dataset('xbin',           data = xbin,                dtype='float')
    dset    = grp2.create_dataset('xbin_cum',       data = xbin_cum,            dtype='float')
    dset    = grp2.create_dataset('Sigma_stars',    data = Sigma_stars,         dtype='float')
    dset    = grp2.create_dataset('Mproj_stars',    data = Mproj_stars,         dtype='float')
    dset    = grp2.create_dataset('Sigma_L_z_stars',data = Sigma_L_z_stars,     dtype='float')
    dset    = grp2.create_dataset('L_z_proj_stars', data = L_z_proj_stars,      dtype='float')
    dset    = grp2.create_dataset('Sigma_L_Y_stars',data = Sigma_L_Y_stars,     dtype='float')
    dset    = grp2.create_dataset('L_Y_proj_stars', data = L_Y_proj_stars,      dtype='float')

    output.close()


plt.show()    
sys.exit() ###################################################################

