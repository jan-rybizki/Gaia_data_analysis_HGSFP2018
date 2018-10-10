import numpy as np

def compute_ruwe(x):
    from scipy.spatial import cKDTree
    # Load the renormalisation tables downloaded from https://www.cosmos.esa.int/web/gaia/dr2-known-issues 
    u0_g = np.load('../data/day_1/DR2_RUWE_V1/table_u0_g.npy')
    u0_gc = np.load('../data/day_1/DR2_RUWE_V1/table_u0_g_col.npy')
    # Making the uwe
    u = np.sqrt(np.divide(x.astrometric_chi2_al,(x.astrometric_n_good_obs_al - 5)))
    g = x.phot_g_mean_mag
    bpmrp = x.phot_bp_mean_mag - x.phot_rp_mean_mag
    # create KDTree for each table to find the nearest neighbour
    g_c_index = cKDTree(np.c_[u0_gc['g_mag'],u0_gc['bp_rp']]) 
    g_index = cKDTree(np.c_[u0_g['g_mag']])
    # discriminate the two cases with and without colour
    no_bprp = (np.isnan(bpmrp))
    # create array and divide the Unit Weight Error (UWE) by the normalisation
    ru = np.zeros_like(u)
    ru[no_bprp] = np.divide(u[no_bprp],u0_g['u0'][g_index.query(np.c_[g[no_bprp]])[1]])
    ru[~no_bprp] = np.divide(u[~no_bprp],u0_gc['u0'][g_c_index.query(np.c_[g[~no_bprp],bpmrp[~no_bprp]])[1]])
    return(ru)

def grvs_for_blue_sources(g,rp):
    """
    This comes from the GDR2 release paper
    """
    grp = g-rp
    a0 = 0.042319
    a1 = -0.65124
    a2 = 1.0215
    a3 = -1.3947
    a4 = 0.53768
    grvs = rp + a0 + a1 * grp + a2 * grp**2 + a3 * grp**3 + a4 * grp**4  
    return(grvs)

def grvs_for_red_sources(g,rp):
    grp = g-rp
    a0 = 132.32
    a1 = -377.28
    a2 = 402.32
    a3 = -190.97
    a4 = 34.026
    grvs = rp + a0 + a1 * grp + a2 * grp**2 + a3 * grp**3 + a4 * grp**4  
    return(grvs)


def linear_motion_approximation(x):
    """
    Calculates from parallax, pmra, pmdec, radial_velocity via linear motion approximation the
        closest distance in pc
        total velocity in km/s
        time of closest approach in Myr
    """
    total_velocity = np.sqrt((x['pmra']**2 + x['pmdec']**2) * np.divide(4.74047,x['parallax'])**2 + x['radial_velocity']**2)
    closest_distance = np.divide(np.divide(1000*4.74047,x['parallax']**2) * np.sqrt(x['pmra']**2 + x['pmdec']**2),total_velocity)
    closest_time = np.divide(-0.97779e9,x['parallax']) * np.divide(x['radial_velocity'],(x['pmra']**2 + x['pmdec']**2) * np.divide(4.74047,x['parallax'])**2 + x['radial_velocity']**2)
    return(closest_distance, total_velocity, np.divide(closest_time,1e6))

def sample_gaia_uncertainty(example,n_sample=100, no_correlation = False, extreme_covariance = False):
    # Mean vector and covariance matrix
    mu = np.array([example.parallax ,  example.pmra, example.pmdec])
    Sigma = np.array([[example.parallax_error**2, example.parallax_error*example.pmra_error*example.parallax_pmra_corr, example.parallax_error*example.pmdec_error*example.parallax_pmdec_corr],
       [example.parallax_error*example.pmra_error*example.parallax_pmra_corr, example.pmra_error**2, example.pmra_error*example.pmdec_error*example.pmra_pmdec_corr],
                 [example.parallax_error*example.pmdec_error*example.parallax_pmdec_corr, example.pmra_error*example.pmdec_error*example.pmra_pmdec_corr, example.pmdec_error**2]] )
    if extreme_covariance:
        Sigma = np.array([[example.parallax_error**2, example.parallax_error*example.pmra_error*1, example.parallax_error*example.pmdec_error*1],
               [example.parallax_error*example.pmra_error*1, example.pmra_error**2, example.pmra_error*example.pmdec_error*1],
                [example.parallax_error*example.pmdec_error*1, example.pmra_error*example.pmdec_error*1, example.pmdec_error**2]] )

    elif no_correlation:
        Sigma = np.array([[example.parallax_error**2, 0, 0],
       [0, example.pmra_error**2, 0],
                 [0, 0, example.pmdec_error**2]] )

    sample = np.random.multivariate_normal(mu, Sigma, size= n_sample, check_valid= 'warn')

    parallax = sample[:,0]
    pmra = sample[:,1]
    pmdec = sample[:,2]

    radial_velocity = np.random.normal(loc = example.radial_velocity, scale = example.radial_velocity_error, size = n_sample)
    return(parallax,pmra,pmdec,radial_velocity)

def linear_motion_approximation_split_input(parallax,pmra,pmdec,radial_velocity):
    """
    Calculates from parallax, pmra, pmdec, radial_velocity via linear motion approximation the
        closest distance in pc
        total velocity in km/s
        time of closest approach in Myr
    """
    total_velocity = np.sqrt((pmra**2 + pmdec**2) * np.divide(4.74047,parallax)**2 + radial_velocity**2)
    closest_distance = np.divide(np.divide(1000*4.74047,parallax**2) * np.sqrt(pmra**2 + pmdec**2),total_velocity)
    closest_time = np.divide(-0.97779e9,parallax) * np.divide(radial_velocity,(pmra**2 + pmdec**2) * np.divide(4.74047,parallax)**2 + radial_velocity**2)
    return(closest_distance, total_velocity, np.divide(closest_time,1e6))

def orbit_distance(o1,o2,time, dist_grid, potential):
    o2.integrate(time,potential)
    return(np.sqrt((o1.x(dist_grid)-o2.x(dist_grid))**2+(o1.y(dist_grid)-o2.y(dist_grid))**2+(o1.z(dist_grid)-o2.z(dist_grid))**2))

def sample_gaia_uncertainty_5d(x,n_sample=100, no_correlation = False):
    # Mean vector and covariance matrix
    mu = np.array([x.ra, x.dec, x.parallax, x.pmra, x.pmdec])
    t00 = x.ra_error**2
    t11 = x.dec_error**2
    t22 = x.parallax_error**2
    t33 = x.pmra_error**2
    t44 = x.pmdec_error**2
    t01 = x.ra_error*x.dec_error*x.ra_dec_corr
    t02 = x.ra_error*x.parallax_error*x.ra_parallax_corr
    t03 = x.ra_error*x.pmra_error*x.ra_pmra_corr
    t04 = x.ra_error*x.pmdec_error*x.ra_pmdec_corr
    t12 = x.dec_error*x.parallax_error*x.dec_parallax_corr
    t13 = x.dec_error*x.pmra_error*x.dec_pmra_corr
    t14 = x.dec_error*x.pmdec_error*x.dec_pmdec_corr
    t23 = x.parallax_error*x.pmra_error*x.parallax_pmra_corr
    t24 = x.parallax_error*x.pmdec_error*x.parallax_pmdec_corr
    t34 = x.pmra_error*x.pmdec_error*x.pmra_pmdec_corr
    
    Sigma = np.array([[t00, t01, t02, t03, t04],
                      [t01, t11, t12, t13, t14],
                      [t02, t12, t22, t23, t24],
                      [t03, t13, t23, t33, t34],
                      [t04, t14, t24, t34, t44]])

    if no_correlation:
        Sigma = np.array([[t00, 0, 0, 0, 0],
                          [0, t11, 0, 0, 0],
                          [0, 0, t22, 0, 0],
                          [0, 0, 0, t33, 0],
                          [0, 0, 0, 0, t44]])

    sample = np.random.multivariate_normal(mu, Sigma, size= n_sample, check_valid= 'warn')

    ra = sample[:,0]
    dec = sample[:,1]
    parallax = sample[:,2]
    pmra = sample[:,3]
    pmdec = sample[:,4]

    radial_velocity = np.random.normal(loc = x.radial_velocity, scale = x.radial_velocity_error, size = n_sample)
    return(ra,dec,parallax,pmra,pmdec,radial_velocity)

def gaia_hpx_factor(healpix_number = 1):
    """
    returns the number by which to divide the source_id in order to get a hpx number of a specific hpx level
    INPUT:
       healpix_number: the healpix level, ranging from 0 to 12, an integer
    OUTPUT:
       the gaia source id factor to get a specific hpx dicretization
    """
    return(np.power(2,35)*np.power(4,12-healpix_number))

def number_of_healpixels(healpix_number = 1):
    """
    returns the number of pixels for a specific level
    """
    return(np.power(4,healpix_number)*12)


def hpx_density(gdr2_source_id,hpx_level):
    """
    A routine to calculate the number of sources per healpix
    INPUT:
       gdr2_source_id = source ID from Gaia
       hpx_level = the healpix level for which to calculate the starcounts
    OUTPUT:
       density = the starcounts alligned in an array where the index corresponds to the hpx number
    """
    source_id_factor = gaia_hpx_factor(hpx_level)
    n_hpx = np.floor_divide(gdr2_source_id,source_id_factor)
    idx, ct = np.unique(n_hpx, return_counts=True)
    density = np.zeros(number_of_healpixels(hpx_level))
    density[idx] = ct
    density[density==0] = np.nan
    return(density)

def hpx_statistic(gdr2_source_id,statistic_fields,hpx_level):
    """
    A routine to calculate the number of sources per healpix
    INPUT:
       gdr2_source_id = source ID from Gaia
       statistic_fields = the vlaues on which the statistic should be evaluated
       hpx_level = the healpix level for which to calculate the starcounts
    OUTPUT:
       mean = mean of the value per healpix sorted in healpix numbers 
       std = same for std
    """
    source_id_factor = gaia_hpx_factor(hpx_level)
    n_hpx = np.floor_divide(gdr2_source_id,source_id_factor)
    sort = np.argsort(n_hpx)
    n_hpx = n_hpx[sort]
    statistic_fields = statistic_fields[sort]
    index = np.searchsorted(n_hpx,np.arange(number_of_healpixels(hpx_level)))
    mean = np.zeros(number_of_healpixels(hpx_level))
    std = np.zeros(number_of_healpixels(hpx_level))
    for i in range(len(index)):
        if i == len(index)-1:
            mean[i] = np.nanmean(statistic_fields[index[i]:])
            std[i] = np.nanstd(statistic_fields[index[i]:])
        else:
            mean[i] = np.nanmean(statistic_fields[index[i]:index[i+1]])
            std[i] = np.nanstd(statistic_fields[index[i]:index[i+1]])
    return(mean,std)

def hpx_prior_rvs(source_id,hpx_level,rvs,index,n_sample = 100):
    source_id_factor = gaia_hpx_factor(hpx_level)
    n_hpx = np.floor_divide(source_id,source_id_factor)
    sample = rvs[index[n_hpx]:index[n_hpx+1]]
    return(np.random.choice(sample,size=n_sample, replace = True))