import numpy as np
import matplotlib.pylab as plt


def plot_hrd(data,title_append,color_data,color_label):
    abs_g = data.phot_g_mean_mag + 5 * np.log10(np.divide(data.parallax,100))
    color = data.phot_bp_mean_mag - data.phot_rp_mean_mag
    plt.figure(figsize = (20,10))
    ax = plt.subplot(111)
    im = ax.scatter(color,abs_g,alpha = 1,s = 7,c = color_data)
    plt.colorbar(im, label = color_label)
    plt.gca().invert_yaxis()
    plt.ylabel(r"G $+ 5\log_{10}\left(\varpi/100\right)$")
    plt.xlabel("BP-RP [mag]")
    plt.title("CMD (%d stars) %s" %(len(data),title_append))
    plt.show()
    plt.clf()
    plt.close() 
    
# A routine to make a scatter plot in mollweide projection
def plot_mollweide(gl,gb,color_data,color_label,  title):
    gl[gl>180] -=360  
    plt.figure(figsize = (20,10))
    ax = plt.subplot(111, projection = 'mollweide')
    im = ax.scatter(np.radians(gl),np.radians(gb),alpha = 1,s = 7,c = color_data)
    plt.title('%s (%d stars)' %(title,len(gl)))
    plt.colorbar(im, label = color_label)
    plt.show()
    plt.clf()
    plt.close()
    
def plot_encounter(data,title_append,color_data,color_label,xlim = None, ylim = None):
    plt.figure(figsize = (20,10))
    ax = plt.subplot(111)
    im = ax.scatter(data.lma_min_time,data.lma_min_distance,alpha = 1,s = 7,c = color_data)
    plt.colorbar(im, label = color_label)
    plt.ylabel("distance [pc]")
    plt.xlabel("time [Myr]")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("Encounter diagramm (%d stars) %s" %(len(data),title_append))
    plt.show()
    plt.clf()
    plt.close()

def plot_lma_vs_orbit(data,xlim=None,ylim=None):
    plt.figure(figsize = (20,10))
    plt.plot(data.lma_min_time,data.lma_min_distance,'.r',label = 'lma')
    plt.plot(data.orbit_min_time,data.orbit_min_distance,'.b', label = 'orbit')
    for i in range(len(data)):
        plt.plot([data[i].orbit_min_time,data[i].lma_min_time], [data[i].orbit_min_distance,data[i].lma_min_distance],'-k', alpha = 0.2)
    plt.legend()
    plt.ylabel('ph dist [pc]')
    plt.xlabel('ph time [Myr]')
    plt.title('lma vs orbit integration')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
    plt.clf()
    
def plot_comparison_orbit(data,mock,xlim=None,ylim=None):
    plt.figure(figsize=(20,10))
    plt.plot(mock.orbit_min_time,mock.orbit_min_distance,'.r', label = 'mock (%d)'%(len(mock)))
    plt.plot(data.orbit_min_time,data.orbit_min_distance,'.b', label = 'data (%d)'%(len(data)))
    plt.legend()
    plt.ylabel('ph dist [pc]')
    plt.xlabel('ph time [Myr]')
    plt.title('orbit integration - mock vs real')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
    plt.clf()
    

def plot_mollweide_log(data):
    import healpy as hp
    from matplotlib.colors import LogNorm
    import matplotlib.pylab as plt
    norm = LogNorm()
    map_mollweide = data        
    total = np.nansum(map_mollweide)
    hp.mollview(map_mollweide, cbar = True, min=None, max=None, nest = True,norm = norm, coord= "CG", unit = 'starcount per hpx',notext =True)
    plt.title("total starcount=%d" %(total))        
    plt.show()