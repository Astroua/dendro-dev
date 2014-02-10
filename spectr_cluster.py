"""
Introducing the spectral clustering
to dendrograms!
"""

from astropy.io import fits
from astropy.io.fits import getheader
from astrodendro import Dendrogram
import os.path
from readcol import readcol
from pdb import set_trace as stop
import numpy as np
from itertools import combinations
import scipy
from scipy import linalg
from numpy import rank
from matplotlib import pyplot as p
from matplotlib import colors as color


# To compute the matrix kernel
def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


# File list and stuff
path = '/Users/Dario/Documents/dendrograms/'

dendro_file = 'compl_dendrogram'
data_file = 'PerA_Extn2MASS_F_Gal.fits'
load_file = 'compl_dendrogram.fits'

dendro_file = path+'orion_dendrogram'
data_file = path+'orion.fits'
load_file = path+'orion_dendrogram.fits'


# Control flow
do_make = False
do_load = True
do_matrix = False
do_weight = False
do_topview = True


# Make the dendrogram of the full cube/image
if do_make:

    print 'Make dendrogram from the full cube'
    data = fits.getdata(data_file)

    if size(shape(data))==4:
        data = data[0,:,:,:]

    rms = 0.4
    pix_beam = 14
    
    d = Dendrogram.compute(data, min_value=2*rms, min_delta=2*rms, min_npix=10, verbose = 1)

    d.save_to(dendro_file+'.fits')    
    d.viewer()


# Load a premade dendrogram
if do_load:

    data = fits.getdata(data_file)

    if size(shape(data))==4:
        data = data[0,:,:,:]
    
    print 'Load dendrogram file: '+load_file
    d = Dendrogram.load_from(load_file)
    #d.viewer()



# Introduction to the spectral clustering:
# make the necessary matrices and some
# experimental ones.    
if do_matrix:

    #Calculate the adjacency matrix

    s = d.trunk[-1]

    #Preparing the matrices
    
    # Finding a number of nodes
    # the last structure is necessary
    # a leave, then:
    
    num = d.leaves[-1].idx+1
        
    #Adjacency matrix A
    A = zeros((num,num), dtype=np.int)

    #Graph degree matrix GD
    GD = zeros((num,num), dtype=np.int)

    #Antenna temperature distance matrix TD
    TD = zeros((num,num)) 

    #Pixel separation distance matrix SD
    SD = zeros((num,num))

    #Descendant matrix?
    DM = zeros((num,num), dtype = np.int)

                                        
    for i in range(num):

        # Local maxima coordinates
        if len(data.shape)==2:
            xi, yi = d[i].get_peak()[0]
        else:
            xi, yi, vi = d[i].get_peak()[0]


        if do_weight:    

            # Filling the weighted adjacent matrix       
            childs = d[i].children
            if len(childs) > 0:
                GD[i,i] = len(childs)*childs[0].level
        
            for child in childs:

                j = child.idx
                A[i,j] = child.level
                A[j,i] = child.level

        else:
            
            # Filling the adjacent matrix       
            childs = d[i].children
            if len(childs) > 0:
                GD[i,i] = len(childs)
        
            for child in childs:

                j = child.idx
                A[i,j] = 1
                A[j,i] = 1

            
        # Filling the descendant matrix            
        descs = d[i].descendants
        DM[i,i] = len(descs)

        for desc in descs:

            j = desc.idx
            DM[i,j] = 1
            DM[j,i] = 1
                        
        
        
        for j in range(num):

            # TD so far is easy
            TD[i,j] = d[i].height-d[j].height
                        
            # SD is more challenging
            # no need to convert in
            # physical units now
            if len(data.shape)==2:
                xj, yj = d[j].get_peak()[0]
            else:
                xj, yj, vj = d[j].get_peak()[0]

            SD[i,j] = ((xj-xi)**2+(yj-yi)**2)**0.5
                
            
    # Laplacian L = GD - A        
    L = GD - A
    
    # Determine the eigenvectors
    # and eigenvalues for the connectivity

    L_eigval = np.linalg.eigvalsh(L)
    #l_eigvec = np.linalg.eigvh(laplacian)

    # The second lower eigenvalue of L
    # gives the algebraic connectivity
    # of L
    conn = np.sort(L_eigval)[1]

    # The dimension of L kernel gives
    # the number of connected components
    # of A

    L_ker = null(L)

    # To calculate the dim(L_ker) I use
    # the rank theorem:
    # dim(L_Ker) = num column(L) - rk(L)
    # this gives the number of connected
    # component of adjmat

    A_conn_compts = L.shape[1] - rank(L)

    # Now start the spectral clustering...
    

# Attempt to visualize the dendrogram from
# the top as in the spectral clustering    

if do_topview:

    num = d.leaves[-1].idx+1
    xs = zeros(num, dtype = np.int)
    ys = zeros(num, dtype = np.int)
    levs = zeros(num, dtype = np.int)

    p.clf()
    p.axis([0,data.shape[0],0,data.shape[1]])

    for i in range(num):
    
        if len(data.shape)==2:
            xi, yi = d[i].get_peak(subtree=True)[0]
        else:
            xi, yi, vi = d[i].get_peak(subtree=True)[0]

        xs[i] = xi
        ys[i] = yi
        levs[i] = d[i].level 

        # Draw connections between branch and leaves
        if d[i].is_branch:

            childs = d[i].children

            for child in childs:

                if len(data.shape)==2:
                    xj, yj = child.get_peak(subtree=True)[0]
                else:
                    xj, yj, vj = child.get_peak(subtree=True)[0]

                #p.plot([xi,yi],[xj,yj], 'b-')        
                p.plot([xi,xj],[yi,yj], 'k-')


    lev_max = max(levs)            
    lev_min = min(levs)
    lev_range = lev_max - lev_min
    
    for i in range(num):

        xi = xs[i]
        yi = ys[i]
        lev = levs[i]

        # Draw a circle for structure
        # the color corresponds to the level
        col = (lev - lev_min)/float(lev_range)
        color = str(col)

        p.plot(xi,yi,'o')
        

            #struct = scatter(xi, yi, marker='o', c=levs)

            #draw()

    #p.clf()
    #p.axis([0,data.shape[0],0,data.shape[1]])
    #struct = scatter(xs, ys, marker='o', c=colors)
    #draw()

            
            

            

                    
    
        
