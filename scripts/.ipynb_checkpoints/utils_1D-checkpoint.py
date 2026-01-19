import sys
sys.path.append('/home/r/Robert.Reichert/juwavelet')
import juwavelet.transform as transform

import copy
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import h_minima

def plot_COI(x,order,ax,**kwargs):
    x2 = x-x[0]
    coi_x = np.minimum(x2,x2[-1]-x2)
    coi_boundary_scale = coi_x / np.sqrt(2)
    coi_boundary_wavelength = 4*np.pi*coi_boundary_scale/(order+np.sqrt(2+order**2))
    ax.plot(x, coi_boundary_wavelength, c='k')
    ax.fill_between(x, coi_boundary_wavelength, **kwargs)


def wavefield_segmentation_1d(data,threshold,connectivity_order=2):

    work = data.copy()
    assert work.ndim == 2, "Expected 2D array."
    
    iwork  = np.nanmax(work) - work 
    
    mins       = h_minima(iwork, h=threshold)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
        
    return watershed(iwork, markers=markers, connectivity=connectivity_order)


def update_segments(WPS, segments, threshold=0.95, mode='max'):
    # unique region labels, excluding 0 (often background)
    labels = np.unique(segments)
    labels = labels[labels != 0]

    # compute power per segment
    if mode == 'max':
        segment_power = np.array([np.max(WPS[segments == l]) for l in labels])
    if mode == 'mean':
        segment_power = np.array([np.mean(WPS[segments == l]) for l in labels])
    if mode == 'median':
        segment_power = np.array([np.median(WPS[segments == l]) for l in labels])
    if mode == 'sum':
        segment_power = np.array([np.sum(WPS[segments == l]) for l in labels])

    # sort by descending power
    order = np.argsort(segment_power)[::-1]

    # cumulative sum and keep only those within threshold fraction
    cumsum = np.cumsum(segment_power[order])
    keep = cumsum <= threshold * cumsum[-1]

    # map old labels to new sorted ones
    new_segments = np.zeros_like(segments)
    for new_label, idx in enumerate(order[keep], start=1):
        label_to_keep = labels[idx]
        new_segments[segments == label_to_keep] = new_label

    return new_segments


def recon_segments_1d(cwt_dict,segments,x):
    recon = np.zeros((np.max(segments),len(x)))
    for soi in np.unique(segments)[1:]:
        wps  = copy.deepcopy(cwt_dict)
        mask = (segments != soi)
        wps["decomposition"][mask] = 0
        recon[soi-1,:] = transform.reconstruct1d(wps)

    return recon


def A_kx(list_of_labels,CWT,segments):

    dim = CWT['decomposition'].shape

    A  = np.zeros(dim[1])
    kx = np.zeros(dim[1])

    P = CWT['period']
    
    for i in range(dim[1]):
        mask = np.isin(segments[:,i], list_of_labels)
        if np.count_nonzero(mask) > 0:
            weights = np.abs(CWT["decomposition"][:,i]) ** 2
            if np.sum(weights[mask]) > 0:
                A[i] = np.sqrt(np.nanmax(weights[mask]))
                true_indices = np.argwhere(mask)
                max_index = np.argmax(weights[mask])  
                true_max_index = true_indices[max_index]  
                kx[i] = 2*np.pi/P[true_max_index[0]]
                    
    return A, kx