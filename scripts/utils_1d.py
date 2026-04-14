import sys
sys.path.append('/home/r/Robert.Reichert/juwavelet')
import juwavelet.transform as transform

import copy
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import h_minima
from sklearn.cluster import DBSCAN
from scipy import stats

def get_basis(x, max_order=1):
    #Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    basis = []
    for i in range(max_order+1):
        basis.append(x**i)
    return basis

def calculate_1dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft(ft)
    return np.fft.fftshift(ft)

def calculate_1dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def BG_removal(data, max_order=1):

    data-=np.nanmean(data)
    
    nx = data.shape[0]
    x  = np.arange(nx)
    dx = 1
    
    b = data
    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    
    basis = get_basis(x, max_order)
    
    A = np.vstack(basis).T
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    fit = np.sum(c[:, None] * basis, axis=0)
    
    detrended_data=data-fit
    detrended_data[np.isnan(detrended_data)]=0
    
    ft = calculate_1dft(detrended_data)
    freqs_x    = np.fft.fftfreq(nx, 1)
    freqs_x    = np.fft.fftshift(freqs_x)
    
    filtered_ft=ft.copy()
    filtered_ft[int(nx/2)-1:int(nx/2)+2]=0
    highpass_data=calculate_1dift(filtered_ft)
    lowpass_data=detrended_data-highpass_data

    return highpass_data, fit+lowpass_data 


def noise_filtering_1d(CWT, white_noise_level=None, sMAD_threshold=None):

    cwt_copy = copy.deepcopy(CWT)
    dec = cwt_copy['decomposition']

    # --- compute WPS ---
    WPS = np.abs(dec)**2

    # --- White noise filtering ---
    if white_noise_level is not None:
        white_mask = WPS < white_noise_level**2
        dec[white_mask] = 0

        # update WPS after masking
        WPS = np.abs(dec)**2

    # --- Red noise filtering (robust) ---
    if sMAD_threshold is not None:

        median_WPS = np.median(WPS, axis=1, keepdims=True)
        sMAD_WPS   = 1.4826 * stats.median_abs_deviation(WPS, axis=1, keepdims=True)

        # avoid division by zero
        sMAD_WPS[sMAD_WPS == 0] = np.finfo(WPS.dtype).eps

        WPS_normed = (WPS - median_WPS) / sMAD_WPS

        sMAD_mask = WPS_normed < sMAD_threshold
        dec[sMAD_mask] = 0

    return transform.reconstruct1d(cwt_copy), cwt_copy


def wavefield_segmentation_1d(data,threshold,connectivity_order=2):

    work = data.copy()
    assert work.ndim == 2, "Expected 2D array."
    
    iwork  = np.nanmax(work) - work 
    
    mins       = h_minima(iwork, h=threshold)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
        
    return watershed(iwork, markers=markers, connectivity=connectivity_order)

"""
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
"""

def recon_WP_and_properties_1d(cwt_dict,segments):

    labels = np.unique(segments)
    mask   = labels > 0
    labels = labels[mask]

    dim    = cwt_dict['decomposition'].shape
    decomp = cwt_dict['decomposition']
    recon  = np.zeros((len(labels),dim[1]))
    amp    = np.zeros((len(labels),dim[1]))
    freq   = np.zeros((len(labels),dim[1]))
    P      = cwt_dict['period']
    
    for soi in labels:
        mask   = (segments != soi)
        backup = decomp[mask].copy()
        decomp[mask] = 0
        recon[soi-1,:] = transform.reconstruct1d(cwt_dict)

        for i in range(dim[1]):
            weights = np.abs(decomp[:,i]) ** 2
            if np.nansum(weights) == 0:
                continue
            else:
                amp[soi-1,i] = np.sqrt(np.nanmax(weights))
                freq[soi-1,i]= 2*np.pi/np.average(P,weights=weights)
        decomp[mask] = backup

    return recon, amp, freq

def recon_segments_1d(cwt_dict,segments):

    labels = np.unique(segments)
    mask   = labels > 0
    labels = labels[mask]

    dim    = cwt_dict['decomposition'].shape
    decomp = cwt_dict['decomposition']
    recon  = np.zeros((len(labels),dim[1]))
    
    for soi in labels:
        mask   = (segments != soi)
        backup = decomp[mask].copy()
        decomp[mask] = 0
        recon[soi-1,:] = transform.reconstruct1d(cwt_dict)
        decomp[mask] = backup

    return recon


def segments2points(WPS, wavelength_x, segments):
    labels = np.unique(segments)
    labels = labels[labels > 0]

    kx = np.zeros(len(labels), dtype=float)
    kx0 = wavelength_x[:,None]

    for idx, soi in enumerate(labels):
        segmask = (segments == soi)
        weights = WPS * segmask

        wsum = weights.sum()
        if wsum > 0:
            kx[idx] = np.sum(kx0 * weights) / wsum

    return kx


def update_segments_v2(CWT, segments, eps=0.2, min_samples=2, threshold=0.99):
    old_labels = np.unique(segments)
    old_labels = old_labels[old_labels > 0]

    new_segments = np.zeros_like(segments)

    freq_seg = segments2points(np.abs(CWT['decomposition'])**2,CWT['period'],segments)

    pts = []
    for idx in range(len(old_labels)):
        a = np.log(freq_seg[idx])
        b = 0.0
        pts.append([a, b])

    pts = np.asarray(pts)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(pts)
    print('Clustering done!')

    next_label = 1
    cluster_map = {}
    old_to_new = {}

    for old_label, clus in zip(old_labels, cluster_labels):
        if clus == -1:
            # Jeder Noise bekommt ein eigenes neues Label
            new_label = next_label
            next_label += 1
        else:
            # Alle Segmente im selben Cluster bekommen dasselbe neue Label
            if clus not in cluster_map:
                cluster_map[clus] = next_label
                next_label += 1
            new_label = cluster_map[clus]

        old_to_new[old_label] = new_label
        new_segments[segments == old_label] = new_label

    recon_new = recon_segments_1d(CWT,new_segments)

    # recon_seg entsprechend der neuen Labels aufsummieren
    #n_new_labels = next_label - 1
    #recon_new = np.zeros((n_new_labels, recon_seg.shape[1]), dtype=recon_seg.dtype)

    #for old_label, new_label in old_to_new.items():
    #    old_idx = old_label - 1   # nur korrekt, wenn recon_seg zu Labels 1..N gehört
    #    new_idx = new_label - 1
    #    recon_new[new_idx, :] += recon_seg[old_idx, :]

    # Varianz pro neuem Label
    recon_var = np.var(recon_new, axis=1)

    # Nach absteigender Varianz sortieren
    order = np.argsort(recon_var)[::-1]

    # Kumulative Varianz
    cumsum = np.cumsum(recon_var[order])
    total = cumsum[-1]

    if total == 0:
        keep_idx = order
    else:
        keep = cumsum <= threshold * total

        # mindestens das stärkste Label behalten
        if not np.any(keep):
            keep[0] = True

        keep_idx = order[keep]

    # Alte neue Labels -> endgültige kompakte Labels 1..K
    new_new_segments = np.zeros_like(new_segments)

    for final_label, idx in enumerate(keep_idx, start=1):
        label_to_keep = idx + 1   # weil idx 0-basiert ist, Segmentlabels aber 1-basiert
        new_new_segments[new_segments == label_to_keep] = final_label

    return new_new_segments
    

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
                kx[i] = 2*np.pi/np.average(P[mask],weights=weights[mask])
                    
    return A, kx