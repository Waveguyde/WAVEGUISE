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


def denoise_1d(CWT, white_noise_level=None, sMAD_threshold=None):

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


def wavefield_segmentation_1d(data,prominence,connectivity_order=2):

    work = data.copy()
    assert work.ndim == 2, "Expected 2D array."
    
    iwork  = np.nanmax(work) - work 
    
    mins       = h_minima(iwork, h=prominence)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
        
    return watershed(iwork, markers=markers, connectivity=connectivity_order)


def find_clusters_in_freq(CWT, segments, eps=0.2, min_samples=2):
    
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
    
    return cluster_labels


def build_cluster_map_with_noise(WP_labels: np.ndarray, cluster_labels: np.ndarray):
    """
    Erstellt ein cluster_map, wobei Noise (-1) NICHT zusammengefasst wird,
    sondern jedes Label eine eigene Gruppe bildet.

    Returns
    -------
    dict:
        cluster_id -> list of labels
        für -1: list of single-element lists
    """
    WP_labels = np.asarray(WP_labels)
    cluster_labels = np.asarray(cluster_labels)

    cluster_map = {}

    for lab, cid in zip(WP_labels, cluster_labels):
        if cid == -1:
            # jedes Noise-Label als eigene Gruppe
            cluster_map.setdefault(-1, []).append([lab])
        else:
            cluster_map.setdefault(cid, []).append(lab)

    return cluster_map
    
def get_label_extents(seg: np.ndarray):
    labels = np.unique(seg)
    extents = {}
    for lab in labels:
        yy, xx = np.where(seg == lab)
        extents[lab] = (xx.min(), xx.max())
    return extents

def labels_touch_along_x_only(ext1, ext2):
    x1_min, x1_max = ext1
    x2_min, x2_max = ext2
    return (x1_min <= x2_max) and (x2_min <= x1_max)

def find_connected_groups_along_x(seg: np.ndarray, cluster_labels: np.ndarray):
    """
    labels: Labelwerte, die laut Clustering zusammengehören.
    Zwei Labels werden verbunden, wenn sich ihre x-Intervalle
    berühren oder überlappen.
    """
    cluster_labels = np.asarray(cluster_labels)
    n = len(cluster_labels)

    extents = get_label_extents(seg)

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        li = cluster_labels[i]
        if li not in extents:
            continue

        for j in range(i + 1, n):
            lj = cluster_labels[j]
            if lj not in extents:
                continue

            if labels_touch_along_x_only(extents[li], extents[lj]):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(cluster_labels[i])

    return list(groups.values())

def relabel_by_x_overlap(seg: np.ndarray, cluster_map: dict) -> np.ndarray:
    seg = np.asarray(seg)
    new_seg = np.zeros_like(seg)
    next_label = 1

    for cid, labels in cluster_map.items():

        if cid == -1:
            # hier sind labels schon einzelne Gruppen: [[5], [7], ...]
            for group in labels:
                new_seg[np.isin(seg, group)] = next_label
                next_label += 1
            continue

        # normale Cluster → erst räumlich aufteilen
        groups = find_connected_groups_along_x(seg, labels)

        for group in groups:
            new_seg[np.isin(seg, group)] = next_label
            next_label += 1

    return new_seg


def variance_filter(CWT, segments, var_threshold=0.99):
     
    recon = recon_segments_1d(CWT,segments)
    recon_var = np.var(recon,axis=1)

    # Nach absteigender Varianz sortieren
    order = np.argsort(recon_var)[::-1]
    
    # Kumulative Varianz
    cumsum = np.cumsum(recon_var[order])
    total = cumsum[-1]
    keep = cumsum <= var_threshold * total
    
    # mindestens das stärkste Label behalten
    if not np.any(keep):
        keep[0] = True
    
    keep_idx = order[keep]
    
    # Alte neue Labels -> endgültige kompakte Labels 1..K
    segments_new = np.zeros_like(segments)
    
    for final_label, idx in enumerate(keep_idx, start=1):
        label_to_keep = idx + 1   # weil idx 0-basiert ist, Segmentlabels aber 1-basiert
        segments_new[segments == label_to_keep] = final_label

    return segments_new


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