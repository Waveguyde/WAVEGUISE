import sys
sys.path.append("/home/r/Robert.Reichert/juwavelet")
import juwavelet.transform as transform
import numpy as np
import copy
from sklearn.cluster import DBSCAN
import scipy.ndimage as ndi
from skimage.measure import label
from skimage.morphology import h_minima
from skimage.segmentation import watershed, relabel_sequential
from numba import njit

def get_basis(x, y, max_order=1):
    #Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


def BG_removal(data, max_order=1):
    
    data-=np.nanmean(data)
    
    ny, nx = data.shape
    y0 = np.arange(ny)
    x0 = np.arange(nx)
    dx, dy = 1, 1

    X, Y = np.meshgrid(x0,y0)
    x, y = X.flatten(), Y.flatten()
    b = data.ravel()
    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    y = y[mask]
    
    basis = get_basis(x, y, max_order)

    A = np.vstack(basis).T
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Calculate the fitted surface from the coefficients, c.
    fit = np.sum(c[:, None, None] * np.array(get_basis(X, Y, max_order)).reshape(len(basis), *X.shape), axis=0)
    
    detrended_data=data-fit
    detrended_data[np.isnan(detrended_data)]=0

    ft = calculate_2dft(detrended_data)
    freqs_x    = np.fft.fftfreq(nx, 1)
    freqs_x    = np.fft.fftshift(freqs_x)
    freqs_y    = np.fft.fftfreq(ny, 1)
    freqs_y    = np.fft.fftshift(freqs_y)
    freqs_X, freqs_Y = np.meshgrid(freqs_x,freqs_y)

    filtered_ft=ft.copy()
    filtered_ft[int(ny/2)-1:int(ny/2)+2,int(nx/2)-1:int(nx/2)+2]=0
    highpass_data=calculate_2dift(filtered_ft)
    lowpass_data=detrended_data-highpass_data
    
    return highpass_data, fit+lowpass_data 


def denoise_2d(CWT, white_noise_level=None, sMAD_threshold=None):

    cwt_copy = copy.deepcopy(CWT)
    dec = cwt_copy['decomposition']

    # --- compute WPS ---
    WAS = np.abs(dec)

    # --- Red noise filtering (robust) ---
    if sMAD_threshold is not None:

        median_WAS = np.median(WAS, axis=(1,2,3), keepdims=True)
        abs_dev    = np.abs(WAS - median_WAS)
        sMAD_WAS   = 1.4826 * np.median(abs_dev, axis=(1,2,3), keepdims=True)

        # avoid division by zero
        sMAD_WAS[sMAD_WAS == 0] = np.finfo(WAS.dtype).eps

        WAS_normed = (WAS - median_WAS) / sMAD_WAS

        sMAD_mask = WAS_normed < sMAD_threshold
        dec[sMAD_mask] = 0

        # update WPS after masking
        WAS = np.abs(dec)

    # --- White noise filtering ---
    if white_noise_level is not None:
        
        # Create a noise WAS as proper reference
        noise = np.random.normal(0,white_noise_level,(dec.shape[2],dec.shape[3]))
        noise_cwt = transform.decompose2d(noise, CWT['dx'], CWT['dy'], CWT['scale'][0], CWT['dj'], CWT['js'], CWT['jt'], aspect=CWT['aspect'], opts=CWT['opts'], mode=CWT['mode'], dtype=np.complex128)
        a, b, noise_WAS = fit_power_law(noise_cwt['scale'],np.mean(np.abs(noise_cwt['decomposition']),axis=(1,2,3)))

        white_mask = WAS < noise_WAS[:,None,None,None]
        dec[white_mask] = 0

    return transform.reconstruct2d(cwt_copy), cwt_copy


def fit_power_law(s, W):
    """
    Fit W = a * s**b to data using a linear fit in log-log space.

    Returns
    -------
    a : float
        Prefactor.
    b : float
        Power-law exponent.
    W_fit : ndarray
        Fitted values at s.
    """
    s = np.asarray(s, dtype=float)
    W = np.asarray(W, dtype=float)

    logs = np.log(s)
    logW = np.log(W)

    b, loga = np.polyfit(logs, logW, deg=1)
    a = np.exp(loga)

    W_fit = a * s**b

    return a, b, W_fit


def _center_slices(orig_shape, periodic_axes):
    center = []
    for ax, n in enumerate(orig_shape):
        if ax in periodic_axes:
            center.append(slice(n, 2*n))  # middle tile along padded axis
        else:
            center.append(slice(0, n))    # unchanged axis
    return tuple(center)

    
def merge_periodic_faces_2D(labels_pad):
    # union-find (minimal)
    parent = {}
    
    def find(x):
        if x == 0: return 0
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
        
    def union(a,b):
        if a==0 or b==0: return
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    # union first and last planes along each periodic axis
    left  = np.take(labels_pad, int(labels_pad.shape[1]/3),  axis=1)
    right = np.take(labels_pad, int(labels_pad.shape[1]*2/3), axis=1)
    pairs = np.stack([left.ravel(), right.ravel()], axis=1)
    pairs = pairs[(pairs[:,0] != 0) & (pairs[:,1] != 0)]
    if pairs.size:
        for a,b in np.unique(pairs, axis=0):
            union(int(a), int(b))

    # apply unions
    uniq = np.unique(labels_pad)
    if uniq.size <= 1:
        return labels_pad
    lut = np.arange(int(uniq.max())+1, dtype=labels_pad.dtype)
    for lbl in uniq:
        if lbl != 0: lut[int(lbl)] = find(int(lbl))
    merged = lut[labels_pad]
    return merged


def wavefield_segmentation_2d(data,prominence,connectivity_order=4):

    assert data.ndim == 4, "Expected 4D array."
        
    iwork  = np.nanmax(data) - data 
    
    pad    = [(0, 0)] * data.ndim
    n      = data.shape[1]
    pad[1] = (n, n)
    
    iwork_pad = np.pad(iwork, pad, mode="wrap")
    
    mins       = h_minima(iwork_pad, h=prominence)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
    labels_pad = watershed(iwork_pad, markers=markers, connectivity=connectivity_order)
    labels     = merge_periodic_faces_2D(labels_pad)
    
    orig_shape    = data.shape
    center        = _center_slices(orig_shape, (1,None))
    center_labels = labels[center].copy()
    
    center_labels, _, _ = relabel_sequential(center_labels)
    
    return center_labels


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def dbscan_periodic_theta(pts, eps=0.25, min_samples=2, theta_period=np.pi, shifts=(0.0, 1.0, -1.0)):
    """
    pts: (N, 2) array-like with columns [logk, theta], theta assumed periodic with period=pi (default).
    Returns: labels for original N points, with duplicates across boundary merged.
    """
    P = np.asarray(pts, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("pts must be an (N, 2) array-like: [logk, theta].")

    N = P.shape[0]
    logk = P[:, 0]
    theta = P[:, 1]

    # Build padded dataset
    aug_pts = []
    aug_orig = []   # which original point index this augmented point came from
    for s in shifts:
        aug_pts.append(np.column_stack([logk, theta + s * theta_period]))
        aug_orig.append(np.arange(N, dtype=int))
    aug_pts = np.vstack(aug_pts)
    aug_orig = np.concatenate(aug_orig)

    # Cluster padded set
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_aug = db.fit_predict(aug_pts)

    # Merge clusters that correspond to the same original point appearing in multiple shifts
    cluster_ids = sorted(set(labels_aug) - {-1})
    uf = UnionFind(cluster_ids)

    # For each original i, union all non-noise cluster labels among its copies
    for i in range(N):
        labs = labels_aug[aug_orig == i]
        labs = [l for l in labs if l != -1]
        if len(labs) >= 2:
            base = labs[0]
            for l in labs[1:]:
                uf.union(base, l)

    # Build a compact relabeling after merging
    root_to_new = {}
    next_id = 0
    for cid in cluster_ids:
        r = uf.find(cid)
        if r not in root_to_new:
            root_to_new[r] = next_id
            next_id += 1

    # Assign final label to each original point:
    # pick any non-noise label from its copies, then map via union-find root -> compact id
    final = np.full(N, -1, dtype=int)
    for i in range(N):
        labs = labels_aug[aug_orig == i]
        labs = [l for l in labs if l != -1]
        if labs:
            r = uf.find(labs[0])
            final[i] = root_to_new[r]

    return final

"""
def find_clusters_in_freq_theta(CWT, segments, eps=0.2, min_samples=2):
    
    old_labels = np.unique(segments)
    old_labels = old_labels[old_labels > 0]
    new_segments = np.zeros_like(segments)

    freq_seg, theta_seg = segments2points(np.abs(CWT['decomposition'])**2,2*np.pi/CWT['period'],CWT['theta'],segments)

    pts = []
    for idx in range(len(old_labels)):
        a = np.log(freq_seg[idx])
        b = theta_seg[idx]
        pts.append([a, b])

    pts = np.asarray(pts)
    cluster_labels = dbscan_periodic_theta(pts, eps=0.25, min_samples=2, theta_period=np.pi)
    
    return cluster_labels
"""

def find_clusters_in_freq_theta(CWT, segments, eps=0.2, min_samples=2):
    
    old_labels = np.unique(segments)
    old_labels = old_labels[old_labels > 0]
    new_segments = np.zeros_like(segments)

    freq_seg, theta_seg = segments2points(np.abs(CWT['decomposition']),2*np.pi/CWT['period'],CWT['theta'],segments)

    pts = []
    for idx in range(len(old_labels)):
        a = np.log(freq_seg[idx])
        b = theta_seg[idx]
        pts.append([a, b])

    pts = np.asarray(pts)
    cluster_labels = dbscan_periodic_theta(pts, eps=0.25, min_samples=2, theta_period=np.pi)
    
    return cluster_labels


def segments2points(WPS, wavelength_x, wavelength_y, segments):
    labels = np.unique(segments)
    labels = labels[labels > 0]

    kx = np.zeros(len(labels), dtype=float)
    ky = np.zeros(len(labels), dtype=float)
    kx0 = wavelength_x[:,None,None,None]
    ky0 = wavelength_y[None,:,None,None]

    for idx, soi in enumerate(labels):
        segmask = (segments == soi)
        weights = WPS * segmask

        wsum = weights.sum()
        if wsum > 0:
            kx[idx] = np.sum(kx0 * weights) / wsum
            ky[idx] = np.sum(ky0 * weights) / wsum

    return kx, ky


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


@njit
def compute_label_extents(seg):
    ny = seg.shape[-2]
    nx = seg.shape[-1]

    max_label = int(seg.max())

    y_min = np.full(max_label + 1, ny, dtype=np.int64)
    y_max = np.full(max_label + 1, -1, dtype=np.int64)
    x_min = np.full(max_label + 1, nx, dtype=np.int64)
    x_max = np.full(max_label + 1, -1, dtype=np.int64)
    exists = np.zeros(max_label + 1, dtype=np.bool_)

    flat = seg.ravel()

    for idx in range(flat.size):
        lab = int(flat[idx])
        if lab <= 0:
            continue

        x = idx % nx
        y = (idx // nx) % ny

        exists[lab] = True

        if y < y_min[lab]:
            y_min[lab] = y
        if y > y_max[lab]:
            y_max[lab] = y
        if x < x_min[lab]:
            x_min[lab] = x
        if x > x_max[lab]:
            x_max[lab] = x

    return y_min, y_max, x_min, x_max, exists


@njit
def intervals_touch_numba(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)


@njit
def labels_touch_numba(
    lab1, lab2,
    y_min, y_max, x_min, x_max
):
    y_touch = intervals_touch_numba(
        y_min[lab1], y_max[lab1],
        y_min[lab2], y_max[lab2]
    )

    x_touch = intervals_touch_numba(
        x_min[lab1], x_max[lab1],
        x_min[lab2], x_max[lab2]
    )

    return y_touch and x_touch


@njit
def find_parent(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit
def union_parent(parent, i, j):
    ri = find_parent(parent, i)
    rj = find_parent(parent, j)

    if ri != rj:
        parent[rj] = ri


@njit
def connected_group_ids_numba(
    labels,
    y_min, y_max, x_min, x_max, exists
):
    n = labels.size
    parent = np.arange(n)

    for i in range(n):
        li = labels[i]

        if li < 0 or li >= exists.size or not exists[li]:
            continue

        for j in range(i + 1, n):
            lj = labels[j]

            if lj < 0 or lj >= exists.size or not exists[lj]:
                continue

            if labels_touch_numba(li, lj, y_min, y_max, x_min, x_max):
                union_parent(parent, i, j)

    for i in range(n):
        parent[i] = find_parent(parent, i)

    return parent


@njit
def apply_label_map(seg, label_map):
    out = np.zeros_like(seg)
    flat_seg = seg.ravel()
    flat_out = out.ravel()

    for i in range(flat_seg.size):
        old = int(flat_seg[i])
        if old > 0 and old < label_map.size:
            flat_out[i] = label_map[old]

    return out


def relabel_by_xy_overlap_numba(seg: np.ndarray, cluster_map: dict) -> np.ndarray:
    seg = np.asarray(seg)

    y_min, y_max, x_min, x_max, exists = compute_label_extents(seg)

    max_label = int(seg.max())
    label_map = np.zeros(max_label + 1, dtype=seg.dtype)

    next_label = 1

    for cid, labels in cluster_map.items():
        labels = np.asarray(labels, dtype=np.int64)

        if cid == -1:
            for lab in labels:
                if 0 <= lab <= max_label and exists[lab]:
                    label_map[lab] = next_label
                    next_label += 1
            continue

        parents = connected_group_ids_numba(
            labels,
            y_min, y_max, x_min, x_max, exists
        )

        root_to_new_label = {}

        for i, lab in enumerate(labels):
            if lab < 0 or lab > max_label or not exists[lab]:
                continue

            root = int(parents[i])

            if root not in root_to_new_label:
                root_to_new_label[root] = next_label
                next_label += 1

            label_map[lab] = root_to_new_label[root]

    return apply_label_map(seg, label_map)


def variance_filter(CWT, segments, var_threshold=0.99):
     
    recon = recon_all_2d(CWT,segments)
    recon_var = np.var(recon,axis=(1,2))

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


def recon_all_2d(cwt_dict,segments):

    labels = np.unique(segments)
    mask   = labels > 0
    labels = labels[mask]

    dim    = cwt_dict['decomposition'].shape
    decomp = cwt_dict['decomposition']
    recon  = np.zeros((len(labels),dim[2],dim[3]))
    
    for soi in labels:
        mask   = (segments != soi)
        backup = decomp[mask].copy()
        decomp[mask] = 0
        recon[soi-1,:,:] = transform.reconstruct2d(cwt_dict)
        decomp[mask] = backup

    return recon


def recon_allWP_2d(cwt_dict,segments):

    import sys
    sys.path.append('/home/r/Robert.Reichert/juwavelet')
    import juwavelet.transform as transform
    import itertools
    import tqdm
    
    labels = np.unique(segments)
    mask   = labels > 0
    labels = labels[mask]
    
    dim    = cwt_dict['decomposition'].shape
    decomp = cwt_dict['decomposition']
    recon  = np.zeros((len(labels),dim[2],dim[3]))
    amp    = np.zeros((len(labels),dim[2],dim[3]))
    kx     = np.zeros((len(labels),dim[2],dim[3]))
    ky     = np.zeros((len(labels),dim[2],dim[3]))
    T, P   = np.meshgrid(cwt_dict['theta'],cwt_dict['period'])
    kx0    = 2*np.pi/P*np.sin(T)
    ky0    = 2*np.pi/P*np.cos(T)
    
    for soi in labels:
        mask   = (segments != soi)
        backup = decomp[mask].copy()
        decomp[mask] = 0
        recon[soi-1,:,:] = transform.reconstruct2d(cwt_dict)
        
        for i, j in tqdm.tqdm(itertools.product(range(dim[2]), range(dim[3])),total=dim[2]*dim[3]):
    
            weights = np.abs(decomp[:,:,i,j])
            if np.nansum(weights) == 0:
                continue
            else:
                amp[soi-1,i,j] = np.nanmax(weights)
                kx[soi-1,i,j]  = np.average(kx0,weights=weights)
                ky[soi-1,i,j]  = np.average(ky0,weights=weights)
        decomp[mask] = backup
    
    return recon, amp, kx, ky


def recon_WPoi_2d(cwt_dict,all_segments,segments_of_interest):

    import sys
    sys.path.append('/home/r/Robert.Reichert/juwavelet')
    import juwavelet.transform as transform
    import itertools
    import tqdm
    
    labels = np.unique(all_segments)
    mask   = labels > 0
    labels = labels[mask]
    
    dim    = cwt_dict['decomposition'].shape
    decomp = cwt_dict['decomposition']
    recon  = np.zeros((dim[2],dim[3]))
    amp    = np.zeros((dim[2],dim[3]))
    kx     = np.zeros((dim[2],dim[3]))
    ky     = np.zeros((dim[2],dim[3]))
    T, P   = np.meshgrid(cwt_dict['theta'],cwt_dict['period'])
    kx0    = 2*np.pi/P*np.sin(T)
    ky0    = 2*np.pi/P*np.cos(T)
    
    mask   = np.isin(all_segments, segments_of_interest)
    backup = decomp[~mask].copy()
    decomp[~mask] = 0
    recon = transform.reconstruct2d(cwt_dict)
    
    for i, j in tqdm.tqdm(itertools.product(range(dim[2]), range(dim[3])),total=dim[2]*dim[3]):

        weights = np.abs(decomp[:,:,i,j])
        if np.nansum(weights) == 0:
            continue
        else:
            amp[i,j] = np.nanmax(weights)
            kx[i,j]  = np.average(kx0,weights=weights)
            ky[i,j]  = np.average(ky0,weights=weights)
    decomp[~mask] = backup
    
    return recon, amp, kx, ky


def A_kx_ky(list_of_labels,CWT,segments,mode='RR'):
    """
    Computes the wave paket properties such as wavelength, propagation direction and amplitude 
    as function of x and y based on the CWT and the labelled region within the WPS.

    Parameters
    ----------
    list_of_labels : list of ints
        label(s) of the wavepaket which properties should be computed.
    CWT : dict
        dictionary containing among other things the wavelet coefficients.
        dictionary is provided by juwavelet.transform.decompose()
    segments : array of ints
        an array of the same dimensions as the CWT['decomposition'] entry that marks clusters in the WPS
        and hence marks individual wave pakets.
    mode: either JU or RR
        JU insists on returning the kx and ky associated with the location of maximum amplitude.
        RR rather uses a amplitude weighted mean to return kx and ky.

    Returns
    -------
    3 x ndarrays
    """

    import itertools
    import tqdm

    dim = CWT['decomposition'].shape

    A  = np.zeros(dim[2:4])
    kx = np.zeros(dim[2:4])
    ky = np.zeros(dim[2:4])

    T, P = np.meshgrid(CWT['theta'],CWT['period'])
    
    for i, j in tqdm.tqdm(list(itertools.product(range(dim[2]), range(dim[3])))):
        mask = np.isin(segments[:,:,i,j], list_of_labels)
        if np.count_nonzero(mask) > 0:
            weights = np.abs(CWT["decomposition"][:,:,i,j]) ** 2
            if np.sum(weights[mask]) > 0:
                A[i,j] = np.sqrt(np.nanmax(weights[mask]))
                if mode == 'JU':
                    true_indices = np.argwhere(mask)
                    max_index = np.argmax(weights[mask])  
                    true_max_index = true_indices[max_index]  
                    kx[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.sin(T[true_max_index[0], true_max_index[1]])
                    ky[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.cos(T[true_max_index[0], true_max_index[1]])
                if mode == 'RR':
                    kx0 = 2*np.pi/P*np.sin(T)
                    ky0 = 2*np.pi/P*np.cos(T)
                    kx[i, j] = np.average(kx0[mask],weights=weights[mask])
                    ky[i, j] = np.average(ky0[mask],weights=weights[mask])
                    
    return A, kx, ky

def kxky_2_lhtheta(kx,ky):
    """
    Convert the wave vector components into a wavelength and an orientation
    Keep in mind that arctan2() returns signed angles between [-np.pi,np.pi] defined from the positive x-axis
    while I defined my angles from [0,2*np.pi] going clockwise from the positive y-axis.
    """
    k     = np.sqrt(kx**2+ky**2)
    theta = np.arctan2(ky, kx)
    theta = np.pi/2-theta
    theta[theta<0] = 2*np.pi + theta[theta<0]
    
    return 2*np.pi/k, theta