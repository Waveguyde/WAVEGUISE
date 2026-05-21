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
import itertools
import tqdm

# BACKROUND REMOVAL # ------------------------------------------------------------

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

def BG_removal(data, max_order=1, fourier_radius=1):

    """
    Removes the mean and a polynomial of degree max_order and Fourier components of fourier_radius.

    Returns
    -------
    highpass_data : array of data.size containing the high_frequency components
    background : array of data.size containing the (low-frequency) background
    """

    data = np.asarray(data, dtype=float)
    
    if data.ndim != 2:
        raise ValueError("data must be a 2D-Array.")

    data = data.copy()
    # Subtract mean
    data-=np.nanmean(data)

    # Determine the polynomial fit of degree max_order
    ny, nx = data.shape
    y0 = np.arange(ny)
    x0 = np.arange(nx)

    Y, X = np.meshgrid(y0, x0, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    b = data.ravel()
    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    y = y[mask]
    
    basis = get_basis(x, y, max_order)

    A = np.vstack(basis).T
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
    full_basis = np.array(get_basis(X, Y, max_order))
    fit = np.sum(c[:, None, None] * full_basis, axis=0)

    # Detrended data is de-meaned data minus the polynomial fit (The implementation copes with Nan values which are simply set to zero in the subsequent FFT step)
    detrended_data = data - fit
    detrended_data[np.isnan(detrended_data)]=0

    # Compute the 2D FFT and filter according to the fourier_radius
    ft = calculate_2dft(detrended_data)
    filtered_ft = ft.copy()
    cy, cx = ny // 2, nx // 2
    r = fourier_radius
    filtered_ft[
        max(0, cy - r):min(ny, cy + r + 1),
        max(0, cx - r):min(nx, cx + r + 1)
    ] = 0
    highpass_data = calculate_2dift(filtered_ft)
    lowpass_data  = detrended_data - highpass_data
    background    = fit + lowpass_data
    
    return highpass_data, background

# ------------------------------------ # BACKGROUND REMOVAL # -------------------------------------------------------





# ------------------------------------ # DENOISING # ----------------------------------------------------------------

def denoise_2d(CWT, white_noise_level=None, sMAD_threshold=None):

    """
    Removes white noise and/or noise that is scale dependent based on the Wavelet Amplitude Spectrum (WAS)
    
    Returns
    -------
    wavy_stuff : array of signal.size containing the denoised data
    """

    cwt_copy = copy.deepcopy(CWT)
    dec = cwt_copy['decomposition']

    # --- compute WAS ---
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

        # update WAS after masking
        WAS = np.abs(dec)

    # --- White noise filtering ---
    """
    A WAS is computed from white noise with the defined standard deviation. The WAS is then averaged resulting in a purely scale-dependent WAS
    """
    if white_noise_level is not None:
        
        # Create a noise WAS as proper reference
        noise = np.random.normal(0,white_noise_level,(dec.shape[2],dec.shape[3]))
        noise_cwt = transform.decompose2d(noise, CWT['dx'], CWT['dy'], CWT['scale'][0], CWT['dj'], CWT['js'], CWT['jt'], aspect=CWT['aspect'], opts=CWT['opts'], mode=CWT['mode'], dtype=np.complex128)
        noise_WAS = fit_power_law(noise_cwt['scale'],np.mean(np.abs(noise_cwt['decomposition']),axis=(1,2,3)))

        white_mask = WAS < noise_WAS[:,None,None,None]
        dec[white_mask] = 0

    return transform.reconstruct2d(cwt_copy)

def fit_power_law(s, W):
    """
    Fit W = a * s**b to data using a linear fit in log-log space.

    Returns
    -------
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

    return W_fit

# -------------------------------------------------- # DENOISING # -------------------------------------------------------------------





# -------------------------------------------------- # SEGMENTATION # -----------------------------------------------------------------

def norm_WAS(WAS):
    
    median_WAS = np.median(WAS, axis=(1,2,3), keepdims=True)
    abs_dev    = np.abs(WAS - median_WAS)
    sMAD_WAS   = 1.4826 * np.median(abs_dev, axis=(1,2,3), keepdims=True)
    sMAD_WAS[sMAD_WAS == 0] = np.finfo(WAS.dtype).eps
    WAS_normed = (WAS - median_WAS) / sMAD_WAS
    
    return WAS_normed

def wavefield_segmentation_2d(data, prominence, connectivity=4, periodic_axis=1, dtype=np.float32):
    
    assert data.ndim == 4, "Expected 4D array."

    work = np.asarray(data, dtype=dtype)

    iwork = np.nanmax(work) - work
    signal_mask = work > prominence

    mins = h_minima(iwork, h=prominence)
    mins &= signal_mask

    structure = ndi.generate_binary_structure(work.ndim, connectivity)

    markers, n_markers = ndi.label(mins, structure=structure)

    if n_markers == 0:
        return np.zeros(data.shape, dtype=np.int32)

    labels = watershed(iwork, markers=markers, connectivity=structure, mask=signal_mask)
    labels = merge_periodic_faces(labels, periodic_axis=periodic_axis)
    labels, _, _ = relabel_sequential(labels)

    return labels.astype(np.int32, copy=False)

def merge_periodic_faces(labels, periodic_axis=1):
    parent = {}

    def find(x):
        if x == 0:
            return 0
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a == 0 or b == 0:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    left  = np.take(labels, 0,  axis=periodic_axis)
    right = np.take(labels, -1, axis=periodic_axis)

    mask = (left != 0) & (right != 0) & (left != right)

    if np.any(mask):
        pairs = np.stack(
            [
                left[mask].astype(np.int64, copy=False),
                right[mask].astype(np.int64, copy=False),
            ],
            axis=1,
        )
        pairs = np.unique(pairs, axis=0)

        for a, b in pairs:
            union(int(a), int(b))

    uniq = np.unique(labels)

    if uniq.size <= 1:
        return labels

    lut = np.arange(int(uniq.max()) + 1, dtype=labels.dtype)

    for lbl in uniq:
        if lbl != 0:
            lut[int(lbl)] = find(int(lbl))

    return lut[labels]

# ------------------------------------------ # SEGMENTATION # ------------------------------------------------------





# ----------------------------------------- # CLUSTERING # --------------------------------------------------------

def find_clusters_logk_theta(CWT, segments, eps=0.2, min_samples=2):
    WAS   = np.abs(CWT["decomposition"])
    k     = 2 * np.pi / CWT["period"]
    theta = CWT["theta"]

    labels, amps_seg, kx_seg, ky_seg, k_seg, theta_seg = segments2points_pi_periodic(WAS, k, theta, segments)
    #mask = kx_seg < 0
    #if np.sum(mask) > 0:
    #    kx_seg[mask]*=-1
    #    ky_seg[mask]*=-1
        
    #k_seg = np.sqrt(kx_seg**2+ky_seg**2)
    #theta_seg = np.arctan2(ky_seg, kx_seg)
    #theta_seg = np.pi/2-theta_seg
    #theta_seg[theta_seg<0] = 2*np.pi + theta_seg[theta_seg<0]
    
    ds = np.abs(np.diff(np.log(k))[0])
    dt = np.diff(theta)[0]
    alpha=ds/dt
    print('suggested scaling:', alpha)

    pts = np.column_stack([np.log(k_seg),alpha*theta_seg])

    cluster_labels = dbscan_periodic_theta(pts, eps=eps, min_samples=min_samples, theta_period=np.pi)

    return labels, cluster_labels 

"""
@njit
def segments2points(WAS, k, theta, segments):
    max_label = int(segments.max())

    sum_w  = np.zeros(max_label + 1, dtype=np.float64)
    sum_kx = np.zeros(max_label + 1, dtype=np.float64)
    sum_ky = np.zeros(max_label + 1, dtype=np.float64)

    n0, n1, n2, n3 = segments.shape

    for i0 in range(n0):
        for i1 in range(n1):
            kx_val = k[i0]*np.sin(2*theta[i1])
            ky_val = k[i0]*np.cos(2*theta[i1])
            for i2 in range(n2):
                for i3 in range(n3):
                    lab = int(segments[i0, i1, i2, i3])

                    if lab <= 0:
                        continue

                    w = WAS[i0, i1, i2, i3]

                    sum_w[lab] += w
                    sum_kx[lab] += kx_val * w
                    sum_ky[lab] += ky_val * w

    labels = []
    amps = []
    kx = []
    ky = []

    for lab in range(1, max_label + 1):
        if sum_w[lab] > 0:
            labels.append(lab)
            kx.append(sum_kx[lab] / sum_w[lab])
            ky.append(sum_ky[lab] / sum_w[lab])
            amps.append(sum_w[lab]) 

    return (
        np.asarray(labels),
        np.asarray(amps),
        np.asarray(kx),
        np.asarray(ky),
    )
"""

@njit
def segments2points_pi_periodic(WAS, k, theta, segments):
    max_label = int(segments.max())

    sum_w = np.zeros(max_label + 1, dtype=np.float64)
    sum_k = np.zeros(max_label + 1, dtype=np.float64)
    sum_sin2 = np.zeros(max_label + 1, dtype=np.float64)
    sum_cos2 = np.zeros(max_label + 1, dtype=np.float64)

    n0, n1, n2, n3 = segments.shape

    for i0 in range(n0):
        k_val = k[i0]

        for i1 in range(n1):
            sin2 = np.sin(2 * theta[i1])
            cos2 = np.cos(2 * theta[i1])

            for i2 in range(n2):
                for i3 in range(n3):
                    lab = int(segments[i0, i1, i2, i3])

                    if lab <= 0:
                        continue

                    w = WAS[i0, i1, i2, i3]

                    sum_w[lab] += w
                    sum_k[lab] += k_val * w
                    sum_sin2[lab] += sin2 * w
                    sum_cos2[lab] += cos2 * w

    labels = []
    amps = []
    k_mean_arr = []
    theta_mean_arr = []
    kx_arr = []
    ky_arr = []

    for lab in range(1, max_label + 1):
        if sum_w[lab] > 0:
            mean_k = sum_k[lab] / sum_w[lab]

            mean_sin2 = sum_sin2[lab] / sum_w[lab]
            mean_cos2 = sum_cos2[lab] / sum_w[lab]

            theta_mean = 0.5 * np.arctan2(mean_sin2, mean_cos2)

            kx_mean = mean_k * np.sin(theta_mean)
            ky_mean = mean_k * np.cos(theta_mean)

            # Positiven Halbraum erzwingen, falls gewünscht
            if kx_mean < 0:
                kx_mean *= -1
                ky_mean *= -1
                theta_mean += np.pi

            theta_mean = theta_mean % np.pi

            labels.append(lab)
            amps.append(sum_w[lab])
            k_mean_arr.append(mean_k)
            theta_mean_arr.append(theta_mean)
            kx_arr.append(kx_mean)
            ky_arr.append(ky_mean)

    return (
        np.asarray(labels),
        np.asarray(amps),
        np.asarray(kx_arr),
        np.asarray(ky_arr),
        np.asarray(k_mean_arr),
        np.asarray(theta_mean_arr),
    )

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
            
def dbscan_periodic_theta(pts, eps=0.2, min_samples=2, theta_period=np.pi, shifts=(0.0, 1.0, -1.0)):
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

def relabel_by_xy_overlap(seg: np.ndarray, cluster_map: dict) -> np.ndarray:
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
def apply_label_map(seg, label_map):
    out = np.zeros_like(seg)
    flat_seg = seg.ravel()
    flat_out = out.ravel()

    for i in range(flat_seg.size):
        old = int(flat_seg[i])
        if old > 0 and old < label_map.size:
            flat_out[i] = label_map[old]

    return out

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

# ---------------------------------------------- # CLUSTERING # ------------------------------------------------------------------





# --------------------------------------------- # RECONSTRUCTION # -------------------------------------------------------------

def recon_allWP_2d(cwt_dict,segments):
    
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