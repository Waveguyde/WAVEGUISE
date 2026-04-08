import sys
sys.path.append("/home/r/Robert.Reichert/juwavelet")
import juwavelet.transform as transform
import numpy as np
import copy
from scipy import stats

def define_figgrid(N):
    delta=[]
    for i in range(1,N):
        j = np.ceil(N/i)
        delta = np.append(delta,abs(j-i))
    
    return int(np.argmin(delta)+1), int(np.ceil(N/(np.argmin(delta)+1)))

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
    y0 = np.arange(ny)#linspace(0, (len(signal[:, 0]) - 1) * 0.625, ny)
    x0 = np.arange(nx)#linspace(0, (len(signal[0, :]) - 1) * 0.625, nx)
    dx, dy = 1, 1#(x0[1] - x0[0]), (y0[1] - y0[0])

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

    import scipy.ndimage as ndi
    from skimage.measure import label
    from skimage.morphology import h_minima
    from skimage.segmentation import watershed, relabel_sequential

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


def recon_segments_2d_v2(cwt_dict,segments):

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
    
            weights = np.abs(decomp[:,:,i,j]) ** 2
            if np.nansum(weights) == 0:
                continue
            else:
                amp[soi-1,i,j] = np.sqrt(np.nanmax(weights))
                kx[soi-1,i,j]  = np.average(kx0,weights=weights)
                ky[soi-1,i,j]  = np.average(ky0,weights=weights)
        decomp[mask] = backup
    
    return recon, amp, kx, ky


def recon_segments_2d(cwt_dict,segments,x,y):

    import sys
    sys.path.append('/home/r/Robert.Reichert/juwavelet')
    import juwavelet.transform as transform

    recon = np.zeros((np.max(segments),len(x),len(y)))
    for soi in np.unique(segments):
        wps  = copy.deepcopy(cwt_dict)
        mask = (segments != soi)
        wps["decomposition"][mask] = 0
        recon[soi-1,:,:] = transform.reconstruct2d(wps)

    var = np.var(recon,axis=(1,2))
    sorted_indices = np.argsort(var)[::-1]

    return recon[sorted_indices,:,:]

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


def red_noise_filtering(CWT,white_noise_level,red_noise_level=8):
    cwt_copy   = copy.deepcopy(CWT)
    # Compute the wavelet power spectrum
    WPS        = np.abs(cwt_copy['decomposition'])**2
    # Set wavelet coefficients to zero where we expect to find white noise
    white_mask = WPS < white_noise_level**2
    cwt_copy['decomposition'][white_mask] = 0
    WPS        = np.abs(cwt_copy['decomposition'])**2
    # Compute med(WPS) and sMAD(WPS) for each scale independently!
    median_WPS = np.median(WPS,axis=(1,2,3))
    sMAD_WPS   = 1.4826*stats.median_abs_deviation(WPS,axis=(1,2,3))
    # Compute the normalized WPS
    WPS_normed = (WPS-median_WPS[:,np.newaxis,np.newaxis,np.newaxis])/sMAD_WPS[:,np.newaxis,np.newaxis,np.newaxis]
    # Set all wavelet coefficients to zero where the normalized WPS is below 8
    red_mask   = WPS_normed < red_noise_level
    cwt_copy['decomposition'][red_mask]=0
    # Reconstruct the wavefield without noise.
    return transform.reconstruct2d(cwt_copy)


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