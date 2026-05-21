import sys
sys.path.append('/home/r/Robert.Reichert/juwavelet')
from juwavelet import parallel
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label
from skimage.morphology import h_minima
from skimage.segmentation import watershed, relabel_sequential
from numba import njit

# BACKGROUND REMOVAL STEP # ----------------------------------------------------

def get_basis(x, y, z, max_order=1):

    basis = []
    for k in range(max_order + 1):
        for i in range(max_order - k + 1):
            for j in range(max_order - k - i + 1):
                basis.append((x ** j) * (y ** i) * (z ** k))
    return basis

def calculate_3dft(arr):
    ft = np.fft.ifftshift(arr, axes=(-3, -2, -1))
    ft = np.fft.fftn(ft, axes=(-3, -2, -1))
    return np.fft.fftshift(ft, axes=(-3, -2, -1))

def calculate_3dift(arr):
    ift = np.fft.ifftshift(arr, axes=(-3, -2, -1))
    ift = np.fft.ifftn(ift, axes=(-3, -2, -1))
    ift = np.fft.fftshift(ift, axes=(-3, -2, -1))
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
    
    if data.ndim != 3:
        raise ValueError("data must be a 3D-Array.")

    data = data.copy()
    # Subtract mean
    data -= np.nanmean(data)

    # Determine the polynomial fit of degree max_order
    nz, ny, nx = data.shape

    z0 = np.arange(nz)
    y0 = np.arange(ny)
    x0 = np.arange(nx)

    Z, Y, X = np.meshgrid(z0, y0, x0, indexing="ij")

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    b = data.ravel()

    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    y = y[mask]
    z = z[mask]

    basis = get_basis(x, y, z, max_order=max_order)
    A = np.vstack(basis).T
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    full_basis = np.array(get_basis(X, Y, Z, max_order=max_order))
    fit = np.sum(c[:, None, None, None] * full_basis, axis=0)
    
    # Detrended data is de-meaned data minus the polynomial fit (The implementation copes with Nan values which are simply set to zero in the subsequent FFT step)
    detrended_data = data - fit
    detrended_data[np.isnan(detrended_data)] = 0

    # Compute the 3D FFT and filter according to the fourier_radius
    ft = calculate_3dft(detrended_data)
    filtered_ft = ft.copy()

    cz, cy, cx = nz // 2, ny // 2, nx // 2
    r = fourier_radius

    filtered_ft[
        max(0, cz - r):min(nz, cz + r + 1),
        max(0, cy - r):min(ny, cy + r + 1),
        max(0, cx - r):min(nx, cx + r + 1)
    ] = 0

    highpass_data = calculate_3dift(filtered_ft)
    lowpass_data = detrended_data - highpass_data

    background = fit + lowpass_data

    return highpass_data, background
    
# BACKGROUND REMOVAL # -----------------------------------------



# DENOISING # ---------------------------------------------------

def denoise_3d(CWT, white_noise_level=None, sMAD_threshold=None):

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

        median_WAS = np.median(WAS, axis=(1,2,3,4,5), keepdims=True)
        abs_dev    = np.abs(WAS - median_WAS)
        sMAD_WAS   = 1.4826 * np.median(abs_dev, axis=(1,2,3,4,5), keepdims=True)

        # avoid division by zero
        sMAD_WAS[sMAD_WAS == 0] = np.finfo(WAS.dtype).eps

        WAS_normed = (WAS - median_WAS) / sMAD_WAS

        sMAD_mask = WAS_normed < sMAD_threshold
        dec[sMAD_mask] = 0

        # update WAS after masking
        WAS = np.abs(dec)

    # --- White noise filtering ---
    if white_noise_level is not None:
        
        # Create a noise WAS as proper reference
        noise = np.random.normal(0,white_noise_level,(dec.shape[3],dec.shape[4],dec.shape[5]))
        noise_cwt = transform.decompose3d(noise, CWT['dx'], CWT['dy'], CWT['dz'], CWT['scale'][0], CWT['dj'], CWT['js'], CWT['jt'], CWT['jp'], aspect=CWT['aspect'], opts=CWT['opts'], mode=CWT['mode'], dtype=np.complex128)
        noise_WAS = fit_power_law(noise_cwt['scale'],np.mean(np.abs(noise_cwt['decomposition']),axis=(1,2,3,4,5)))

        white_mask = WAS < noise_WAS[:,None,None,None,None,None]
        dec[white_mask] = 0

    return parallel.reconstruct3d_parallel(cwt_copy)

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

# DENOISING # ------------------------------------------------------------




def reduce_WAS_624D(decomp,percentile=95):

    """
    For large data sets it is necessary to reduce the dimensionality of the 6D wavelet amplitude spectrum. This function computes a percentile along two spatial dimensions.
    
    Returns
    -------
    result : 4D array containing the percentile wavelet amplitudes. 
    """
    
    L0 = len(decomp)
    L1 = len(decomp[0])
    L2 = len(decomp[0][0])
    L3 = decomp[0][0][0].shape[2]
    
    result = np.empty((L0, L1, L2, L3), dtype=np.float64)  # oder ggf. float32
    
    for i, mid in enumerate(decomp):
        for j, sub in enumerate(mid):
            for k, arr3d in enumerate(sub):
                if arr3d is None:
                    result[i, j, k, :] = 0
                else:
                    for l in range(L3):
                        result[i, j, k, l] = np.nanpercentile(np.abs(arr3d[:,:,l]),percentile)

    return result

"""
def denoise_reduced_WAS(reduced_WAS, white_noise_level=None, sMAD_threshold=None):

    # --- Red noise filtering (robust) ---
    if sMAD_threshold is not None:

        median_WAS = np.median(reduced_WAS, axis=(1,2,3), keepdims=True)
        abs_dev    = np.abs(reduced_WAS - median_WAS)
        sMAD_WAS   = 1.4826 * np.median(abs_dev, axis=(1,2,3), keepdims=True)

        # avoid division by zero
        sMAD_WAS[sMAD_WAS == 0] = np.finfo(reduced_WAS.dtype).eps

        WAS_normed = (reduced_WAS - median_WAS) / sMAD_WAS

        sMAD_mask = WAS_normed < sMAD_threshold
        reduced_WAS[sMAD_mask] = 0

    # --- White noise filtering ---
    if white_noise_level is not None:
        white_mask = reduced_WAS < white_noise_level
        reduced_WAS[white_mask] = 0

    return reduced_WAS
"""

# SEGMENTATION # -------------------------------------------------------------

def _center_slices(orig_shape, periodic_axes):
    center = []
    for ax, n in enumerate(orig_shape):
        if ax in periodic_axes:
            center.append(slice(n, 2*n))  # middle tile along padded axis
        else:
            center.append(slice(0, n))    # unchanged axis
    return tuple(center)

def merge_periodic_faces_3D(labels_pad, periodic_axes):
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
    for ax in periodic_axes:
        left  = np.take(labels_pad, int(labels_pad.shape[ax]/3),  axis=ax)
        right = np.take(labels_pad, int(labels_pad.shape[ax]*2/3), axis=ax)
        if ax == 1:
            right = np.flip(right,axis=1)
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
    
def wavefield_segmentation_3d(data,prominence,periodic_axes=None,connectivity_order=4):

    """
    The watershed function from skimage.segmentation is used to label coherent chunks in the wavelet amplitude spectrum.
    
    Returns
    -------
    seg : array of data.size containing integer labels
    """

    assert data.ndim == 4, "Expected 4D array."

    # data has to be inverted as we are looking not for watersheds but for canyons in the data. We wanna separate peaks.
    iwork      = np.nanmax(data) - data
    flip_iwork = np.flip(iwork,axis=2)
    
    # wrap-pad along periodic axes
    pad    = [(0, 0)] * data.ndim
    n      = data.shape[2]
    pad[2] = (n, n)

    iwork_pad      = np.pad(iwork, pad, mode="wrap")
    flip_iwork_pad = np.pad(flip_iwork, pad, mode="wrap")
    new_work       = np.concat((flip_iwork_pad,iwork_pad,flip_iwork_pad),axis=1)

    # markers & watershed on padded data
    mins       = h_minima(new_work, h=prominence)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
    labels_pad = watershed(new_work, markers=markers, connectivity=connectivity_order)
    labels     = merge_periodic_faces_3D(labels_pad, periodic_axes)

    # crop center tile
    orig_shape    = data.shape
    center        = _center_slices(orig_shape, periodic_axes)
    center_labels = labels[center].copy()

    center_labels, _, _ = relabel_sequential(center_labels)
    
    return center_labels

# SEGMENTATION # --------------------------------------------------------------------------
        

def recon_soi_3d(cwt_dict,segments,soi):

    import sys
    sys.path.append('/home/r/Robert.Reichert/juwavelet')
    from juwavelet import parallel

    # Nur flache Kopie des Dicts
    CWT_filt = cwt_result.copy()

    # Neue verschachtelte Listenstruktur, aber Inhalte zunächst referenzieren
    dec_src = cwt_result['decomposition']
    dec_new = [[[cell for cell in row2] for row2 in row1] for row1 in dec_src]

    mask = (segments == soi)

    # Indizes, die auf 0 gesetzt werden sollen
    idxs = np.argwhere(~mask)

    # Merken, welche Blöcke bereits kopiert wurden
    copied_blocks = set()

    for i, j, k, l in idxs:
        block = dec_new[i][j][k]
    
        if block is None:
            continue
    
        key = (i, j, k)
    
        # Nur beim ersten Zugriff auf diesen Block kopieren
        if key not in copied_blocks:
            dec_new[i][j][k] = block.copy()
            copied_blocks.add(key)
    
        dec_new[i][j][k][:, :, l].fill(0)
    
    CWT_filt['decomposition'] = dec_new
    
    return parallel.reconstruct3d_parallel(CWT_filt)



@njit
def segments2points(reduced_WAS, wavelength_x, wavelength_y, wavelength_z, segments):
    max_label = int(segments.max())

    sum_w  = np.zeros(max_label + 1, dtype=np.float64)
    sum_kx = np.zeros(max_label + 1, dtype=np.float64)
    sum_ky = np.zeros(max_label + 1, dtype=np.float64)
    sum_kz = np.zeros(max_label + 1, dtype=np.float64)

    n0, n1, n2, n3 = segments.shape

    for i0 in range(n0):
        kx_val = wavelength_x[i0]

        for i1 in range(n1):
            ky_val = wavelength_y[i1]

            for i2 in range(n2):
                kz_val = wavelength_z[i2]
                
                for i3 in range(n3):
                    lab = int(segments[i0, i1, i2, i3])

                    if lab <= 0:
                        continue

                    w = reduced_WAS[i0, i1, i2, i3]

                    sum_w[lab] += w
                    sum_kx[lab] += kx_val * w
                    sum_ky[lab] += ky_val * w
                    sum_kz[lab] += kz_val * w

    labels = []
    amps = []
    kx = []
    ky = []
    kz = []

    for lab in range(1, max_label + 1):
        if sum_w[lab] > 0:
            labels.append(lab)
            kx.append(sum_kx[lab] / sum_w[lab])
            ky.append(sum_ky[lab] / sum_w[lab])
            kz.append(sum_kz[lab] / sum_w[lab])
            amps.append(sum_w[lab])            

    return (
        np.asarray(labels),
        np.asarray(amps),
        np.asarray(kx),
        np.asarray(ky),
        np.asarray(kz),
    )




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


def filter_coefficients_3d(decomp, filt_WAS):

    L0, L1, L2, L3 = filt_WAS.shape

    for i in range(L0):
        for j in range(L1):
            for k in range(L2):

                arr3d = decomp[i][j][k]
                if arr3d is None:
                    continue

                mask_l = (filt_WAS[i, j, k, :] == 0)

                if np.any(mask_l):
                    arr3d[:, :, mask_l] = 0  

    return decomp


def recon_dominant_3d(dec):

    shape_xyz = None
    for a in dec:
        for b in a:
            for c in b:
                if c is not None:
                    shape_xyz = c.shape
                    break
            if shape_xyz is not None:
                break
        if shape_xyz is not None:
            break
    
    if shape_xyz is None:
        raise ValueError("dec enthält kein gültiges 3-D Array.")
    
    max_abs = np.full(shape_xyz, -np.inf, dtype=float)
    domi_coeff = np.zeros(shape_xyz, dtype=np.complex128)
    
    idx_i = np.full(shape_xyz, -1, dtype=int)
    idx_j = np.full(shape_xyz, -1, dtype=int)
    idx_k = np.full(shape_xyz, -1, dtype=int)
    
    for i in range(len(dec)):
        for j in range(len(dec[i])):
            for k in range(len(dec[i][j])):
                arr = dec[i][j][k]
                if arr is None:
                    continue
    
                abs_arr = np.abs(arr)
                mask = abs_arr > max_abs
    
                max_abs[mask] = abs_arr[mask]
                domi_coeff[mask] = arr[mask]
                idx_i[mask] = i
                idx_j[mask] = j
                idx_k[mask] = k
    
    return np.real(domi_coeff)