import numpy as np

def get_basis_3d(x, y, z, max_order=1):
    """
    Erzeugt 3D-Polynombasis bis zur Gesamtordnung max_order.
    
    Für max_order=1:
        1, x, y, z
    Für max_order=2:
        1, x, y, z, x^2, xy, xz, y^2, yz, z^2
    """
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


def BG_removal_3d(data, max_order=1, fourier_radius=1):
    """
    Entfernt Hintergrund in 3D:
      1) globaler 3D-Polynomfit
      2) Entfernen niedriger 3D-Fourierkomponenten um das Zentrum
    
    Parameter
    ---------
    data : np.ndarray
        3D-Array mit Form (nz, ny, nx)
    max_order : int
        Ordnung des 3D-Polynomfits
    fourier_radius : int
        Radius um die zentrale Fourierkomponente.
        1 => es wird ein Block von 3x3x3 genullt
        2 => 5x5x5, usw.
    
    Returns
    -------
    highpass_data : np.ndarray
        Daten nach Abzug von Fit und niedrigen Fourierkomponenten
    background : np.ndarray
        Geschätzter Hintergrund = Polynomfit + lowpass-Anteil
    """
    data = np.asarray(data, dtype=float)
    
    if data.ndim != 3:
        raise ValueError("data muss ein 3D-Array mit Form (nz, ny, nx) sein.")

    data = data.copy()
    data -= np.nanmean(data)

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

    # 3D-Polynomfit
    basis = get_basis_3d(x, y, z, max_order=max_order)
    A = np.vstack(basis).T
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Fit auf gesamtem Gitter auswerten
    full_basis = np.array(get_basis_3d(X, Y, Z, max_order=max_order))
    fit = np.sum(c[:, None, None, None] * full_basis, axis=0)

    detrended_data = data - fit
    detrended_data[np.isnan(detrended_data)] = 0

    # 3D-FFT
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

    import scipy.ndimage as ndi
    from skimage.measure import label
    from skimage.morphology import h_minima
    from skimage.segmentation import watershed, relabel_sequential

    assert data.ndim == 4, "Expected 4D array."

    iwork = np.nanmax(data) - data

    flip_iwork    = np.flip(iwork,axis=2)
    #flip_mask     = np.flip(mask,axis=2)
    
    # 3) wrap-pad along periodic axes
    pad = [(0, 0)] * data.ndim
    n = data.shape[2]
    pad[2] = (n, n)

    iwork_pad      = np.pad(iwork, pad, mode="wrap")
    flip_iwork_pad = np.pad(flip_iwork, pad, mode="wrap")
    #mask_pad       = np.pad(mask, pad, mode="wrap")
    #flip_mask_pad  = np.pad(flip_mask, pad, mode="wrap")
    new_work  = np.concat((flip_iwork_pad,iwork_pad,flip_iwork_pad),axis=1)
    #new_mask = np.concat((flip_mask_pad,mask_pad,flip_mask_pad),axis=1)

    # 4) markers & watershed on padded data
    mins       = h_minima(new_work, h=prominence)
    structure  = ndi.generate_binary_structure(mins.ndim, 1)
    markers, _ = ndi.label(mins, structure=structure)
    labels_pad = watershed(new_work, markers=markers, connectivity=connectivity_order)
    labels     = merge_periodic_faces_3D(labels_pad, periodic_axes)

    # 5) crop center tile
    orig_shape    = data.shape
    center        = _center_slices(orig_shape, periodic_axes)
    center_labels = labels[center].copy()

    center_labels, _, _ = relabel_sequential(center_labels)
    
    return center_labels
        
    #ds = np.abs(param['wavelength_z'])
    #smooth_work = np.zeros_like(work)
    #for s, t, p in itertools.product(range(len(param['periods'])), range(len(param['thetas'])), range(len(param['phis']))):
    #    filter_size = int(param['smoothing_parameter']*np.ceil(ds[s,t,p]/param['dz']))
    #    if filter_size >= work.shape[3]:
    #        smooth_work[s,t,p,:]=np.max(work[s,t,p,:])
    #    else:
    #        smooth_work[s,t,p,:]=maximum_filter1d(work[s,t,p,:], size=filter_size)

