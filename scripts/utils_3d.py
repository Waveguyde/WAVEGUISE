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