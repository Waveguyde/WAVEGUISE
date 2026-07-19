import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from pyproj import Geod

def define_figgrid(N, data_aspect=1.0, target_aspect=16/9):
    """
    Bestimmt eine sinnvolle subplot-grid Anordnung unter Berücksichtigung
    des Seitenverhältnisses der einzelnen Datenplots.

    Parameters
    ----------
    N : int
        Anzahl der benötigten Subplots.

    data_aspect : float
        Seitenverhältnis eines einzelnen Datenplots: Breite / Höhe.
        Beispiel:
            data_aspect = (xmax - xmin) / (ymax - ymin)

    target_aspect : float
        Gewünschtes Seitenverhältnis der gesamten Figure: Breite / Höhe.
        Beispiel:
            16/9, 4/3, 1.0

    Returns
    -------
    nrows, ncols : int
        Anzahl Zeilen und Spalten.
    """

    best_score = np.inf
    best_grid = None

    for nrows in range(1, N + 1):
        ncols = int(np.ceil(N / nrows))

        grid_aspect = (ncols / nrows) * data_aspect

        # Vergleich im Log-Raum, damit z.B. Faktor 2 und Faktor 1/2
        # gleich stark bestraft werden
        aspect_error = abs(np.log(grid_aspect / target_aspect))

        # kleine Strafe für leere Panels
        empty_panels = nrows * ncols - N
        empty_penalty = 0.05 * empty_panels

        score = aspect_error + empty_penalty

        if score < best_score:
            best_score = score
            best_grid = (nrows, ncols)

    return best_grid
    

def plot_COI(x,order,ax,**kwargs):
    x2 = x-x[0]
    coi_x = np.minimum(x2,x2[-1]-x2)
    coi_boundary_scale = coi_x / np.sqrt(2)
    coi_boundary_wavelength = 4*np.pi*coi_boundary_scale/(order+np.sqrt(2+order**2))
    ax.plot(x, coi_boundary_wavelength, c='k')
    ax.fill_between(x, coi_boundary_wavelength, **kwargs)


def nice_boundary_path_for_maps(lon,lat):
    
    Path = mpath.Path
    path_data = [(Path.MOVETO, (lon.min(), lat.min()))]

    for lo in lon:
        path_data.append((Path.LINETO, (lo, lat.min())))
    
    path_data.append((Path.LINETO, (lon.max(), lat.max())))

    for lo in np.flip(lon):
        path_data.append((Path.LINETO, (lo, lat.max())))
    
    path_data.append((Path.CLOSEPOLY, (lon.min(), lat.min())))
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)

    return path


def plot_AWE(lon, lat, data,
    levels=None, cmap="viridis", ax=None,
    *, add_states=True, coastline_res="110m",
    boundary_samples=80, gridline_kwargs=None, contourf_kwargs=None):
    """
    Plot filled contours on a Cartopy GeoAxes with a nice boundary + gridlines.

    Returns
    -------
    contour : QuadContourSet
        The contourf result (useful for colorbars).
    """
    if ax is None:
        raise ValueError("Please pass a Cartopy GeoAxes in `ax`.")

    # Robust min/max for numpy or xarray
    lon_min = float(np.nanmin(lon))#+1
    lon_max = float(np.nanmax(lon))#-1
    lat_min = float(np.nanmin(lat))#+0.5
    lat_max = float(np.nanmax(lat))#-0.5

    # Boundary + extent
    xs = np.linspace(lon_min, lon_max, boundary_samples)
    ys = np.linspace(lat_min, lat_max, boundary_samples)
    path = nice_boundary_path_for_maps(xs, ys)
    ax.set_boundary(path, transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Features
    ax.coastlines(resolution=coastline_res)
    if add_states:
        try:
            ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor="k", facecolor="none")
        except Exception:
            pass  # e.g. non-US or feature unavailable

    # Gridlines
    if gridline_kwargs is None:
        gridline_kwargs = dict(draw_labels=True, linewidth=1, color="gray",
                               alpha=0.5, linestyle="--")
    gl = ax.gridlines(**gridline_kwargs)
    gl.top_labels = False
    gl.right_labels = True  # usually cleaner than True
    gl.bottom_labels = True
    gl.left_labels = False

    # Contours
    if contourf_kwargs is None:
        contourf_kwargs = {}
    contour = ax.contourf(lon, lat, data, levels=levels,
        cmap=cmap, transform=ccrs.PlateCarree(), extend="both", **contourf_kwargs)

    return contour


def darken(color, factor=0.8):
    rgb = mcolors.to_rgb(color)
    return tuple(factor * c for c in rgb)


def get_distinct_colors(n, saturation=0.75, value=0.9):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, saturation, value)) for h in hues]
    return colors

def compute_offsets(wavy_stuff, amp_seg):
    base = -np.max(np.abs(wavy_stuff))
    offsets = []
    cumulative = 0.0

    for seg in amp_seg:
        current = base - cumulative - 2*np.max(seg)
        offsets.append(current)
        cumulative += 2*np.max(seg)

    return np.array(offsets)


def plot_k_vector(lon, lat, k_along, k_across, step_size, ax, proj, scale=None, mask=None):

    def bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return np.arctan2(x, y)

    theta = np.empty_like(lat)
    theta[:-1, :] = bearing(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    theta[-1, :] = theta[-2, :]

    k = k_along * np.sin(theta) + k_across * np.cos(theta)
    l = k_along * np.cos(theta) - k_across * np.sin(theta)

    geod = Geod(ellps="WGS84")

    azimuth = np.degrees(np.arctan2(k, l))
    lon2, lat2, _ = geod.fwd(lon, lat, azimuth, np.full_like(lon, 10_000.0))

    p1 = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    p2 = proj.transform_points(ccrs.PlateCarree(), lon2, lat2)

    x = p1[..., 0]
    y = p1[..., 1]
    u_proj = p2[..., 0] - p1[..., 0]
    v_proj = p2[..., 1] - p1[..., 1]

    sl = (slice(None, None, step_size), slice(None, None, step_size))

    x_plot = x[sl]
    y_plot = y[sl]
    u_plot = u_proj[sl]
    v_plot = v_proj[sl]

    if mask is not None:
        mask_plot = np.asarray(mask, dtype=bool)[sl]
        x_plot = x_plot[mask_plot]
        y_plot = y_plot[mask_plot]
        u_plot = u_plot[mask_plot]
        v_plot = v_plot[mask_plot]

    return ax.quiver(
        x_plot, y_plot, u_plot, v_plot,
        pivot="middle",
        angles="xy",
        scale_units="xy",
        scale=scale
    )