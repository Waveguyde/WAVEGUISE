import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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
    lon_min = float(np.nanmin(lon))+1
    lon_max = float(np.nanmax(lon))-1
    lat_min = float(np.nanmin(lat))+0.5
    lat_max = float(np.nanmax(lat))-0.5

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