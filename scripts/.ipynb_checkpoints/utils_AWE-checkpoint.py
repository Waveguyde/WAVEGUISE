import numpy as np

R = 6371000.0  # Erdradius in m

def get_track_unit_vectors(lat, lon):
    """
    lat, lon: 2D arrays oder 1D arrays entlang des Tracks
    Erwartung: along-track ist Achse 0
    Rückgabe:
        e_along_east, e_along_north,
        e_across_east, e_across_north
    """

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.unwrap(np.deg2rad(lon))

    # zentrale Differenzen entlang along-track
    dlat = np.zeros_like(lat_rad)
    dlon = np.zeros_like(lon_rad)

    dlat[1:-1] = lat_rad[2:] - lat_rad[:-2]
    dlon[1:-1] = lon_rad[2:] - lon_rad[:-2]

    # Ränder: einfache Differenzen
    dlat[0] = lat_rad[1] - lat_rad[0]
    dlat[-1] = lat_rad[-1] - lat_rad[-2]
    dlon[0] = lon_rad[1] - lon_rad[0]
    dlon[-1] = lon_rad[-1] - lon_rad[-2]

    dx = np.cos(lat_rad) * dlon   # East
    dy = dlat                     # North

    norm = np.hypot(dx,dy)
    norm[norm == 0] = np.nan

    e_along_e = dx / norm
    e_along_n = dy / norm

    # 90° gegen den Uhrzeigersinn:
    # positive Across-Track-Richtung liegt links der Flugrichtung
    e_across_e = -e_along_n
    e_across_n =  e_along_e

    #e_across_e = e_along_n
    #e_across_n = -e_along_e

    return e_along_e, e_along_n, e_across_e, e_across_n


def rotate_wave_vector(k_across, k_along, lat, lon):
    e_along_e, e_along_n, e_across_e, e_across_n = get_track_unit_vectors(lat, lon)

    k_east  = k_across * e_across_e + k_along * e_along_e
    k_north = k_across * e_across_n + k_along * e_along_n

    return k_east, k_north