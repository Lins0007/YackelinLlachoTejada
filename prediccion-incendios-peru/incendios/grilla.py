"""Grilla regular sobre el Perú: convierte lat/lon en celdas y viceversa."""
import numpy as np
import pandas as pd

from . import config as C


def dimensiones(res: float = C.RESOLUCION) -> tuple[int, int]:
    ny = int(np.ceil((C.LAT_MAX - C.LAT_MIN) / res))
    nx = int(np.ceil((C.LON_MAX - C.LON_MIN) / res))
    return ny, nx


def celda_de(lat, lon, res: float = C.RESOLUCION):
    """Devuelve (iy, ix, cell_id) para arrays de lat/lon."""
    ny, nx = dimensiones(res)
    iy = np.floor((np.asarray(lat) - C.LAT_MIN) / res).astype(int)
    ix = np.floor((np.asarray(lon) - C.LON_MIN) / res).astype(int)
    iy = np.clip(iy, 0, ny - 1)
    ix = np.clip(ix, 0, nx - 1)
    return iy, ix, iy * nx + ix


def centro_de(cell_id, res: float = C.RESOLUCION):
    _, nx = dimensiones(res)
    cell_id = np.asarray(cell_id)
    iy, ix = cell_id // nx, cell_id % nx
    return C.LAT_MIN + (iy + 0.5) * res, C.LON_MIN + (ix + 0.5) * res


def tabla_celdas(cell_ids, res: float = C.RESOLUCION) -> pd.DataFrame:
    cell_ids = np.sort(np.unique(cell_ids))
    _, nx = dimensiones(res)
    lat, lon = centro_de(cell_ids, res)
    return pd.DataFrame(
        {"cell_id": cell_ids, "iy": cell_ids // nx, "ix": cell_ids % nx, "lat": lat, "lon": lon}
    )
