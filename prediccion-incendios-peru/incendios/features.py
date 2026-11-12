"""Construcción de la tabla de entrenamiento: una fila por (celda, día).

Todas las variables usan solo información disponible hasta el día t (inclusive).
El objetivo es 1 si hay al menos un foco en la celda entre t+1 y t+H.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .grilla import dimensiones, tabla_celdas

COL_OBJETIVO = "fuego_prox"
DIAS_CALENTAMIENTO = 90  # días iniciales sin historia suficiente para las ventanas


# ---------- utilidades sobre arrays [día, ...] ----------
def suma_movil(a: np.ndarray, k: int) -> np.ndarray:
    """Suma de los últimos k días incluyendo el actual (eje 0)."""
    cs = np.cumsum(a, axis=0, dtype=float)
    out = cs.copy()
    out[k:] = cs[k:] - cs[:-k]
    return out


def suma_futura(a: np.ndarray, h: int) -> np.ndarray:
    """Suma de los días t+1..t+h; NaN donde el futuro no está completo."""
    n = a.shape[0]
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=float)])
    out = np.full(a.shape, np.nan)
    idx = np.arange(n - h)
    out[idx] = cs[idx + h + 1] - cs[idx + 1]
    return out


def dias_desde(evento: np.ndarray, tope: int = 365) -> np.ndarray:
    out = np.empty(evento.shape, dtype=float)
    cont = np.full(evento.shape[1:], tope, dtype=float)
    for t in range(evento.shape[0]):
        cont = np.where(evento[t], 0, np.minimum(cont + 1, tope))
        out[t] = cont
    return out


def suma_vecinos(g: np.ndarray, radio: int) -> np.ndarray:
    """Suma en una ventana (2r+1)x(2r+1) sobre la grilla [día, ny, nx]."""
    p = np.pad(g, ((0, 0), (radio, radio), (radio, radio)))
    cs = p.cumsum(1).cumsum(2)
    cs = np.pad(cs, ((0, 0), (1, 0), (1, 0)))
    w = 2 * radio + 1
    return cs[:, w:, w:] - cs[:, :-w, w:] - cs[:, w:, :-w] + cs[:, :-w, :-w]


# ---------- climatología sin fuga ----------
def climatologia(diario: pd.DataFrame, cell_ids, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Fracción de días con fuego por (celda, mes) usando SOLO años anteriores."""
    anios = range(diario["fecha"].dt.year.min(), fechas.year.max() + 1)
    base = pd.MultiIndex.from_product([cell_ids, anios, range(1, 13)], names=["cell_id", "anio", "mes"])
    dias_mes = pd.Series(
        [pd.Period(f"{a}-{m:02d}").days_in_month for _, a, m in base], index=base, dtype=float
    )
    f = diario[diario.cell_id.isin(cell_ids)]
    fuego = (
        f.assign(anio=f.fecha.dt.year, mes=f.fecha.dt.month)
        .groupby(["cell_id", "anio", "mes"])["fecha"].nunique()
        .reindex(base, fill_value=0)
        .astype(float)
    )
    acum_f = fuego.groupby(level=["cell_id", "mes"]).cumsum() - fuego
    acum_d = dias_mes.groupby(level=["cell_id", "mes"]).cumsum() - dias_mes
    tasa = (acum_f / acum_d.replace(0, np.nan)).rename("clim_tasa_mes")
    return tasa.reset_index()


# ---------- tabla principal ----------
def construir(
    diario: pd.DataFrame,
    clima: pd.DataFrame,
    celdas: pd.Index,
    horizonte: int = C.HORIZONTE_DIAS,
    res: float = C.RESOLUCION,
    diario_clim: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    diario: cell_id, fecha, n_focos (todas las celdas, para vecindad)
    clima:  cell_id, fecha, variables de C.VARIABLES_CLIMA (celdas del dominio)
    celdas: celdas del dominio a predecir
    diario_clim: histórico para la climatología (por defecto, `diario`)
    """
    ny, nx = dimensiones(res)
    fechas = pd.date_range(clima.fecha.min(), clima.fecha.max(), freq="D")
    nd = len(fechas)
    cel = tabla_celdas(celdas, res)
    cid = cel.cell_id.to_numpy()
    iy, ix = cel.iy.to_numpy(), cel.ix.to_numpy()

    # Grilla densa de fuego [día, ny, nx]
    d = diario[(diario.fecha >= fechas[0]) & (diario.fecha <= fechas[-1])]
    ti = (d.fecha - fechas[0]).dt.days.to_numpy()
    gy, gx = d.cell_id.to_numpy() // nx, d.cell_id.to_numpy() % nx
    focos = np.zeros((nd, ny, nx), dtype=np.float32)
    np.add.at(focos, (ti, gy, gx), d.n_focos.to_numpy())
    fuego = (focos > 0).astype(np.float32)

    v1_7 = suma_movil(suma_vecinos(fuego, 1) - fuego, 7)[:, iy, ix]
    v3_7 = suma_movil(suma_vecinos(fuego, 3), 7)[:, iy, ix]
    v3_30 = suma_movil(suma_vecinos(fuego, 3), 30)[:, iy, ix]
    fc = fuego[:, iy, ix]
    nc = focos[:, iy, ix]
    del focos, fuego

    X = {
        "fuego_hoy": fc,
        "dias_fuego_7": suma_movil(fc, 7),
        "dias_fuego_30": suma_movil(fc, 30),
        "dias_fuego_90": suma_movil(fc, 90),
        "focos_7": suma_movil(nc, 7),
        "dias_desde_fuego": dias_desde(fc > 0),
        "vecinos1_fuego_7": v1_7,
        "vecinos3_fuego_7": v3_7,
        "vecinos3_fuego_30": v3_30,
    }

    # Clima: matriz [día, celda] por variable
    cl = clima[clima.cell_id.isin(cid)]

    def matriz(var: str) -> np.ndarray:
        m = cl.pivot_table(index="fecha", columns="cell_id", values=var).reindex(index=fechas, columns=cid)
        return m.ffill().to_numpy(dtype=float)

    pp = np.nan_to_num(matriz("precipitation_sum"))
    tmax, tmin = matriz("temperature_2m_max"), matriz("temperature_2m_min")
    hr, viento = matriz("relative_humidity_2m_min"), matriz("wind_speed_10m_max")
    et0, rad = matriz("et0_fao_evapotranspiration"), matriz("shortwave_radiation_sum")
    X.update(
        {
            "tmax": tmax,
            "amplitud_termica": tmax - tmin,
            "hr_min": hr,
            "viento_max": viento,
            "radiacion": rad,
            "pp_hoy": pp,
            "pp_7": suma_movil(pp, 7),
            "pp_30": suma_movil(pp, 30),
            "pp_90": suma_movil(pp, 90),
            "dias_sin_lluvia": dias_desde(pp >= 1.0),
            "tmax_media_7": suma_movil(tmax, 7) / 7,
            "hr_min_media_7": suma_movil(hr, 7) / 7,
            "viento_max_media_7": suma_movil(viento, 7) / 7,
            "et0_30": suma_movil(et0, 30),
        }
    )
    X["deficit_hidrico_30"] = X["et0_30"] - X["pp_30"]
    futuro = suma_futura(fc, horizonte)
    X[COL_OBJETIVO] = np.where(np.isnan(futuro), np.nan, (futuro > 0).astype(float))

    df = pd.DataFrame({k: v.ravel() for k, v in X.items()})
    df.insert(0, "fecha", np.repeat(fechas.values, len(cid)))
    df.insert(1, "cell_id", np.tile(cid, nd))
    df["lat"] = np.tile(cel.lat.to_numpy(), nd)
    df["lon"] = np.tile(cel.lon.to_numpy(), nd)
    doy = df.fecha.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["mes"] = df.fecha.dt.month

    clim = climatologia(diario_clim if diario_clim is not None else diario, cid, fechas)
    df["anio"] = df.fecha.dt.year
    df = df.merge(clim, on=["cell_id", "anio", "mes"], how="left").drop(columns=["anio"])

    df = df[df.fecha >= fechas[0] + pd.Timedelta(days=DIAS_CALENTAMIENTO)]
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df.reset_index(drop=True)


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fecha", "cell_id", COL_OBJETIVO)]
