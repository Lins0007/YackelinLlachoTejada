"""Clima diario por celda desde Open-Meteo (gratuito, sin API key).

- Histórico: Archive API (reanálisis ERA5), ~5 días de retraso.
- Para predecir hoy: Forecast API con `past_days`, que da los últimos 92 días.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from . import config as C

URL_ARCHIVO = "https://archive-api.open-meteo.com/v1/archive"
URL_PRONOSTICO = "https://api.open-meteo.com/v1/forecast"
LOTE = 10  # ubicaciones por petición (menos ubicaciones = menos errores 429)


def _pedir(url: str, params: dict, intentos: int = 6) -> list[dict]:
    for i in range(intentos):
        r = requests.get(url, params=params, timeout=300)
        if r.status_code == 200:
            js = r.json()
            return js if isinstance(js, list) else [js]
        if r.status_code == 429 or r.status_code >= 500:  # límite de uso: esperar
            espera = 60 * (i + 1)
            print(f"    Open-Meteo {r.status_code}; esperando {espera}s")
            time.sleep(espera)
            continue
        raise RuntimeError(f"Open-Meteo {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Open-Meteo: demasiados reintentos")


def _a_tabla(respuestas: list[dict], cell_ids: list[int]) -> pd.DataFrame:
    partes = []
    for cid, js in zip(cell_ids, respuestas):
        d = pd.DataFrame(js["daily"])
        d["cell_id"] = cid
        partes.append(d)
    df = pd.concat(partes, ignore_index=True).rename(columns={"time": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def descargar_historico(celdas: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """`celdas` debe tener cell_id, lat, lon. Guarda un parquet por lote (reanudable)."""
    C.crear_directorios()
    partes = []
    for k in range(0, len(celdas), LOTE):
        lote = celdas.iloc[k : k + LOTE]
        destino = C.RAW_CLIMA / f"clima_{inicio}_{fin}_{C.RESOLUCION}_{k:05d}.parquet"
        if destino.exists():
            partes.append(pd.read_parquet(destino))
            continue
        print(f"  clima lote {k // LOTE + 1}/{-(-len(celdas) // LOTE)}")
        params = {
            "latitude": ",".join(f"{v:.3f}" for v in lote.lat),
            "longitude": ",".join(f"{v:.3f}" for v in lote.lon),
            "start_date": inicio,
            "end_date": fin,
            "daily": ",".join(C.VARIABLES_CLIMA),
            "timezone": C.ZONA_HORARIA,
        }
        df = _a_tabla(_pedir(URL_ARCHIVO, params), lote.cell_id.tolist())
        df.to_parquet(destino, index=False)
        partes.append(df)
        time.sleep(1)
    return pd.concat(partes, ignore_index=True)


def descargar_reciente(celdas: pd.DataFrame, dias_pasados: int = 92) -> pd.DataFrame:
    partes = []
    for k in range(0, len(celdas), LOTE):
        lote = celdas.iloc[k : k + LOTE]
        params = {
            "latitude": ",".join(f"{v:.3f}" for v in lote.lat),
            "longitude": ",".join(f"{v:.3f}" for v in lote.lon),
            "past_days": dias_pasados,
            "forecast_days": 1,
            "daily": ",".join(C.VARIABLES_CLIMA),
            "timezone": C.ZONA_HORARIA,
        }
        partes.append(_a_tabla(_pedir(URL_PRONOSTICO, params), lote.cell_id.tolist()))
        time.sleep(1)
    return pd.concat(partes, ignore_index=True)
