"""Descarga y limpieza de focos de calor de NASA FIRMS.

- Histórico: archivos anuales por país (no requieren API key)
  https://firms.modaps.eosdis.nasa.gov/data/country/<fuente>/<año>/<fuente>_<año>_Peru.csv
- Tiempo casi real (últimos días): API de área, requiere MAP_KEY gratuita
  https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""
from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests

from . import config as C
from .grilla import celda_de

URL_PAIS = "https://firms.modaps.eosdis.nasa.gov/data/country/{fuente}/{anio}/{fuente}_{anio}_{pais}.csv"
URL_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{fuente}/{bbox}/{dias}/{fecha}"
FUENTE_NRT = {"viirs-snpp": "VIIRS_SNPP_NRT", "modis": "MODIS_NRT", "viirs-noaa20": "VIIRS_NOAA20_NRT"}


def _get(url: str, intentos: int = 4) -> requests.Response:
    for i in range(intentos):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                r.raise_for_status()
        except requests.ConnectionError:
            pass
        time.sleep(2 ** (i + 1))
    r.raise_for_status()
    return r


def descargar_historico(anios: list[int], fuentes: list[str] = C.FUENTES_FIRMS) -> None:
    C.crear_directorios()
    for fuente in fuentes:
        for anio in anios:
            destino = C.RAW_FIRMS / f"{fuente}_{anio}_{C.PAIS_FIRMS}.csv"
            if destino.exists():
                print(f"  ya existe {destino.name}")
                continue
            url = URL_PAIS.format(fuente=fuente, anio=anio, pais=C.PAIS_FIRMS)
            print(f"  descargando {url}")
            try:
                destino.write_bytes(_get(url).content)
            except requests.HTTPError as e:
                print(f"  ! no disponible ({e}); se omite")


def descargar_nrt(fecha: str, dias: int = 5, fuente: str = "viirs-snpp", map_key: str | None = None) -> pd.DataFrame:
    """Focos de los últimos `dias` (máx. 5) hasta `fecha` (AAAA-MM-DD)."""
    map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise RuntimeError("Define la variable de entorno FIRMS_MAP_KEY (gratis en https://firms.modaps.eosdis.nasa.gov/api/map_key/)")
    bbox = f"{C.LON_MIN},{C.LAT_MIN},{C.LON_MAX},{C.LAT_MAX}"
    url = URL_AREA.format(key=map_key, fuente=FUENTE_NRT[fuente], bbox=bbox, dias=dias, fecha=fecha)
    df = pd.read_csv(io.StringIO(_get(url).text))
    df["fuente"] = fuente
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra detecciones confiables de vegetación y deja columnas mínimas."""
    df = df.copy()
    conf = df["confidence"].astype(str).str.lower()
    es_viirs = conf.isin(["l", "n", "h", "low", "nominal", "high"])
    conf_num = pd.to_numeric(df["confidence"], errors="coerce")
    ok = (es_viirs & ~conf.isin(["l", "low"])) | (~es_viirs & (conf_num >= 50))
    if "type" in df.columns:  # 0 = vegetación; 1 volcán, 2 fuente estática, 3 offshore
        ok &= df["type"].fillna(0).astype(int) == 0
    df = df[ok]
    out = pd.DataFrame(
        {
            "fecha": pd.to_datetime(df["acq_date"]).dt.normalize(),
            "lat": df["latitude"].astype(float),
            "lon": df["longitude"].astype(float),
            "frp": pd.to_numeric(df.get("frp"), errors="coerce"),
        }
    )
    out = out[(out.lat.between(C.LAT_MIN, C.LAT_MAX)) & (out.lon.between(C.LON_MIN, C.LON_MAX))]
    return out.reset_index(drop=True)


def cargar_historico() -> pd.DataFrame:
    archivos = sorted(C.RAW_FIRMS.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay CSV en {C.RAW_FIRMS}. Ejecuta: python -m incendios descargar-firms")
    partes = []
    for f in archivos:
        d = pd.read_csv(f, low_memory=False)
        d["fuente"] = f.name.split("_")[0]
        partes.append(d)
    return limpiar(pd.concat(partes, ignore_index=True))


def agregar_por_celda(focos: pd.DataFrame, res: float = C.RESOLUCION) -> pd.DataFrame:
    """Cuenta detecciones por celda y día (MODIS+VIIRS ven el mismo fuego: se suman igual,
    el modelo usa sobre todo el indicador 'hubo fuego')."""
    _, _, cid = celda_de(focos["lat"], focos["lon"], res)
    f = focos.assign(cell_id=cid)
    return (
        f.groupby(["cell_id", "fecha"])
        .agg(n_focos=("lat", "size"), frp_sum=("frp", "sum"))
        .reset_index()
    )


def celdas_dominio(diario: pd.DataFrame, min_dias_fuego: int = 2) -> pd.Index:
    """Celdas donde vale la pena predecir: con al menos `min_dias_fuego` días con fuego
    en el histórico. Reduce el problema a zonas inflamables y las consultas de clima."""
    dias = diario.groupby("cell_id")["fecha"].nunique()
    return pd.Index(dias[dias >= min_dias_fuego].index, name="cell_id")
