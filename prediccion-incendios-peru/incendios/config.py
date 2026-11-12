"""Configuración central del proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RAW_FIRMS = DATA / "raw" / "firms"
RAW_CLIMA = DATA / "raw" / "clima"
PROCESADO = DATA / "processed"
MODELOS = RAIZ / "modelos"
SALIDAS = RAIZ / "salidas"

# Rectángulo que contiene al Perú (lat/lon, grados decimales)
LAT_MIN, LAT_MAX = -18.4, 0.1
LON_MIN, LON_MAX = -81.4, -68.6

# Tamaño de celda de la grilla en grados (0.25° ≈ 28 km; 0.5° ≈ 55 km)
RESOLUCION = 0.5

# Horizonte de predicción: ¿habrá fuego en la celda en los próximos H días?
HORIZONTE_DIAS = 7

# Fuentes de focos de calor FIRMS (archivo anual por país, sin API key)
FUENTES_FIRMS = ["viirs-snpp", "modis"]
PAIS_FIRMS = "Peru"

# Variables diarias de Open-Meteo (reanálisis ERA5)
VARIABLES_CLIMA = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
]
ZONA_HORARIA = "America/Lima"

# Partición temporal (nunca aleatoria: evita fuga de información del futuro)
ANIO_FIN_TRAIN = 2021
ANIO_VALIDACION = 2022  # años posteriores = test

SEMILLA = 42


def crear_directorios() -> None:
    for d in (RAW_FIRMS, RAW_CLIMA, PROCESADO, MODELOS, SALIDAS):
        d.mkdir(parents=True, exist_ok=True)
