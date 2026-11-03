"""Mapa de riesgo para los próximos H días a partir de los datos más recientes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clima, config as C, firms
from .features import construir
from .grilla import tabla_celdas
from .modelo import cargar

NIVELES = [(0.0, "bajo"), (0.6, "moderado"), (0.85, "alto"), (0.95, "muy alto")]  # por percentil


def nivel(pct: float) -> str:
    return [n for corte, n in NIVELES if pct >= corte][-1]


def predecir_hoy(fecha: str | None = None) -> pd.DataFrame:
    """Descarga focos NRT + clima de los últimos 92 días y puntúa cada celda."""
    diario_hist = pd.read_parquet(C.PROCESADO / "focos_diarios.parquet")
    celdas = pd.Index(pd.read_parquet(C.PROCESADO / "celdas_dominio.parquet").cell_id)
    fecha = fecha or pd.Timestamp.today().strftime("%Y-%m-%d")

    print("Clima reciente (Open-Meteo)...")
    cl = clima.descargar_reciente(tabla_celdas(celdas))
    cl = cl[cl.fecha <= fecha]

    print("Focos recientes (FIRMS NRT)...")
    partes = []
    # la API da hasta 5 días por llamada: 19 llamadas cubren los ~92 días de ventana
    for fin in pd.date_range(end=fecha, periods=19, freq="5D"):
        try:
            partes.append(firms.descargar_nrt(fin.strftime("%Y-%m-%d"), dias=5))
        except Exception as e:  # p. ej. fechas fuera de la retención NRT
            print(f"  ! focos hasta {fin.date()} no disponibles: {e}")
    nrt = firms.agregar_por_celda(firms.limpiar(pd.concat(partes, ignore_index=True)))
    diario = (
        pd.concat([diario_hist, nrt])
        .drop_duplicates(["cell_id", "fecha"], keep="last")
    )
    return puntuar(diario, cl, celdas, diario_hist)


def puntuar(diario, cl, celdas, diario_hist) -> pd.DataFrame:
    df = construir(diario, cl, celdas, diario_clim=diario_hist)
    ultimo = df[df.fecha == df.fecha.max()].copy()
    modelo = cargar()
    ultimo["prob_fuego"] = modelo.predict(ultimo[modelo.feature_name()])
    ultimo["percentil"] = ultimo["prob_fuego"].rank(pct=True)
    ultimo["nivel_riesgo"] = ultimo["percentil"].map(nivel)
    return ultimo[["fecha", "cell_id", "lat", "lon", "prob_fuego", "percentil", "nivel_riesgo",
                   "dias_fuego_7", "dias_sin_lluvia", "hr_min"]].sort_values("prob_fuego", ascending=False)


def exportar(riesgo: pd.DataFrame, res: float = C.RESOLUCION) -> None:
    C.crear_directorios()
    f = pd.Timestamp(riesgo.fecha.iloc[0]).strftime("%Y-%m-%d")
    riesgo.to_csv(C.SALIDAS / f"riesgo_{f}.csv", index=False)
    h = res / 2
    feats = []
    for r in riesgo.itertuples():
        poly = [[r.lon - h, r.lat - h], [r.lon + h, r.lat - h], [r.lon + h, r.lat + h], [r.lon - h, r.lat + h], [r.lon - h, r.lat - h]]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly]},
            "properties": {"cell_id": int(r.cell_id), "prob_fuego": round(float(r.prob_fuego), 4),
                           "nivel_riesgo": r.nivel_riesgo},
        })
    (C.SALIDAS / f"riesgo_{f}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"Guardado en {C.SALIDAS}/riesgo_{f}.csv y .geojson")
