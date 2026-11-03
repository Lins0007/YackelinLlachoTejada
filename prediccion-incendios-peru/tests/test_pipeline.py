"""Prueba de humo del pipeline con datos SINTÉTICOS (solo verifica que el código
funcione y no tenga fuga de información; sus métricas no significan nada)."""
import numpy as np
import pandas as pd
import pytest

from incendios import config as C, features, firms, modelo
from incendios.grilla import celda_de, centro_de, tabla_celdas


def datos_sinteticos(seed=0):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2019-01-01", "2023-12-31")
    celdas = tabla_celdas(celda_de(rng.uniform(-14, -6, 40), rng.uniform(-77, -70, 40))[2])
    n = len(fechas)
    temporada = np.cos(2 * np.pi * (fechas.dayofyear.to_numpy() - 240) / 365.25)  # pico ago-sep
    clima, focos = [], []
    for c in celdas.itertuples():
        pp = rng.gamma(0.5, 6, n) * (1 - 0.8 * temporada.clip(0))
        clima.append(pd.DataFrame({
            "fecha": fechas, "cell_id": c.cell_id,
            "temperature_2m_max": 28 + 4 * temporada + rng.normal(0, 1, n),
            "temperature_2m_min": 16 + rng.normal(0, 1, n),
            "precipitation_sum": pp,
            "relative_humidity_2m_min": 55 - 20 * temporada + rng.normal(0, 5, n),
            "wind_speed_10m_max": 10 + rng.normal(0, 2, n),
            "et0_fao_evapotranspiration": 4 + temporada,
            "shortwave_radiation_sum": 18 + 3 * temporada,
        }))
        p = 1 / (1 + np.exp(-(-5 + 3 * temporada)))
        dias = fechas[rng.random(n) < p]
        lat, lon = centro_de(np.full(len(dias), c.cell_id))
        focos.append(pd.DataFrame({"acq_date": dias.strftime("%Y-%m-%d"), "latitude": lat, "longitude": lon,
                                   "confidence": "n", "type": 0, "frp": 10.0}))
    return pd.concat(focos), pd.concat(clima), celdas


def test_grilla_ida_y_vuelta():
    lat, lon = np.array([-12.05, -3.7]), np.array([-77.04, -73.25])
    _, _, cid = celda_de(lat, lon)
    clat, clon = centro_de(cid)
    assert np.all(np.abs(clat - lat) <= C.RESOLUCION / 2) and np.all(np.abs(clon - lon) <= C.RESOLUCION / 2)


def test_limpiar_filtra_baja_confianza():
    df = pd.DataFrame({"acq_date": ["2020-01-01"] * 4, "latitude": -10.0, "longitude": -75.0,
                       "confidence": ["l", "h", "30", "80"], "type": [0, 0, 0, 2], "frp": 1.0})
    assert len(firms.limpiar(df)) == 1


def test_objetivo_mira_solo_al_futuro():
    a = np.zeros((10, 1)); a[5] = 1
    fut = features.suma_futura(a, 3)
    assert list(fut[:, 0][:7]) == [0, 0, 1, 1, 1, 0, 0] and np.isnan(fut[-1, 0])
    assert features.suma_movil(a, 3)[:, 0].tolist()[4:9] == [0, 1, 1, 1, 0]


@pytest.fixture(scope="module")
def tabla():
    focos_raw, clima, celdas = datos_sinteticos()
    diario = firms.agregar_por_celda(firms.limpiar(focos_raw))
    return features.construir(diario, clima, pd.Index(celdas.cell_id))


def test_features_sin_fuga(tabla):
    # La climatología del primer año disponible no puede conocer ese año
    primer = tabla[tabla.fecha.dt.year == 2019]
    assert primer["clim_tasa_mes"].isna().all()
    assert tabla[features.COL_OBJETIVO].dropna().isin([0, 1]).all()
    assert features.COL_OBJETIVO not in features.columnas_modelo(tabla)


def test_entrenamiento_y_metricas(tabla, monkeypatch):
    monkeypatch.setattr(C, "ANIO_FIN_TRAIN", 2021)
    monkeypatch.setattr(C, "ANIO_VALIDACION", 2022)
    m, rep = modelo.entrenar(tabla, rondas=50)
    assert {"validacion", "test"} <= rep["evaluacion"].keys()
    assert 0 <= rep["evaluacion"]["test"]["modelo"]["roc_auc"] <= 1


def test_objetivo_coincide_con_focos(tabla):
    focos_raw, _, _ = datos_sinteticos()
    diario = firms.agregar_por_celda(firms.limpiar(focos_raw))
    fila = tabla[tabla[features.COL_OBJETIVO] == 1].iloc[0]
    ventana = diario[(diario.cell_id == fila.cell_id) & (diario.fecha > fila.fecha)
                     & (diario.fecha <= fila.fecha + pd.Timedelta(days=C.HORIZONTE_DIAS))]
    assert len(ventana) > 0


def test_prediccion_exporta_mapa(tabla, tmp_path, monkeypatch):
    from incendios import predecir
    for nombre in ("MODELOS", "SALIDAS", "PROCESADO", "RAW_FIRMS", "RAW_CLIMA"):
        monkeypatch.setattr(C, nombre, tmp_path / nombre)
    m, rep = modelo.entrenar(tabla, rondas=30)
    modelo.guardar(m, rep)
    focos_raw, clima, celdas = datos_sinteticos()
    diario = firms.agregar_por_celda(firms.limpiar(focos_raw))
    reciente = clima[clima.fecha >= "2023-09-01"]
    riesgo = predecir.puntuar(diario, reciente, pd.Index(celdas.cell_id), diario)
    assert len(riesgo) == len(celdas) and riesgo.prob_fuego.between(0, 1).all()
    predecir.exportar(riesgo)
    assert list((tmp_path / "SALIDAS").glob("riesgo_*.geojson"))
