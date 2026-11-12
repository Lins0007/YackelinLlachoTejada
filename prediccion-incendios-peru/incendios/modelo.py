"""Entrenamiento y evaluación del modelo (LightGBM) con partición temporal."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C
from .features import COL_OBJETIVO, columnas_modelo

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": C.SEMILLA,
    "verbose": -1,
}


def particionar(df: pd.DataFrame):
    df = df[df[COL_OBJETIVO].notna()]
    anio = df.fecha.dt.year
    # se descartan los últimos H días de cada bloque para que el objetivo no "mire" al siguiente
    h = pd.Timedelta(days=C.HORIZONTE_DIAS)
    fin_train = pd.Timestamp(f"{C.ANIO_FIN_TRAIN}-12-31") - h
    fin_val = pd.Timestamp(f"{C.ANIO_VALIDACION}-12-31") - h
    train = df[(anio <= C.ANIO_FIN_TRAIN) & (df.fecha <= fin_train)]
    val = df[(anio == C.ANIO_VALIDACION) & (df.fecha <= fin_val)]
    test = df[anio > C.ANIO_VALIDACION]
    return train, val, test


def recall_top_k(df: pd.DataFrame, score: np.ndarray, frac: float = 0.10) -> float:
    """De todas las celdas que tuvieron fuego, ¿qué fracción estaba en el top `frac`
    de riesgo de su día? Mide si sirve para priorizar patrullaje."""
    d = pd.DataFrame({"fecha": df.fecha.values, "y": df[COL_OBJETIVO].values, "s": score})
    d["rank"] = d.groupby("fecha")["s"].rank(ascending=False, pct=True)
    pos = d[d.y == 1]
    return float((pos["rank"] <= frac).mean()) if len(pos) else float("nan")


def metricas(df: pd.DataFrame, score: np.ndarray) -> dict:
    y = df[COL_OBJETIVO].to_numpy()
    s = np.nan_to_num(score)
    out = {
        "n": int(len(y)),
        "tasa_positivos": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "recall_top10pct": recall_top_k(df, s, 0.10),
        "recall_top20pct": recall_top_k(df, s, 0.20),
    }
    if np.all((s >= 0) & (s <= 1)):
        out["brier"] = float(brier_score_loss(y, s))
    return out


def lineas_base(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "persistencia (fuego últimos 7 días)": (df["dias_fuego_7"] + 0.1 * df["vecinos1_fuego_7"]).to_numpy(),
        "climatología (tasa histórica del mes)": df["clim_tasa_mes"].to_numpy(),
    }


def entrenar(df: pd.DataFrame, rondas: int = 3000) -> tuple[lgb.Booster, dict]:
    train, val, test = particionar(df)
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Faltan datos: revisa los años de train/validación en config.py")
    cols = columnas_modelo(df)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}  variables={len(cols)}")

    dtr = lgb.Dataset(train[cols], train[COL_OBJETIVO])
    dva = lgb.Dataset(val[cols], val[COL_OBJETIVO], reference=dtr)
    modelo = lgb.train(
        PARAMS,
        dtr,
        num_boost_round=rondas,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    reporte = {"horizonte_dias": C.HORIZONTE_DIAS, "resolucion": C.RESOLUCION,
               "mejor_iteracion": modelo.best_iteration, "variables": cols, "evaluacion": {}}
    for nombre, parte in (("validacion", val), ("test", test)):
        if len(parte) == 0:
            continue
        res = {"modelo": metricas(parte, modelo.predict(parte[cols], num_iteration=modelo.best_iteration))}
        for base, s in lineas_base(parte).items():
            res[base] = metricas(parte, s)
        reporte["evaluacion"][nombre] = res

    imp = pd.Series(modelo.feature_importance("gain"), index=cols).sort_values(ascending=False)
    reporte["importancia"] = (imp / imp.sum()).round(4).to_dict()
    return modelo, reporte


def guardar(modelo: lgb.Booster, reporte: dict) -> None:
    C.crear_directorios()
    modelo.save_model(str(C.MODELOS / "lgbm_incendios.txt"), num_iteration=modelo.best_iteration)
    (C.MODELOS / "reporte.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False))


def cargar() -> lgb.Booster:
    return lgb.Booster(model_file=str(C.MODELOS / "lgbm_incendios.txt"))


def imprimir_reporte(reporte: dict) -> None:
    for parte, res in reporte["evaluacion"].items():
        print(f"\n== {parte.upper()} ==")
        tabla = pd.DataFrame(res).T[["roc_auc", "pr_auc", "recall_top10pct", "recall_top20pct", "tasa_positivos"]]
        print(tabla.round(3).to_string())
    print("\nVariables más importantes:")
    for k, v in list(reporte["importancia"].items())[:10]:
        print(f"  {k:<24} {v:.3f}")
