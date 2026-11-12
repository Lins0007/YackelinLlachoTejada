"""Línea de comandos.

    python -m incendios descargar-firms --desde 2014 --hasta 2024
    python -m incendios preparar-focos
    python -m incendios descargar-clima --desde 2014-01-01 --hasta 2024-12-31
    python -m incendios construir-features
    python -m incendios entrenar
    python -m incendios predecir            # requiere FIRMS_MAP_KEY
"""
import argparse

import pandas as pd

from . import clima, config as C, features, firms, modelo, predecir
from .grilla import tabla_celdas


def main() -> None:
    p = argparse.ArgumentParser(prog="incendios", description="Predicción de incendios forestales en Perú")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("descargar-firms", help="Focos de calor históricos (NASA FIRMS)")
    a.add_argument("--desde", type=int, default=2014)
    a.add_argument("--hasta", type=int, default=2024)

    a = sub.add_parser("preparar-focos", help="Agrega focos por celda y día y define el dominio")
    a.add_argument("--min-dias-fuego", type=int, default=2)

    a = sub.add_parser("descargar-clima", help="Clima diario por celda (Open-Meteo/ERA5)")
    a.add_argument("--desde", default="2014-01-01")
    a.add_argument("--hasta", default="2024-12-31")

    sub.add_parser("construir-features", help="Tabla celda-día con variables y objetivo")
    sub.add_parser("entrenar", help="Entrena LightGBM y evalúa contra líneas base")
    a = sub.add_parser("predecir", help="Mapa de riesgo para los próximos días")
    a.add_argument("--fecha", default=None, help="AAAA-MM-DD (por defecto hoy)")

    args = p.parse_args()
    C.crear_directorios()

    if args.cmd == "descargar-firms":
        firms.descargar_historico(list(range(args.desde, args.hasta + 1)))

    elif args.cmd == "preparar-focos":
        diario = firms.agregar_por_celda(firms.cargar_historico())
        dominio = firms.celdas_dominio(diario, args.min_dias_fuego)
        diario.to_parquet(C.PROCESADO / "focos_diarios.parquet", index=False)
        tabla_celdas(dominio).to_parquet(C.PROCESADO / "celdas_dominio.parquet", index=False)
        print(f"{len(diario):,} celda-días con fuego; dominio = {len(dominio):,} celdas de {C.RESOLUCION}°")

    elif args.cmd == "descargar-clima":
        celdas = pd.read_parquet(C.PROCESADO / "celdas_dominio.parquet")
        cl = clima.descargar_historico(celdas, args.desde, args.hasta)
        cl.to_parquet(C.PROCESADO / "clima_diario.parquet", index=False)
        print(f"{len(cl):,} filas de clima")

    elif args.cmd == "construir-features":
        diario = pd.read_parquet(C.PROCESADO / "focos_diarios.parquet")
        cl = pd.read_parquet(C.PROCESADO / "clima_diario.parquet")
        celdas = pd.Index(pd.read_parquet(C.PROCESADO / "celdas_dominio.parquet").cell_id)
        df = features.construir(diario, cl, celdas)
        df.to_parquet(C.PROCESADO / "features.parquet", index=False)
        print(f"{len(df):,} filas, {df[features.COL_OBJETIVO].mean():.3%} positivas")

    elif args.cmd == "entrenar":
        df = pd.read_parquet(C.PROCESADO / "features.parquet")
        m, rep = modelo.entrenar(df)
        modelo.guardar(m, rep)
        modelo.imprimir_reporte(rep)

    elif args.cmd == "predecir":
        riesgo = predecir.predecir_hoy(args.fecha)
        predecir.exportar(riesgo)
        print(riesgo.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
