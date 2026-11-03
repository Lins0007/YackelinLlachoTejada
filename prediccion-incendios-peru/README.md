# Predicción anticipada de incendios forestales en el Perú

Modelo de machine learning que estima, para cada celda de una grilla sobre el Perú,
la **probabilidad de que se detecte un incendio en los próximos 7 días**. Sirve para
priorizar patrullaje, alertas y recursos de prevención antes de que el fuego aparezca.

## Enfoque

| Elemento | Decisión |
|---|---|
| Unidad de análisis | Celda de 0.5° (≈55 km) × día. Configurable en `incendios/config.py` |
| Dominio | Celdas con ≥2 días de fuego en el histórico (zonas inflamables) |
| Objetivo | `fuego_prox = 1` si hay ≥1 foco de calor confiable entre t+1 y t+7 |
| Datos de fuego | NASA FIRMS, VIIRS S-NPP (375 m) + MODIS (1 km); se excluyen la confianza baja y las fuentes no vegetales |
| Datos de clima | Open-Meteo Archive (reanálisis ERA5): temperatura, lluvia, humedad, viento, evapotranspiración y radiación |
| Modelo | LightGBM (gradient boosting) con early stopping |
| Validación | Partición **temporal**: train ≤2021, validación 2022, test 2023 en adelante |
| Comparación | Líneas base de persistencia (¿hubo fuego la última semana?) y climatología (tasa histórica del mes) |

### Variables (todas usan solo información hasta el día t)

- **Historia de fuego**: fuego hoy, días con fuego en los últimos 7/30/90 días, días desde el último fuego, detecciones de la semana.
- **Vecindad**: fuego en las 8 celdas vecinas y en un radio de 3 celdas (propagación y quemas agrícolas en la zona).
- **Sequedad**: lluvia acumulada de 7/30/90 días, días sin lluvia, déficit hídrico (ET0 − lluvia, 30 días).
- **Meteorología**: Tmax, amplitud térmica, humedad mínima, viento máximo, radiación y sus medias de 7 días.
- **Estacionalidad y lugar**: día del año (seno/coseno), mes, lat/lon, y la tasa histórica de fuego de la celda en ese mes, calculada **solo con años anteriores** para evitar fuga de información.

### Métricas

El fuego es un evento raro y desbalanceado, así que la exactitud (accuracy) no sirve. Se reportan:
- **PR-AUC**: la métrica principal para clases raras.
- **ROC-AUC**.
- **Recall top 10% / top 20%**: de las celdas que realmente se quemaron, qué porcentaje estaba entre el 10% / 20% de mayor riesgo de ese día. Es la métrica operativa: "si patrullo solo el 10% del territorio, ¿cuántos incendios cubro?".

El modelo solo vale la pena si supera claramente a las dos líneas base.

## Uso

```bash
pip install -r requirements.txt
```

```bash
python -m incendios descargar-firms --desde 2014 --hasta 2024
```

```bash
python -m incendios preparar-focos
```

```bash
python -m incendios descargar-clima --desde 2014-01-01 --hasta 2024-12-31
```

```bash
python -m incendios construir-features
```

```bash
python -m incendios entrenar
```

`entrenar` imprime la tabla de métricas frente a las líneas base y guarda
`modelos/lgbm_incendios.txt` y `modelos/reporte.json` (métricas e importancia de variables).

### Predicción con datos de hoy

Requiere una MAP_KEY gratuita de FIRMS (https://firms.modaps.eosdis.nasa.gov/api/map_key/):

```bash
export FIRMS_MAP_KEY=tu_clave
```

```bash
python -m incendios predecir
```

Genera `salidas/riesgo_AAAA-MM-DD.csv` y `.geojson` con la probabilidad y el nivel de riesgo
(bajo / moderado / alto / muy alto, por percentil) de cada celda. El GeoJSON se abre
directamente en QGIS, geojson.io o Power BI (visual de mapa de formas).

### Pruebas

```bash
python -m pytest -q tests
```

Las pruebas usan datos **sintéticos** solo para verificar que el código funciona y que no hay
fuga de información (el objetivo mira solo al futuro, la climatología solo al pasado).
Sus métricas no representan el desempeño real del modelo.

## Notas prácticas

- **Límite de Open-Meteo**: el plan gratuito cuenta cada 2 semanas de datos por ubicación como una
  llamada (10 000 por día). Con ~500 celdas y 11 años, la descarga toma varios días. El comando
  es **reanudable**: guarda cada lote en `data/raw/clima/`, así que puedes volver a ejecutarlo
  y continúa donde quedó. Para ir más rápido, reduce los años o usa ERA5-Land de Copernicus (ver mejoras).
- **Resolución 0.25°**: más detalle, pero unas 4 veces más celdas y más descarga de clima.
- Los archivos anuales de FIRMS por país suelen publicarse con algunos meses de retraso. El año
  en curso se cubre con la API NRT.

## Mejoras siguientes

1. **Vegetación y combustible**: NDVI/EVI de MODIS (MOD13A2) y cobertura del suelo (MapBiomas Perú).
2. **Actividad humana**: distancia a vías y a la frontera agrícola, densidad poblacional. Gran parte de los incendios en la sierra y la selva vienen de quemas agrícolas.
3. **Pronóstico meteorológico** de los días t+1..t+7 como variables (en el entrenamiento se usaría el clima observado como sustituto).
4. **ERA5-Land vía Copernicus CDS** para descargar el clima de toda la grilla en pocas peticiones.
5. Calibración de probabilidades (isotónica) y umbrales de alerta acordados con SERFOR, INDECI o los gobiernos regionales.
6. Un modelo por macro-región (costa, sierra, selva), cuyas temporadas y causas son distintas.

## Estructura

```
incendios/
  config.py      parámetros (grilla, horizonte, años de partición)
  grilla.py      lat/lon <-> celda
  firms.py       descarga y limpieza de focos de calor
  clima.py       descarga de clima (Open-Meteo)
  features.py    tabla celda-día, variables y objetivo
  modelo.py      entrenamiento, líneas base y métricas
  predecir.py    mapa de riesgo actual (CSV + GeoJSON)
  __main__.py    línea de comandos
tests/           pruebas con datos sintéticos
```
