# DATAFEST 2026 BCP - Predicción de conversión

Proyecto del CodingChallenge para estimar la probabilidad de primera conversión de clientes en **diciembre de 2026**. Este repositorio conserva las etapas de desarrollo, la validación temporal, las predicciones y los modelos finales para que el equipo pueda revisar y reproducir la solución.

**Estado al 6 de octubre de 2026:** validación retrospectiva junio-noviembre completada y archivo de diciembre generado. La configuración adoptada es CatBoost `depth=3`, `l2_leaf_reg=10`, `learning_rate=0.03`, historial completo, pesos uniformes y promedio de las semillas `2026`, `2027` y `2028`. La inestabilidad entre semillas sigue siendo una limitación del procedimiento adoptado.

## 1. Por dónde empezar

1. Leer este README y el [informe de validación actual](resultados_fase2_validacion/INFORME_VALIDACION.md).
2. Consultar [configuracion_adoptada.json](resultados_fase2_validacion/configuracion_adoptada.json) para la decisión y [diciembre/manifiesto.json](resultados_fase2_validacion/diciembre/manifiesto.json) para el entrenamiento final y sus hashes.
3. Usar [baseline_datafest_2.1.py](baseline_datafest_2.1.py) como punto de entrada para la validación y finalización actuales.
4. Para comparar con otros modelos, utilizar las predicciones de validación y el mismo protocolo temporal antes de ordenar propuestas por Gini.

El archivo correspondiente a la configuración adoptada es:

```text
resultados_fase2_validacion/diciembre/submission_fase2_validada.csv
```

Contiene **9.900 filas**, en el orden de `sample_submission.csv`, y exactamente las columnas `id_cliente,prediccion`. Las predicciones son probabilidades continuas entre 0 y 1. Este archivo se generó con los modelos de la validación actual; las entregas de `resultados_history/` y `resultados_banking/` pertenecen al baseline inicial.

## 2. Datos y objetivo

| Archivo | Contenido |
|---|---|
| `train.csv` | 110.100 filas, 25 columnas, enero-noviembre de 2026; incluye `objetivo`. |
| `test.csv` | 9.900 filas, 24 columnas, diciembre de 2026; sin `objetivo`. |
| `sample_submission.csv` | IDs, orden y estructura requeridos para la entrega. |
| `metaData.csv` | Diccionario de campos proporcionado para el reto. |
| `DATASET_DESCRIPTION.md` | Descripción del dataset, tipos de variables y formato de entrega. |
| `instrucciones_adicionales_datafest_2026.md` | Instrucciones de la competencia. |

Cada fila representa un cliente observado en un mes. `mes` tiene formato entero `AAAAMM`. Un cliente puede aparecer en varios meses; `objetivo=1` indica su primera conversión en ese mes. El loader verifica que no existan observaciones de entrenamiento posteriores a una conversión del mismo cliente.

La métrica oficial es:

```text
Gini = 2 * ROC AUC - 1
```

Las métricas de este proyecto se calculan con probabilidades, sin convertirlas a clases. Los promedios reportados son **medias no ponderadas de los AUC mensuales** y sus Gini correspondientes. No deben confundirse con el AUC calculado agrupando todas las filas.

## 3. Estructura del proyecto

```text
.
├── README.md
├── train.csv / test.csv / sample_submission.csv / metaData.csv
├── DATASET_DESCRIPTION.md
├── instrucciones_adicionales_datafest_2026.md
├── diagnostico_datafest.py / diagnostico_datafest.json
├── baseline_datafest.py                 # Baseline inicial: CatBoost + LightGBM
├── baseline_datafest_1.0.py             # Validación temporal, búsqueda CatBoost, logística
├── baseline_datafest_2.0.py             # Experimentos de pesos e historial
├── baseline_datafest_2.1.py             # Validación actual y modelos de diciembre
├── test_baseline_datafest_2_0.py         # Pruebas de Fase 2
├── test_baseline_datafest_2_1.py         # Causalidad, métricas y modelos guardados
├── informe_validacion_fase2.py          # Informe y contrastes descriptivos por semilla
├── reutilizar_folds_identicos.py        # Utilidad de deduplicación de la ejecución original
├── resultados_fase1/                   # Evidencia histórica; requerida por Fase 2.1
├── resultados_fase2/                   # Evidencia histórica; requerida por Fase 2.1
├── resultados_fase2_validacion/         # Resultados de referencia actuales
│   ├── requirements.txt
│   ├── protocolo_fijado.json
│   ├── configuracion_adoptada.json
│   ├── INFORME_VALIDACION.md
│   ├── metricas.json
│   ├── predicciones_validacion.csv
│   ├── ajustes/
│   └── diciembre/
│       ├── modelo_2026.cbm / modelo_2027.cbm / modelo_2028.cbm
│       ├── parametros_*.json / seleccion_*.json
│       ├── predicciones_*.csv
│       ├── manifiesto.json
│       └── submission_fase2_validada.csv
├── resultados_history/ / resultados_banking/  # Resultados del baseline inicial
├── validacion_local/                   # Otra ejecución local, separada de la referencia
├── prueba_fase1/                       # Prueba rápida; no utilizar para comparar scores
└── output/pdf/auditoria_completa_datafest_2026.pdf
```

`tmp/` y `__pycache__/` contienen intermediarios o cachés y no son necesarios para ejecutar el modelo. Para un traspaso completo, compartir el código, los cuatro CSV de entrada, las carpetas de resultados y sus manifiestos. En particular, **Fase 2.1 no funciona como un script independiente que solo requiere train/test**: audita y reutiliza checkpoints de Fases 1 y 2.

Los hashes de `originales_sha256.json` también cubren los archivos presentes en `resultados_history/` y `resultados_banking/`. Para reanudar la ejecución de referencia hay que conservar esas carpetas tal como estaban al fijar el protocolo.

## 4. Entorno e instalación

Entorno registrado para las ejecuciones de referencia:

| Componente | Versión |
|---|---|
| Python | **3.14.7** |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.9.1 |
| CatBoost | 1.2.10 |

La ruta actual usa CPU y **4 threads** por ajuste de CatBoost. No requiere GPU. LightGBM solo es necesario si se desea ejecutar el baseline inicial; no forma parte de la receta actual ni del archivo de dependencias de la validación.

### Preparar un entorno en Windows / PowerShell

Abrir la terminal en la raíz del repositorio, con Python 3.14.7 instalado:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r resultados_fase2_validacion/requirements.txt
.\.venv\Scripts\python.exe -c "import sys, numpy, pandas, sklearn, catboost; print(sys.version); print(numpy.__version__, pandas.__version__, sklearn.__version__, catboost.__version__)"
```

Comprobar que la versión de Python sea **3.14.7**, no solamente 3.14. Los manifiestos comparan las versiones exactas. Los comandos siguientes usan directamente el Python del entorno, por lo que no es necesario activar `.venv` ni cambiar políticas de ejecución de PowerShell.

En otros sistemas, utilizar el ejecutable Python del entorno virtual equivalente y las mismas versiones. No se ha verificado aquí la reproducción entre sistemas operativos.

### Conservar la integridad al clonar

La validación compara SHA-256 de los **bytes** del código y los datos. La conversión automática de finales de línea puede cambiar un hash aunque el código tenga el mismo significado. Para un clon nuevo, sustituir `URL_DEL_REPOSITORIO` por la URL del equipo:

```powershell
git -c core.autocrlf=false clone "URL_DEL_REPOSITORIO"
```

Evitar guardar los CSV originales desde Excel o editar scripts con una validación ya congelada en curso. Si se detecta un hash diferente, revisar la causa y preservar la evidencia histórica; no modificar el manifiesto para forzar que el control pase.

## 5. Revisar la ejecución existente

Para una revisión de los resultados guardados no hace falta volver a entrenar.

```powershell
.\.venv\Scripts\python.exe baseline_datafest_2.1.py self-test
.\.venv\Scripts\python.exe test_baseline_datafest_2_1.py --completed
```

`self-test` comprueba el AUC ponderado con empates, la independencia de las características respecto al objetivo y la regla de mediana. `--completed` verifica además la causalidad temporal, las 288 métricas, los 24 ensembles, la alineación de IDs, los árboles seleccionados, los hashes originales y las probabilidades de diciembre recargando los tres modelos `.cbm`. No entrena modelos.

Estas pruebas de artefactos usan la carpeta fija **`resultados_fase2_validacion/`**.

## 6. Reproducir la validación y la entrega

Mantener todos los scripts y los CSV de entrada en la raíz, además de las carpetas históricas. Elegir una carpeta de salida nueva y vacía; en este ejemplo, `reproduccion_equipo/`:

```powershell
.\.venv\Scripts\python.exe baseline_datafest_2.1.py audit --output reproduccion_equipo
.\.venv\Scripts\python.exe baseline_datafest_2.1.py evaluate --output reproduccion_equipo
.\.venv\Scripts\python.exe baseline_datafest_2.1.py finalize --output reproduccion_equipo
```

| Etapa | Qué hace |
|---|---|
| `audit` | Verifica versiones, hashes y métricas históricas; reproduce agosto/2028; fija el protocolo y guarda el código. Requiere carpeta vacía. |
| `evaluate` | Compara dos historiales y dos reglas de árboles, con tres semillas y junio-noviembre; genera métricas, predicciones, bootstrap y la decisión. |
| `finalize` | Selecciona árboles usando noviembre, refit enero-noviembre, guarda tres modelos y genera el promedio de probabilidades de diciembre. |

Las etapas se ejecutan en ese orden. `evaluate` y `finalize` pueden reanudarse repitiendo su comando en la misma carpeta; Fase 2.1 **no utiliza `--resume`**. El código, las versiones, los datos originales y el número de réplicas deben coincidir con el protocolo fijado. No editar el script entre etapas.

El número predeterminado de réplicas bootstrap es **2.000**, con semilla `20261006`. Si se cambia `--bootstrap`, pasar el mismo valor en las tres etapas. Los parámetros de CatBoost, las semillas, los threads y los meses se mantienen fijos en esta versión; no hay una búsqueda masiva de hiperparámetros.

Los CSV de entrada se leen junto al script; Fase 2.1 no ofrece `--data-dir`. Ejecutar desde la raíz simplifica también el uso de las utilidades auxiliares. La duración depende del equipo y de los checkpoints reutilizados; no existe una prueba rápida de entrenamiento de Fase 2.1 que sustituya esta comparación.

### Informe y utilidades auxiliares

```powershell
.\.venv\Scripts\python.exe informe_validacion_fase2.py
```

Este comando recalcula resúmenes e informe a partir de resultados existentes y puede calcular bootstrap descriptivo por semilla si su archivo aún no existe. Escribe en **`resultados_fase2_validacion/`** y requiere que su finalización de diciembre esté completa. No acepta `--output`: el informe y las pruebas de artefactos **no apuntan automáticamente a `reproduccion_equipo/`**. Para una reproducción en otra carpeta, consultar allí los CSV/JSON generados por el script principal.

`reutilizar_folds_identicos.py` fue una utilidad de optimización de la ejecución original: comprobó equivalencia exacta de características antes de copiar checkpoints junio/julio entre historiales. No es un paso necesario de la reproducción y rechaza destinos que ya existen.

## 7. Receta del modelo actual

| Decisión | Configuración adoptada |
|---|---|
| Algoritmo | CatBoostClassifier, CPU |
| Objetivo de entrenamiento | Logloss |
| Selección de árboles | AUC del mes previo; regla original independiente por semilla |
| Profundidad / L2 / learning rate | `3` / `10` / `0.03` |
| Máximo interno / paciencia | `2500` / `150` |
| Características | 32: 21 originales y 11 derivadas |
| Historial | Completo; todas las observaciones anteriores disponibles |
| Pesos de clase y temporales | Uniformes |
| Semillas | 2026, 2027, 2028 |
| Combinación | Media aritmética de las tres probabilidades |
| Refit final | 110.100 filas enero-noviembre |
| Árboles finales | 713, 1166 y 626, respectivamente |

`id_cliente`, `mes`, `objetivo` y `dias_ultima_interaccion` se excluyen de las entradas del modelo. El ID y el mes se utilizan para construir historial, dividir temporalmente y alinear resultados. `dias_ultima_interaccion` se excluyó tras el diagnóstico de su cambio de comportamiento temporal.

Las cinco categóricas se procesan de forma nativa por CatBoost: `ocupacion`, `region`, `canal_adquisicion`, `banda_riesgo`, `dispositivo_principal`. Las cinco booleanas se convierten a enteros. No se aplica escalado ni one-hot encoding a la receta CatBoost.

Las 11 características derivadas son:

| Variable | Construcción |
|---|---|
| `saldo_ingreso_mensual` | Saldo / (ingreso anual / 12). |
| `saldo_por_producto` | Saldo / número de productos. |
| `visitas_por_producto` | Visitas web / número de productos. |
| `productos_flags` | Suma de indicadores de tarjeta, préstamo y seguro. |
| `movil_x_visitas` | Indicador de móvil × visitas web. |
| `log_ingresos` | `log1p` del ingreso limitado inferiormente a cero. |
| `log_saldo_promedio` | `log1p` del saldo limitado inferiormente a cero. |
| `log_dias_ultima_transaccion` | `log1p` de días limitado inferiormente a cero. |
| `observaciones_previas` | Número de filas anteriores del cliente. |
| `meses_desde_observacion_previa` | Distancia calendario a la última fila anterior; -1 si no existe. |
| `meses_desde_primera_observacion` | Distancia calendario a la primera fila previa disponible; 0 sin historial. |

Las variables históricas no utilizan etiquetas. En la variante `ultimos6`, solo las tres últimas características se restringen al intervalo calendario `[M-6, M)`; las filas antiguas siguen formando parte del entrenamiento. Los datos de entrada deben cumplir el esquema y las validaciones de ausencia de nulos; los resultados indefinidos de transformaciones numéricas se reemplazan por -1.

## 8. Protocolo temporal y resultados

Para evaluar un mes **M**:

1. Entrenar internamente con meses anteriores a **M-1**.
2. Seleccionar el número de árboles en **M-1** mediante early stopping.
3. Reentrenar desde cero con todas las filas anteriores a **M**, usando ese número de árboles.
4. Predecir **M** y calcular AUC/Gini sin ponderar.

La alternativa evaluada fue usar la **mediana de los tres números de árboles seleccionados en M-1**, como número común de refit de las tres semillas. Su regla y el criterio de adopción se fijaron en `protocolo_fijado.json` antes de calcular sus nuevas métricas externas.

| Historial | Regla de árboles | AUC medio junio-noviembre | Gini medio |
|---|---|---:|---:|
| Completo | Original, adoptada | 0,627077 | 0,254154 |
| Seis meses | Original | 0,627127 | 0,254255 |
| Completo | Mediana | 0,627100 | 0,254199 |
| Seis meses | Mediana | 0,627093 | 0,254185 |

La diferencia de AUC de seis meses frente a completo, bajo la regla original, fue **0,000051**, con IC bootstrap 95% **[-0,000251; 0,000357]**. Se conservó completo por ausencia de una ventaja demostrada de la ventana. Esto no demuestra superioridad del historial completo.

En agosto/2028 se reprodujo exactamente el AUC **0,57417** con **55 árboles**. La mediana de los tres números internos (`422`, `500`, `55`) fue **422**, y esa semilla alcanzó AUC **0,63378** con la alternativa. No se encontró desalineación ni error de indexación.

La mediana redujo un **88,6%** la desviación media del AUC entre semillas, pero el AUC medio del ensemble de seis meses bajó **0,000035**. No cumplió el criterio predefinido de igualar o mejorar ese promedio, por lo que se mantuvo la regla original. La diferencia es minúscula y su intervalo incluye cero; no demuestra que la mediana sea peor. **La inestabilidad sigue pendiente en el procedimiento adoptado**, y ninguna semilla se eliminó retrospectivamente.

Se reporta `nuevo` cuando un cliente no tiene observaciones previas en el dataset y `conocido` cuando sí las tiene. Esta definición es distinta del campo bancario `es_nuevo_cliente` y se mantiene aunque una ventana de seis meses no encuentre observaciones recientes.

El bootstrap utiliza clientes como unidad de remuestreo, conserva juntas sus observaciones y usa las mismas multiplicidades para ambos modelos de cada contraste. Los intervalos son condicionales a predicciones fijas; no incorporan toda la incertidumbre del reentrenamiento, selección o cambios futuros de población, y no se ajustaron por multiplicidad.

**Junio-noviembre son evaluación retrospectiva de desarrollo. Noviembre ya fue consultado y no constituye evaluación ciega. No existe un Gini observado de diciembre.** La disponibilidad temporal de los campos originales respecto a la conversión sigue sin documentación suficiente; las comprobaciones del código no resuelven esa limitación de origen.

## 9. Qué archivos consultar o compartir para comparar modelos

Todos los siguientes están en `resultados_fase2_validacion/`:

| Archivo | Uso |
|---|---|
| `INFORME_VALIDACION.md` | Explicación completa del diagnóstico, protocolo, decisión y límites. |
| `metricas.json` | Resumen estructurado de métricas, contrastes y finalización. |
| `metricas_mensuales.csv` | AUC, Gini, n, positivos y logloss por modelo, semilla, mes y segmento. |
| `resumen.csv` | Medias, desviaciones mensuales y extremos por modelo/semilla/segmento. |
| `predicciones_validacion.csv` | Predicciones alineadas con `id_cliente`, `mes`, `objetivo`, `segmento`, `prediccion`, `modelo`, `semilla`. |
| `bootstrap_pareado.csv` | Diferencias e intervalos entre ensembles, por periodo y segmento. |
| `bootstrap_semillas.csv` | Contrastes descriptivos por semilla; no intervinieron en la adopción. |
| `estabilidad_semillas.csv` | Dispersión entre semillas por mes y receta. |
| `particiones.csv` / `iteraciones.csv` | Filas, positivos y meses de cada partición; árboles elegidos. |
| `verificacion.json` / `metricas_reproducidas.csv` | Evidencia de reproducción de los resultados históricos. |
| `originales_sha256.json` / `protocolo_fijado.json` | Integridad de fuentes y reglas congeladas, con versiones y código. |
| `ajustes/` / `diagnostico_agosto/` | Checkpoints, curvas y predicciones del diagnóstico y la comparación. |
| `diciembre/` | Modelos finales, parámetros, curvas de selección, predicciones y entrega actual. |

`configuracion_adoptada.json` conserva el estado en el momento de decidir, previo al refit final; por eso puede mostrar «final diciembre pendiente». El estado de finalización de referencia se encuentra en `estado_ejecucion.json` y `diciembre/manifiesto.json`.

Para comparar propuestas del equipo, alinear **los mismos clientes y meses**, la definición de segmentos, el momento de disponibilidad de las variables y el tipo de promedio. Utilizar diferencias pareadas y documentar qué meses se consultaron para elegir cada modelo. Las métricas de Fases 1 y 2, limitadas a junio-octubre, no deben ordenarse directamente junto a una media junio-noviembre.

## 10. Etapas históricas y trabajo en equipo

| Script | Función histórica |
|---|---|
| `baseline_datafest.py` | CatBoost + LightGBM, desarrollo agosto-octubre y evaluación de noviembre; generó entregas anteriores. |
| `baseline_datafest_1.0.py` | Búsqueda de profundidad/L2 en CatBoost, control de regresión logística y tres semillas para finalistas, junio-octubre. |
| `baseline_datafest_2.0.py` | 12 combinaciones de pesos/historial con d3/L2=10/2026 y posterior integración de tres configuraciones finalistas. |
| `baseline_datafest_2.1.py` | Comparación controlada de historial y selección de árboles, junio-noviembre, y finalización actual. |

Las opciones `--quick` de Fases 1 y 2 producen pruebas funcionales; sus scores no sirven para elegir un modelo. Para reconstruir esos resultados desde cero, consultar `--help` y usar carpetas nuevas. Fase 2.1 espera sus checkpoints históricos en `resultados_fase1/` y `resultados_fase2/`.

Para nuevas propuestas, crear un script y una carpeta propios, registrar versiones, semillas y criterio de decisión antes de medir nuevas variantes, y conservar los resultados de referencia. Compartir el repositorio completo o un paquete que mantenga esta estructura y los archivos requeridos; evitar compartir únicamente el CSV final, porque no permite auditar la solución.

Según las [instrucciones proporcionadas](instrucciones_adicionales_datafest_2026.md), el equipo entrega **una única solución mediante un representante**, con fecha límite **7 de octubre**, por correo. El equipo debe acordar qué archivo presentar después de comparar sus propuestas. Los scripts de este repositorio generan archivos locales; no realizan el envío.
