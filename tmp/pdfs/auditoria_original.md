**Conservaría este modelo como candidato del equipo, pero no lo presentaría como ganador definitivo ni como un modelo con Gini esperado de 0,2596 en diciembre.** Ese es su mejor resultado de desarrollo, después de seleccionar configuraciones sobre los mismos meses.

Realicé una auditoría de lectura: no modifiqué archivos ni entrené modelos. Recalculé métricas desde las predicciones, comprobé los ensembles y contrasté las huellas de los datos y del código con los manifiestos guardados.

**1. Resumen ejecutivo.** La candidata actual es **CatBoost `depth=3`, `l2_leaf_reg=10`, semilla 2026**, con 32 variables y las características históricas limitadas a los seis meses anteriores. Conserva todas las filas disponibles para entrenar. No utiliza ponderación de clases ni decaimiento temporal: ambos se probaron, pero no ganaron.

Su resultado verificado es **AUC mensual medio 0,629802 y Gini medio 0,259603**, evaluando junio–octubre mediante entrenamiento temporal, selección de iteraciones en el mes anterior y reentrenamiento antes del mes evaluado.

La desviación mensual es **0,004114 para AUC y 0,008229 para Gini**. La mejora frente al control con historial completo es solamente **0,000258 de AUC**; un bootstrap pareado por cliente no permite distinguirla claramente de cero.

Sus tres principales limitaciones son:

- Los meses de evaluación también se utilizaron para seleccionar modelos, características y políticas de ponderación.
- Hay inestabilidad entre semillas: en agosto, una semilla de la misma configuración obtuvo AUC 0,57417 frente a 0,63447 de la semilla de exploración.
- No está documentado el instante de disponibilidad de las variables originales respecto a la conversión.

**Veredicto:** candidato válido para comparación común, con reservas. No encontré fuga directa confirmada en la construcción de características, pero falta una evaluación independiente de la candidata actual y todavía no existe una entrega de diciembre producida por Fase 2.

Fuente principal: [resultados completos de Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase2/metricas.json>).

---

**2. Inventario y reproducibilidad.** Hay tres versiones principales y una prueba funcional claramente diferenciada.

| Versión | Contenido | Evidencia de ejecución | Estado |
|---|---|---|---|
| Baseline original | CatBoost + LightGBM; variantes banking/history; selección de blend | Predicciones de desarrollo, noviembre y diciembre | Ejecutado |
| Fase 1 | Búsqueda de CatBoost, tres semillas para finalistas y logística L2 | `status=completo`, `quick=false`, métricas y predicciones | Ejecutado |
| Fase 2 | 12 combinaciones de pesos e historial; integración de tres finalistas | `status=completo`, `quick=false`, métricas y predicciones | Ejecutado |
| `prueba_fase1` | Ejecución reducida | `quick=true` | Solo prueba funcional; excluir de la comparación |

Los scripts son independientes: Fase 2 contiene sus propias funciones y no importa Fase 1.

- [Baseline original](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest.py>)
- [Fase 1](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_1.0.py>)
- [Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_2.0.py>)
- [Pruebas de Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/test_baseline_datafest_2_0.py>)

Funciones relevantes de Fase 2:

| Función | Responsabilidad |
|---|---|
| `load_data` | Validación y selección de enero–octubre |
| `history_features` | Características bancarias e historial completo |
| `build_features` | Restricción de las tres variables históricas a seis meses |
| `training_weights` | Pesos temporales y de clase |
| `fold_indices` | Particiones temporales |
| `run_fold` | Early stopping, reentrenamiento y predicción externa |
| `run_cached` | Persistencia de resultados y reanudación |
| `segment_metrics` / `summarize` | Métricas por segmento y agregación mensual |
| `main` | Exploración, selección de combinación e integración |

**Entorno registrado en ambas fases:**

| Componente | Versión/configuración |
|---|---|
| Python | 3.14.7 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.9.1 |
| CatBoost | 1.2.10 |
| Semillas | 2026, 2027, 2028 |
| Hilos | 4 |
| GPU | No solicitada en el código |

Son las versiones registradas en los manifiestos, no una nueva verificación del entorno que ejecutó el entrenamiento. Las huellas SHA-256 del código actual y de los datos enero–octubre coinciden con esos manifiestos. [Configuración Fase 1](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase1/configuracion.json>), [configuración Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase2/configuracion.json>).

Para reproducir las ejecuciones completas, desde la carpeta del proyecto y con las versiones registradas:

```powershell
python baseline_datafest_1.0.py --output reproduccion_fase1 --threads 4 --iterations 2500 --patience 150 --top-k 3 --seeds 2026 2027 2028

python baseline_datafest_2.0.py --output reproduccion_fase2 --threads 4 --iterations 2500 --patience 150 --half-life 3
```

No ejecuté estos comandos durante la auditoría.

**Artefactos disponibles:** predicciones, métricas, configuraciones, curvas internas, iteraciones elegidas y tiempos por ajuste. **No se guardan los modelos entrenados**: no hay serialización de CatBoost ni de sus árboles. `--resume` reutiliza predicciones y metadatos; no restaura un modelo parcialmente entrenado.

**Coste registrado:**

- Fase 1: suma de tiempos por ajuste de **6.787 segundos, aproximadamente 113 minutos**.
- Fase 2: **7.412 segundos, aproximadamente 124 minutos**.
- No hay medición completa del tiempo de pared, consumo máximo de RAM ni identificación del procesador.
- Los tiempos anteriores excluyen parte del procesamiento y escritura de archivos.

---

**3. Datos y población.** Las dimensiones y las reglas principales coinciden con la documentación.

| Comprobación | Resultado |
|---|---:|
| Train | 110.100 filas × 25 columnas |
| Test | 9.900 filas × 24 columnas |
| Clientes únicos en train | 24.628 |
| Clientes de train con más de una observación | 20.110 |
| Clientes únicos entre train y test | 26.467 |
| Conversiones en train | 16.567; 15,05% |
| Faltantes en train/test | 0 |
| Duplicados cliente-mes | 0 |
| Observaciones posteriores a una conversión | 0 |
| Categorías de test no presentes en train | 0 |
| IDs y orden de `sample_submission` frente a test | Coinciden |

La unidad de observación es **cliente-mes**, no cliente único. Las 110.100 filas no constituyen 110.100 clientes independientes.

“Nuevos” significa **sin observaciones en meses anteriores del archivo**, no necesariamente clientes recién incorporados al banco. Esta definición tampoco equivale al campo original `es_nuevo_cliente`.

| Mes | Filas | Positivos | Conversión | Nuevos observados | Nuevos, % |
|---|---:|---:|---:|---:|---:|
| Enero | 10.400 | 1.521 | 14,63% | 10.400 | 100,00% |
| Febrero | 9.800 | 1.543 | 15,74% | 921 | 9,40% |
| Marzo | 10.300 | 1.527 | 14,83% | 2.043 | 19,83% |
| Abril | 9.600 | 1.473 | 15,34% | 827 | 8,61% |
| Mayo | 10.100 | 1.584 | 15,68% | 1.973 | 19,53% |
| Junio | 10.350 | 1.600 | 15,46% | 1.834 | 17,72% |
| Julio | 9.700 | 1.410 | 14,54% | 950 | 9,79% |
| Agosto | 10.050 | 1.449 | 14,42% | 1.760 | 17,51% |
| Septiembre | 9.900 | 1.393 | 14,07% | 1.299 | 13,12% |
| Octubre | 10.400 | 1.628 | 15,65% | 1.893 | 18,20% |
| Noviembre | 9.500 | 1.439 | 15,15% | 728 | 7,66% |
| Diciembre | 9.900 | Desconocido | Desconocida | 1.839 | 18,58% |

Enero aparece como 100% nuevo por el inicio del archivo: **no prueba que todos fueran nuevos para el banco**.

Las fases 1 y 2 emplean 100.600 filas de enero–octubre. No realizan submuestreo, SMOTE ni exclusiones basadas en pertenecer a test.

**Anomalías y cambios relevantes:**

- **21 de las 22 características originales permanecen constantes por cliente**, incluyendo ingresos, saldo, antigüedad y recencia de transacciones.
- Solo `dias_ultima_interaccion` cambia por cliente; está excluida de la candidata.
- Su correlación con `dias_ultima_transaccion` pasa de **1,00 en enero a 0,102 en noviembre y −0,003 en diciembre**.
- Las medias numéricas noviembre–diciembre cambian poco: la mayor diferencia estandarizada observada es aproximadamente **0,014 desviaciones**, en ingresos. Esto no descarta cambios en relaciones entre variables y objetivo.
- Diciembre contiene bastante más proporción de nuevos que noviembre.

**Disponibilidad temporal no verificada:** los documentos describen los campos, pero no precisan si saldo, productos, visitas o recencias se capturaron antes de la conversión. Que estén en el CSV no demuestra disponibilidad al momento de predecir.

Fuentes: [descripción del conjunto](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/DATASET_DESCRIPTION.md>) y [metadatos](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/metaData.csv>).

---

**4. Diccionario completo de variables y campos.** La candidata recibe **32 variables: 21 originales y 11 derivadas**.

Para leer las tablas:

- **C:** constante por cliente en los archivos observados.
- **T:** disponibilidad anterior al evento no documentada.
- Los campos originales corresponden al registro del mes; no se conoce el instante exacto del corte.
- `load_data` rechaza faltantes originales. No se aprendieron imputadores globales.

| Campo original | Tipo observado | Significado/unidad | Uso y observación |
|---|---|---|---|
| `id_cliente` | Entero | Identificador | Excluido de X; agrupación, alineación y segmentación |
| `mes` | Entero AAAAMM | Mes de observación | Excluido de X; particiones e historial |
| `edad` | Entero | Años | Predictor; C, T |
| `ingresos` | Decimal | Ingreso anual; moneda no documentada | Predictor; C, T |
| `ratio_deuda_ingresos` | Decimal | Proporción deuda/ingresos | Predictor; C, T |
| `antiguedad_cuenta_meses` | Entero | Meses de antigüedad | Predictor; C, T; no aumenta entre meses |
| `numero_productos` | Entero | Productos bancarios activos | Predictor; C, T |
| `saldo_promedio` | Decimal | Saldo; moneda y periodo de promedio no precisados | Predictor; C, T |
| `dias_ultima_transaccion` | Entero | Días desde última transacción | Predictor; C, T; recencia congelada por cliente |
| `ocupacion` | Texto | Categoría laboral | Predictor categórico; C, T |
| `region` | Texto | Región | Predictor categórico; C, T |
| `canal_adquisicion` | Texto | Canal de adquisición | Predictor categórico; C, T |
| `banda_riesgo` | Texto | Banda de riesgo | Predictor categórico; C, T; no se impone orden numérico |
| `tiene_tarjeta_credito` | Booleano → entero 0/1 | Tenencia de tarjeta | Predictor; C, T |
| `activo_movil` | Booleano → entero 0/1 | Actividad móvil | Predictor; C, T |
| `es_nuevo_cliente` | Booleano → entero 0/1 | Indicador original de cliente nuevo | Predictor; C, T; distinto de nuestro segmento “nuevo” |
| `tiene_prestamo` | Booleano → entero 0/1 | Tenencia de préstamo | Predictor; C, T |
| `antiguedad_direccion_meses` | Entero | Meses asociados a la dirección | Predictor; C, T |
| `visitas_web_ultimos_90_dias` | Entero | Visitas en 90 días | Predictor; C, T; corte de los 90 días no documentado |
| `distancia_sucursal_km` | Decimal | Kilómetros | Predictor; C, T |
| `dispositivo_principal` | Texto | Dispositivo registrado | Predictor categórico; C, T |
| `dia_preferido_pago` | Entero | Día del mes | Predictor; C, T; representación numérica, no cíclica |
| `tiene_seguro` | Booleano → entero 0/1 | Tenencia de seguro | Predictor; C, T |
| `dias_ultima_interaccion` | Entero | Días desde última interacción | Excluido por inestabilidad temporal observada |
| `objetivo` | Entero 0/1 | Primera conversión del mes | Etiqueta; nunca predictor |

Las categorías observadas son:

- `ocupacion`: clerical, manager, manual, professional, self_employed.
- `region`: central, east, north, south, west.
- `canal_adquisicion`: branch, call_center, mobile, partner, web.
- `banda_riesgo`: high, low, medium.
- `dispositivo_principal`: android, ios, otro, web.

CatBoost recibe esas cinco columnas como categóricas. No existe un diccionario externo de target encoding calculado sobre todo el conjunto.

Para las derivadas, \(m\) es el índice mensual y \(H_m\) contiene las observaciones del cliente en **\([m-6,m)\)** en la candidata actual.

| Variable derivada | Tipo | Fórmula exacta o definición | Tratamiento y riesgo |
|---|---|---|---|
| `saldo_ingreso_mensual` | Decimal | `saldo_promedio / (ingresos / 12)` | Denominador 0 → NaN → −1; hereda T |
| `saldo_por_producto` | Decimal | `saldo_promedio / numero_productos` | Denominador 0 → −1; hereda T |
| `visitas_por_producto` | Decimal | `visitas_web_ultimos_90_dias / numero_productos` | Denominador 0 → −1; hereda T |
| `productos_flags` | Entero | Tarjeta + préstamo + seguro | Suma de tres indicadores; no equivale necesariamente a `numero_productos` |
| `movil_x_visitas` | Entero | `activo_movil × visitas_web_ultimos_90_dias` | Interacción de dos campos del mes |
| `log_ingresos` | Decimal | `log(1 + max(ingresos, 0))` | Valores negativos se recortarían a 0 |
| `log_saldo_promedio` | Decimal | `log(1 + max(saldo_promedio, 0))` | Mismo tratamiento |
| `log_dias_ultima_transaccion` | Decimal | `log(1 + max(dias_ultima_transaccion, 0))` | Mismo tratamiento |
| `observaciones_previas` | Entero | Número de observaciones en \(H_m\) | 0 si no hay historial en la ventana; no usa objetivos |
| `meses_desde_observacion_previa` | Decimal | \(m-\max(H_m)\) | −1 si no hay observación previa en la ventana |
| `meses_desde_primera_observacion` | Entero | \(m-\min(H_m)\) | 0 si no hay historial; con ventana significa primera **dentro de la ventana** |

La ventana es de **meses calendario, no de seis filas**. El mes actual nunca cuenta como pasado.

En el control `completo`, las tres últimas variables utilizan todo el pasado disponible. En `decay`, también utilizan todo el pasado: el decaimiento modifica los pesos de entrenamiento, no las características.

Los cocientes no encontraron denominadores cero en los datos auditados: ingreso mínimo 18.000 y número mínimo de productos 1. El código sustituye infinitos y NaN derivados por −1, sin ajustar estadísticas de imputación.

Una precisión semántica: `meses_desde_primera_observacion` no representa antigüedad real del cliente. Está condicionada al inicio del archivo y, en la candidata, al límite de seis meses.

Implementación: [características base](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_2.0.py:49>) y [ventana histórica](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_2.0.py:171>).

Los campos auxiliares y de salida tampoco entran al modelo:

| Campo(s) | Tipo/uso |
|---|---|
| `_order`, `_row` | Enteros temporales para restaurar alineación |
| `_month_index` | Índice mensual entero; auxiliar de cálculo |
| `segmento` | Texto: nuevo/conocido según todo el pasado disponible |
| `prediccion` | Decimal entre 0 y 1; salida de `predict_proba`, sin calibración posterior |
| `modelo`, `familia`, `semilla` | Identificación de configuración, algoritmo y aleatoriedad |
| `combinacion`, `politica_temporal`, `modo_peso_clase`, `etapa` | Identificación del experimento |
| `n`, `positivos`, `tasa_positivos` | Tamaño y composición del grupo evaluado |
| `auc`, `gini`, `logloss` | Métricas externas sin ponderar |
| `iteraciones`, `auc_entrenamiento` | Número de árboles e indicador de ajuste en entrenamiento |
| `factor_positivo_interno`, `factor_positivo_refit` | Multiplicadores realmente utilizados |
| `auc_media`, `auc_std_mensual`, `auc_min`, `auc_max` | Resúmenes de AUC entre meses |
| `gini_media`, `gini_std_mensual` | Resúmenes de Gini |
| `meses_validos`, `observaciones` | Meses con métrica válida y total de filas, no clientes independientes |
| `delta_auc_vs_control` | Diferencia de AUC medio frente a `completo_clase_1` |
| `calibration_month`, `calibration_auc_*` | Mes y métricas de selección de iteraciones; **no calibración de probabilidades** |
| `weights_inner`, `weights_refit` | Conteos, multiplicador, ancla temporal, normalización y resumen de pesos |
| `curves`, `seconds`, `evaluated_iterations`, `iteration_limit_hit` | Curvas internas, tiempo e información de parada |

---

**5. Preprocesamiento y modelado.** La candidata utiliza los siguientes parámetros explícitos:

| Parámetro | Valor |
|---|---|
| Algoritmo | `CatBoostClassifier` |
| Profundidad | 3 |
| Regularización L2 | 10 |
| Learning rate | 0,03 |
| Pérdida de entrenamiento | Logloss |
| Métrica de early stopping | ROC AUC |
| Máximo de iteraciones internas | 2.500 |
| Paciencia | 150 |
| Semilla de exploración | 2026 |
| Categóricas | Las cinco columnas indicadas |
| Escalado | Ninguno para CatBoost |
| Pesos de la candidata | Uniformes |
| Historial de características | Seis meses anteriores |
| Filas de entrenamiento | Todas las anteriores al corte |

Los parámetros restantes proceden de los defaults de CatBoost de la versión registrada. No se guardó una captura de `get_all_params()` ni el modelo serializado para inspeccionar todos los parámetros resueltos.

**Experimentos ejecutados en Fase 2:**

- Multiplicador positivo: 1, 2, 3 o \(N_0/N_1\).
- Política: completo, últimos seis meses o decaimiento.
- Total: 12 combinaciones sobre d3/L2=10/2026.

Para decaimiento:

\[
w_{t,i}=2^{-(m_{\max,\mathrm{entrenamiento}}-m_i)/3}
\]

\[
w_i=\frac{w_{t,i}\,w_{c,i}}
{\operatorname{media}(w_t w_c)}
\]

El factor balanceado utiliza conteos sin ponderar de cada ajuste. En las particiones observadas varió aproximadamente entre **5,547 y 5,681**. Con decaimiento, esto no garantiza igualdad de masas ponderadas entre clases.

Los pesos se aplican una sola vez mediante `sample_weight`. No se combinan adicionalmente con `class_weights` o `auto_class_weights`. **La combinación ganadora no activa ninguno de estos cambios de peso.**

Los ensembles promedian probabilidades:

- Tres semillas por configuración: peso \(1/3\) por semilla.
- Ensemble de nueve modelos: peso \(1/9\) por modelo.
- No se optimizaron pesos de mezcla en Fase 2.
- No se implementó una regla que envíe clientes nuevos a d6 y conocidos a d3.

La logística de control de Fase 1 utiliza one-hot encoding con categorías desconocidas ignoradas, estandarización numérica y regresión logística L2. El preprocesamiento se ajusta dentro de cada entrenamiento. Selecciona `C` en el mes anterior; los valores elegidos para junio–octubre fueron **0,1; 1; 0,1; 10; 10**. No contiene splines: esa transformación suave mencionada como posibilidad no se implementó.

**Diferencia con producción:** las fases 1 y 2 terminan en evaluaciones mensuales. No incluyen el entrenamiento definitivo de la candidata con enero–noviembre ni su predicción de diciembre.

Fuente: [entrenamiento y pesos de Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_2.0.py:202>).

---

**6. Protocolo de validación y auditoría de fugas.** Cada celda de tamaño muestra **filas / clientes únicos / positivos**.

| Mes evaluado | Entrenamiento interno | Selección de iteraciones | Reentrenamiento | Evaluación |
|---|---|---|---|---|
| Junio | Ene–abr: 40.100 / 14.191 / 6.064 | Mayo: 10.100 / 10.100 / 1.584 | Ene–may: 50.200 / 16.164 / 7.648 | 10.350 / 10.350 / 1.600 |
| Julio | Ene–may: 50.200 / 16.164 / 7.648 | Junio: 10.350 / 10.350 / 1.600 | Ene–jun: 60.550 / 17.998 / 9.248 | 9.700 / 9.700 / 1.410 |
| Agosto | Ene–jun: 60.550 / 17.998 / 9.248 | Julio: 9.700 / 9.700 / 1.410 | Ene–jul: 70.250 / 18.948 / 10.658 | 10.050 / 10.050 / 1.449 |
| Septiembre | Ene–jul: 70.250 / 18.948 / 10.658 | Agosto: 10.050 / 10.050 / 1.449 | Ene–ago: 80.300 / 20.708 / 12.107 | 9.900 / 9.900 / 1.393 |
| Octubre | Ene–ago: 80.300 / 20.708 / 12.107 | Septiembre: 9.900 / 9.900 / 1.393 | Ene–sep: 90.200 / 22.007 / 13.500 | 10.400 / 10.400 / 1.628 |

El mes externo no elige sus propias iteraciones. Sin embargo, los resultados externos de junio–octubre sí se utilizan después para elegir configuraciones: son **evaluaciones de desarrollo de configuraciones fijas**, no un test independiente del ganador seleccionado.

| Hallazgo | Clasificación | Evaluación |
|---|---|---|
| Particiones cronológicas, sin split aleatorio | Práctica válida | Reproduce predicción de un mes posterior |
| Clientes repetidos entre entrenamiento y evaluación | Práctica válida | Diciembre contiene 81,42% de clientes conocidos |
| Históricos con observaciones anteriores al mes | Práctica válida | No encontré uso de observaciones futuras en las características |
| Uso de última aparición o presencia futura | No observado | No se utilizan como predictores |
| Filtrar train por pertenencia a test | No observado | No se seleccionan supervivientes de diciembre |
| Objetivo incluido en X o en agregados históricos | No observado | Se elimina antes de construir características |
| Preprocesamiento de la logística aprendido globalmente | No observado | Se ajusta dentro de cada entrenamiento |
| Elegir configuración y reportar su score sobre junio–octubre | Sesgo de selección | Afecta a Fases 1 y 2 |
| Mismo mes para early stopping y evaluación en el baseline original | Optimismo de desarrollo | Corregido estructuralmente en Fases 1 y 2 |
| Uso repetido de noviembre | Sesgo de selección | Sus resultados anteriores motivaron cambios posteriores |
| Disponibilidad temporal de variables originales | Riesgo pendiente | No hay documentación del corte respecto a la conversión |
| Variables estáticas capaces de individualizar clientes | Riesgo de generalización | No demuestra fuga; requiere vigilar nuevos/conocidos |
| Inspección de test para detectar inestabilidad | Uso de información no etiquetada | La exclusión de `dias_ultima_interaccion` estuvo informada también por diciembre |

**Conclusión sobre fugas:** no encontré una fuga directa confirmada en la lógica revisada. Esto no certifica que las variables originales estén libres de información posterior al evento.

En las fases 1 y 2 noviembre se elimina antes de construir características, pero **el proceso completo de desarrollo no fue independiente de noviembre**, porque ya habíamos examinado resultados de ese mes. Tampoco fue completamente independiente de las covariables de diciembre.

Fuente: [particiones y comprobaciones temporales](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/baseline_datafest_2.0.py:105>).

---

**7. Resultados verificados.** Recalculé 405 métricas mes/segmento de Fase 1, 195 de la integración de Fase 2 y los AUC globales de los 60 folds de exploración de Fase 2. Las diferencias frente a los archivos guardados fueron únicamente de precisión numérica, del orden de \(10^{-16}\). Los promedios de los ensembles también coinciden.

La candidata `d3_l2_10`, semilla 2026 y `ultimos6_clase_1`, obtuvo:

| Mes | AUC | Gini | Logloss | AUC entrenamiento | Árboles |
|---|---:|---:|---:|---:|---:|
| Junio | 0,625713 | 0,251426 | 0,414881 | 0,665662 | 874 |
| Julio | 0,630283 | 0,260566 | 0,397308 | 0,653605 | 727 |
| Agosto | 0,634468 | 0,268935 | 0,395905 | 0,642740 | 422 |
| Septiembre | 0,625502 | 0,251005 | 0,390863 | 0,646218 | 574 |
| Octubre | 0,633043 | 0,266085 | 0,416773 | 0,640414 | 463 |

Resumen:

- AUC medio: **0,629802**.
- Gini medio: **0,259603**.
- Desviación mensual AUC: **0,004114**.
- Desviación mensual Gini: **0,008229**.
- AUC mínimo/máximo: **0,625502 / 0,634468**.

Se utiliza la **media simple de cinco AUC mensuales**, no el AUC de todas las filas agrupadas. Así se evita incorporar comparaciones entre clientes de meses distintos al criterio principal.

Resultados por segmento de la candidata:

| Mes | Nuevos: filas / positivos | AUC nuevos | Conocidos: filas / positivos | AUC conocidos |
|---|---|---:|---|---:|
| Junio | 1.834 / 309 | 0,657024 | 8.516 / 1.291 | 0,616744 |
| Julio | 950 / 162 | 0,667411 | 8.750 / 1.248 | 0,625081 |
| Agosto | 1.760 / 309 | 0,633553 | 8.290 / 1.140 | 0,632169 |
| Septiembre | 1.299 / 218 | 0,605594 | 8.601 / 1.175 | 0,627514 |
| Octubre | 1.893 / 341 | 0,647234 | 8.507 / 1.287 | 0,627693 |
| **Media mensual** | **7.736 / 1.339** | **0,642163** | **42.664 / 6.141** | **0,625840** |

El Gini medio es **0,284326 en nuevos y 0,251680 en conocidos**. El global no se obtiene simplemente promediando esos dos AUC: también interviene el orden relativo entre segmentos.

Comparación global de configuraciones:

| Modelo | AUC medio | Gini medio | Desv. mensual AUC |
|---|---:|---:|---:|
| Fase 1, d3/2026, historial completo | 0,629544 | 0,259088 | 0,004486 |
| **Fase 2, d3/2026, últimos seis meses** | **0,629802** | **0,259603** | **0,004114** |
| Fase 2, d3, promedio de tres semillas | 0,629355 | 0,258711 | 0,003817 |
| Fase 2, d6/2026 | 0,628493 | 0,256986 | 0,006734 |
| Fase 2, d6, promedio de tres semillas | 0,628469 | 0,256937 | 0,005837 |
| Fase 2, d4/2026 | 0,628657 | 0,257314 | 0,006563 |
| Fase 2, d4, promedio de tres semillas | 0,627894 | 0,255787 | 0,004835 |
| Fase 2, promedio de nueve modelos | 0,629006 | 0,258011 | 0,004886 |
| Logística de control, Fase 1 | 0,618591 | 0,237181 | 0,005722 |

Para nuevos, d6/2026 obtuvo en Fase 2 **AUC 0,645493 y Gini 0,290985**. Esto no demuestra todavía que una asignación de modelos por segmento mejore el resultado global.

**Pesos de clase e historial.** El orden superior de la exploración fue:

| Combinación | AUC medio |
|---|---:|
| Últimos seis meses, peso 1 | 0,629802 |
| Completo, peso 1 | 0,629544 |
| Completo, peso 3 | 0,628974 |
| Decaimiento, balanceado | 0,628862 |
| Últimos seis meses, peso 3 | 0,628859 |

Ninguna configuración ponderada superó al control sin ponderación. Por tanto, **no hay evidencia aquí de que rebalancear haya mejorado el ranking**.

**Semillas de d3 en Fase 2:**

| Semilla | AUC medio | Gini medio | Desv. mensual AUC |
|---|---:|---:|---:|
| 2026 | 0,629802 | 0,259603 | 0,004114 |
| 2027 | 0,629129 | 0,258259 | 0,004147 |
| 2028 | 0,617376 | 0,234753 | 0,024389 |

El problema más visible está en agosto:

- Semilla 2026: 422 árboles, AUC externo **0,634468**.
- Semilla 2027: 500 árboles, AUC externo **0,634199**.
- Semilla 2028: 55 árboles, AUC externo **0,574170**.

La tercera semilla había conseguido AUC interno **0,630600** en julio. El número de árboles seleccionado no se trasladó de forma estable al reentrenamiento. Esto es compatible con un ajuste insuficiente después del refit, pero no basta para establecer una causa única.

No se alcanzó el límite de 2.500 iteraciones en los ajustes completos auditados de las fases 1 y 2.

**Probabilidades y calibración.** Sobre las 50.400 predicciones de la candidata, calculé adicionalmente:

- Conversión observada: **14,8413%**.
- Probabilidad media: **14,5083%**.
- Logloss global por fila: **0,403388**.
- Brier: **0,121974**.

Son descriptores de desarrollo. La cercanía entre media predicha y tasa observada **no demuestra calibración**. No hay calibrador posterior ni análisis completo de confiabilidad por rangos de probabilidad.

**Noviembre corresponde a modelos anteriores:**

| Versión anterior | AUC noviembre | Gini noviembre |
|---|---:|---:|
| History, CatBoost d5/L2=8 | 0,619341 | 0,238682 |
| History, LightGBM | 0,614772 | 0,229545 |
| History, blend | 0,616903 | 0,233806 |
| Banking, CatBoost d5/L2=8 | 0,619174 | 0,238349 |

Estos resultados **no son una evaluación de d3 de Fase 2** y no deben combinarse con su Gini de desarrollo para estimar una mejora.

Los dos `submission.csv` existentes pertenecen al baseline original. Tienen formato, orden y probabilidades válidos, pero no representan a la candidata actual. No hay resultados públicos de competición ni etiquetas de diciembre disponibles.

Fuentes: [métricas de finalistas](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase2/metricas_finalistas.csv>), [ranking de experimentos](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase2/ranking_experimentos.csv>) y [noviembre del baseline history](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_history/metricas.json>).

---

**8. Incertidumbre y solidez de las mejoras.** La mejora de la candidata frente al control se distribuye así:

| Mes | Diferencia AUC Fase 2 − control |
|---|---:|
| Junio | 0 |
| Julio | 0 |
| Agosto | 0 |
| Septiembre | +0,001310 |
| Octubre | −0,000020 |

**La mejora media procede prácticamente de septiembre.** No constituye una mejora repetida en cinco meses.

Realicé un bootstrap pareado con **2.000 remuestreos por cliente**, conservando juntas las observaciones mensuales de cada cliente. Hay **16.252 clientes únicos** entre las 50.400 filas evaluadas.

| Comparación | Diferencia AUC media | Intervalo percentil aproximado del 95% |
|---|---:|---:|
| Candidata seis meses − control completo | +0,000258 | −0,000195 a +0,000706 |
| Ensemble d3 de tres semillas − d3/2026 | −0,000446 | −0,001211 a +0,000305 |

Ambos intervalos incluyen cero. No tenemos evidencia concluyente de mejora por limitar el historial ni de deterioro por promediar las tres semillas.

**Límites de estos intervalos:** mantienen fijas las predicciones y los modelos entrenados. No incluyen incertidumbre por reentrenamiento, búsqueda de hiperparámetros, elección del ganador ni cambios futuros de población. Tampoco son intervalos del AUC de diciembre.

El proceso de selección incluyó:

- Fase 1: 16 combinaciones profundidad/L2 más el control original.
- Tres finalistas con tres semillas.
- Cuatro valores de regularización logística por fold.
- Fase 2: 12 combinaciones sobre la configuración seleccionada.
- Transferencia de la ganadora a tres modelos y comparación de ensembles.

No son todos ensayos independientes, pero sí son **múltiples oportunidades de seleccionar un resultado favorable sobre los mismos meses**.

La estabilidad mensual de la semilla 2026 no debe confundirse con estabilidad del procedimiento completo. La semilla 2028 muestra una caída importante. Tampoco sería correcto descartarla retrospectivamente solo porque empeoró el promedio.

En diciembre la proporción de nuevos aumenta respecto a noviembre. Esto hace relevante el resultado por segmento, pero no garantiza que el mejor modelo para nuevos mantenga su ventaja ni que sus probabilidades sean comparables con las de otro modelo usado para conocidos.

---

**9. Comparabilidad con otros modelos del equipo.** Esta candidata puede entrar a una comparación común, pero todavía no puede declararse superior a modelos cuyos resultados desconozco.

Para comparar justamente, el equipo debe utilizar:

- Las mismas observaciones cliente-mes y etiquetas.
- El mismo instante supuesto de disponibilidad de información.
- Los mismos meses de entrenamiento, selección y evaluación.
- La misma definición de nuevos/conocidos.
- El mismo AUC mensual y método de agregación.
- Predicciones alineadas por `id_cliente` y `mes`.
- Un registro del presupuesto de búsqueda y del uso previo de noviembre.
- La configuración congelada de cada candidato antes de la comparación acordada.

Formato propuesto:

```text
id_cliente,mes,objetivo,segmento,modelo,semilla,prediccion
```

Las [predicciones de Fase 2](</C:/Users/jaero/OneDrive/Desktop/Datafest Coding Test/resultados_fase2/predicciones_finalistas.csv>) ya contienen esos campos y una columna adicional `combinacion`.

Si todos han utilizado noviembre para ajustar decisiones, una nueva comparación en noviembre puede ser útil, pero **no se convierte en una evaluación ciega por volver a ejecutarla**.

La comparación debería presentar diferencias pareadas por mes y por cliente. No corresponde inventar un “Gini corregido por optimismo” ni aplicar descuentos arbitrarios al score.

---

**10. Veredicto y ficha final.** Recomiendo **conservar d3/2026 y su promedio de tres semillas como candidatos**, manteniendo el control de historial completo para comprobar si la pequeña ventaja de seis meses persiste.

Lo demostrado es que:

- Los archivos y métricas son internamente consistentes.
- El protocolo temporal de Fases 1 y 2 separa selección de iteraciones y evaluación externa.
- La ponderación de clases y el decaimiento no mejoraron el mejor AUC observado.
- La candidata obtiene Gini medio de desarrollo cercano a 0,260.

Lo que no está demostrado es:

- Que alcance ese Gini en diciembre.
- Que seis meses sea realmente superior al historial completo.
- Que una sola semilla represente de forma robusta el procedimiento.
- Que las probabilidades estén calibradas.
- Que los campos originales estuvieran disponibles antes del evento.
- Que supere a los modelos de otros integrantes bajo un protocolo común.

Antes de elegir la entrega, son imprescindibles la revisión del corte temporal de las variables, la comparación común entre candidatos, la documentación del uso previo de noviembre y el entrenamiento/predicción final de la configuración elegida. **No descartaría este trabajo por fuga confirmada; sí exigiría esas comprobaciones antes de presentarlo como el modelo más realista o destacado.**

| Campo para comparar entre integrantes | Candidata principal | Alternativa del mismo proyecto |
|---|---|---|
| Modelo y versión | Fase 2, `cat_d3_l2_10`, semilla 2026 | Fase 2, `cat_d3_l2_10_3semillas` |
| Algoritmo | CatBoost, profundidad 3, L2=10 | Promedio de tres CatBoost con la misma configuración |
| Variables | 32: 21 originales + 11 derivadas | Las mismas 32 |
| Ventana de datos | Entrenamiento creciente desde enero hasta el mes anterior | Igual |
| Historial de características | Seis meses anteriores; excluye el mes actual | Igual |
| Desbalance | Sin ponderación; multiplicador positivo 1 | Igual |
| Decaimiento temporal | No utilizado en la combinación seleccionada | Igual |
| Validación | Junio–octubre; selección en mes anterior y posterior refit | Mismo protocolo para cada semilla |
| AUC / Gini de desarrollo | **0,629802 / 0,259603** | **0,629355 / 0,258711** |
| Desv. mensual AUC / Gini | 0,004114 / 0,008229 | 0,003817 / 0,007635 |
| AUC mínimo–máximo mensual | 0,625502–0,634468 | 0,624788–0,633153 |
| Variación entre semillas | AUC medio: 0,629802; 0,629129; 0,617376 | Promedia 2026, 2027 y 2028; no elimina la semilla desfavorable |
| AUC nuevos / conocidos | 0,642163 / 0,625840 | 0,642652 / 0,625136 |
| Gini nuevos / conocidos | 0,284326 / 0,251680 | 0,285303 / 0,250272 |
| Evaluación independiente de esta versión | **No disponible** | **No disponible** |
| Rendimiento de diciembre | Desconocido | Desconocido |
| Calibración | No demostrada; sin calibrador posterior | No demostrada |
| Riesgos principales | Selección sobre desarrollo, sensibilidad a semillas y corte de variables desconocido | Mismos riesgos; menor desviación mensual observada, sin mejora de AUC demostrada |
| Reproducibilidad | Código/datos con hashes coincidentes; versiones y predicciones registradas; sin modelo serializado | Igual |
| Coste registrado de su validación | ≈9,8 minutos, cinco folds con selección y refit | ≈20,5 minutos, tres semillas y cinco folds |
| Estado de preparación | Candidato para comparación; sin entrega actual de diciembre | Alternativa para comparación; sin entrega actual de diciembre |
| Evidencia pendiente | Disponibilidad temporal, comparación común y evaluación de generalización con selección congelada | La misma; comprobar si la reducción de variabilidad compensa su complejidad |