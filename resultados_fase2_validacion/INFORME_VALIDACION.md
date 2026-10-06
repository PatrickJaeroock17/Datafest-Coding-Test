# Validación de la candidata Fase 2

Fecha de informe: 6 de octubre de 2026. La candidata es CatBoost d3/L2=10/lr=0.03, 32 variables y pesos uniformes. Este informe no corresponde a los baselines antiguos que ya tenían submission.

## Decisión aplicada

Historial adoptado: **completo**. Regla de árboles: **original**. Predicción: promedio uniforme de las tres semillas 2026/2027/2028. Ninguna semilla se descartó. La decisión aplica las reglas fijadas en protocolo_fijado.json; no demuestra superioridad futura.

La mediana no se adopta porque no satisface la condición predefinida de AUC medio no inferior en la candidata de seis meses. La diferencia puntual es minúscula y su intervalo incluye cero: esto NO demuestra que la mediana sea peor. Se mantiene el control de iteraciones por coherencia con la regla fijada. El incidente de agosto se mitigó con la alternativa, pero la inestabilidad entre semillas sigue pendiente en el procedimiento adoptado.

## Verificación y conservación

Se reprodujeron 30 folds históricos desde sus predicciones con error máximo AUC 0.0. Se verificaron hashes de código, datos, versiones y configuración. Se reprodujo por entrenamiento el fold agosto/2028, con error máximo de predicción cero. originales_sha256.json registra los originales y se comprobó que permanecen intactos.

Los controles de junio-octubre se reutilizan de Fase 1 (historial completo, mismas variables y parámetros) y Fase 2 (seis meses). Las nuevas ejecuciones usan exactamente las versiones originales. No se reutilizan entregas de los baselines antiguos.

Entorno: {"python": "3.14.7", "packages": {"numpy": "2.4.4", "pandas": "3.0.2", "scikit-learn": "1.9.1", "catboost": "1.2.10"}}

## Diagnóstico de agosto/2028

La curva interna entrenada en enero-junio tiene su máximo AUC de julio en el árbol 55: 0.63059954. El algoritmo evaluó 205 árboles y se detuvo tras 150 sin superar ese máximo. La indexación y el refit con 55 árboles son correctos. El refit enero-julio alcanza AUC entrenamiento 0.58387698 y AUC agosto 0.57417008. No se encontró desalineación de IDs, etiquetas o predicciones.

El número de árboles elegido en la muestra interna no mantuvo el rendimiento al refit. Es compatible con insuficiente aprendizaje del refit de esa semilla a 55 árboles, pero no identifica una causa única ni demuestra por sí solo overfitting o drift. Las curvas internas y su reproducción están en diagnostico_agosto/. No se inspeccionaron múltiples números de árboles externos para elegir el mejor.

La única alternativa usa la mediana de los tres números de árboles seleccionados con el mismo mes previo; no usa las etiquetas del mes externo para elegir árboles. En agosto, mediana(422,500,55)=422. Se mantiene cada semilla, cambiando únicamente su número de árboles.

Comparación del incidente, incluyendo el control:

| modelo | semilla | iteraciones | auc | gini | auc_entrenamiento |
| --- | --- | --- | --- | --- | --- |
| ultimos6__original | 2026 | 422.00000000 | 0.63446769 | 0.26893538 | 0.64273988 |
| ultimos6__original | 2027 | 500.00000000 | 0.63419897 | 0.26839794 | 0.64355689 |
| ultimos6__original | 2028 | 55.00000000 | 0.57417008 | 0.14834016 | 0.58387698 |
| ultimos6__original | promedio | No aplica | 0.63315298 | 0.26630596 | No aplica |
| ultimos6__mediana | 2026 | 422.00000000 | 0.63446769 | 0.26893538 | 0.64273988 |
| ultimos6__mediana | 2027 | 422.00000000 | 0.63257462 | 0.26514924 | 0.64113350 |
| ultimos6__mediana | 2028 | 422.00000000 | 0.63377659 | 0.26755319 | 0.64284644 |
| ultimos6__mediana | promedio | No aplica | 0.63372797 | 0.26745594 | No aplica |

## Protocolo fijado antes de evaluar la alternativa

Registro UTC: 2026-10-06T05:30:21.359349+00:00. La hipótesis se formuló tras conocer la anomalía histórica; la comparación sigue siendo de desarrollo, no una prueba independiente.

Adoptar mediana solo si su promedio de AUC mensual del ensemble es >= control, disminuye la media de la desviacion entre semillas por mes y ningun mes pierde mas de 0.002 AUC; en otro caso conservar original. Regla operativa, no prueba de significancia.

Comparar ultimos6 menos completo bajo la regla de iteraciones adoptada: mantener seis meses solo si el limite inferior del IC bootstrap pareado 95% de la diferencia de AUC medio es >0; si no, completo por menor dependencia de una ventana elegida.

Para cada mes M: entrenamiento interno enero a M-2, selección en M-1, refit enero a M-1, evaluación en M. La ventana de seis meses modifica solo tres variables históricas; conserva todas las filas. Nuevo significa sin observación anterior en el dataset, no necesariamente nuevo cliente bancario.

**Noviembre ya fue consultado: es evaluación retrospectiva, no ciega.** No se hace búsqueda de profundidades, L2, variables, pesos, semillas ni pesos de ensemble.

## Muestras y positivos por mes y segmento

| mes | segmento | n | positivos | tasa_positivos |
| --- | --- | --- | --- | --- |
| 202606 | todos | 10350 | 1600 | 0.15458937 |
| 202606 | nuevo | 1834 | 309 | 0.16848419 |
| 202606 | conocido | 8516 | 1291 | 0.15159699 |
| 202607 | todos | 9700 | 1410 | 0.14536082 |
| 202607 | nuevo | 950 | 162 | 0.17052632 |
| 202607 | conocido | 8750 | 1248 | 0.14262857 |
| 202608 | todos | 10050 | 1449 | 0.14417910 |
| 202608 | nuevo | 1760 | 309 | 0.17556818 |
| 202608 | conocido | 8290 | 1140 | 0.13751508 |
| 202609 | todos | 9900 | 1393 | 0.14070707 |
| 202609 | nuevo | 1299 | 218 | 0.16782140 |
| 202609 | conocido | 8601 | 1175 | 0.13661202 |
| 202610 | todos | 10400 | 1628 | 0.15653846 |
| 202610 | nuevo | 1893 | 341 | 0.18013735 |
| 202610 | conocido | 8507 | 1287 | 0.15128718 |
| 202611 | todos | 9500 | 1439 | 0.15147368 |
| 202611 | nuevo | 728 | 143 | 0.19642857 |
| 202611 | conocido | 8772 | 1296 | 0.14774282 |

Noviembre tiene 728 clientes sin observación previa de 9500; diciembre tiene 1839 de 9900. La composición cambia. El AUC global no es un promedio lineal de los AUC de segmentos, por lo que no se ajusta artificialmente el Gini con estas proporciones.

## Números de árboles

| historial | mes | semilla_2026 | semilla_2027 | semilla_2028 | mediana |
| --- | --- | --- | --- | --- | --- |
| completo | 202606 | 874 | 752 | 639 | 752 |
| completo | 202607 | 727 | 606 | 1105 | 727 |
| completo | 202608 | 422 | 500 | 55 | 422 |
| completo | 202609 | 574 | 797 | 879 | 797 |
| completo | 202610 | 316 | 393 | 316 | 316 |
| completo | 202611 | 703 | 496 | 656 | 656 |
| ultimos6 | 202606 | 874 | 752 | 639 | 752 |
| ultimos6 | 202607 | 727 | 606 | 1105 | 727 |
| ultimos6 | 202608 | 422 | 500 | 55 | 422 |
| ultimos6 | 202609 | 574 | 797 | 879 | 797 |
| ultimos6 | 202610 | 463 | 314 | 295 | 314 |
| ultimos6 | 202611 | 579 | 606 | 570 | 579 |

## Resultados globales de cada semilla y del promedio

| modelo | semilla | auc_media | gini_media | auc_std_mensual | gini_std_mensual | auc_min | auc_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| completo__mediana | 2028 | 0.62728992 | 0.25457983 | 0.00594222 | 0.01188444 | 0.61745323 | 0.63377659 |
| ultimos6__original | 2026 | 0.62723114 | 0.25446228 | 0.00729312 | 0.01458623 | 0.61437817 | 0.63446769 |
| ultimos6__mediana | 2028 | 0.62718593 | 0.25437187 | 0.00626785 | 0.01253570 | 0.61698356 | 0.63377659 |
| completo__original | 2026 | 0.62714589 | 0.25429178 | 0.00711315 | 0.01422630 | 0.61515655 | 0.63446769 |
| ultimos6__original | promedio | 0.62712750 | 0.25425500 | 0.00643710 | 0.01287421 | 0.61598846 | 0.63315298 |
| completo__mediana | 2026 | 0.62711886 | 0.25423772 | 0.00718980 | 0.01437960 | 0.61503844 | 0.63446769 |
| completo__mediana | promedio | 0.62709973 | 0.25419945 | 0.00657866 | 0.01315732 | 0.61604027 | 0.63372797 |
| ultimos6__mediana | promedio | 0.62709255 | 0.25418509 | 0.00645565 | 0.01291131 | 0.61604432 | 0.63372797 |
| completo__original | promedio | 0.62707683 | 0.25415367 | 0.00657558 | 0.01315117 | 0.61588225 | 0.63315298 |
| ultimos6__original | 2027 | 0.62702811 | 0.25405622 | 0.00634458 | 0.01268916 | 0.61652140 | 0.63419897 |
| ultimos6__mediana | 2026 | 0.62700232 | 0.25400464 | 0.00713454 | 0.01426908 | 0.61437817 | 0.63446769 |
| completo__original | 2027 | 0.62678161 | 0.25356322 | 0.00724940 | 0.01449881 | 0.61480326 | 0.63419897 |
| ultimos6__mediana | 2027 | 0.62676641 | 0.25353281 | 0.00597062 | 0.01194124 | 0.61660778 | 0.63257462 |
| completo__mediana | 2027 | 0.62651666 | 0.25303333 | 0.00656209 | 0.01312418 | 0.61553276 | 0.63284521 |
| completo__original | 2028 | 0.61740265 | 0.23480529 | 0.02179103 | 0.04358205 | 0.57417008 | 0.63167952 |
| ultimos6__original | 2028 | 0.61730177 | 0.23460354 | 0.02181499 | 0.04362999 | 0.57417008 | 0.63207656 |

AUC medio y Gini medio son promedios no ponderados de los seis meses. La desviación mensual no es un intervalo de confianza. No se selecciona la semilla con mayor media.

## Gini mensual de los ensembles

| mes | completo__mediana | completo__original | ultimos6__mediana | ultimos6__original |
| --- | --- | --- | --- | --- |
| 202606 | 0.25206200 | 0.25185614 | 0.25206200 | 0.25185614 |
| 202607 | 0.26018257 | 0.26069502 | 0.26018257 | 0.26069502 |
| 202608 | 0.26745594 | 0.26630596 | 0.26745594 | 0.26630596 |
| 202609 | 0.24811584 | 0.24849609 | 0.24903160 | 0.24957539 |
| 202610 | 0.26529983 | 0.26580428 | 0.26428980 | 0.26512056 |
| 202611 | 0.23208054 | 0.23176450 | 0.23208865 | 0.23197692 |

## Dispersión entre semillas

| modelo | mes | auc_min_semilla | auc_max_semilla | auc_std_semillas | gini_std_semillas |
| --- | --- | --- | --- | --- | --- |
| completo__mediana | 202606 | 0.62508543 | 0.62627264 | 0.00062465 | 0.00124930 |
| completo__mediana | 202607 | 0.62934707 | 0.63030482 | 0.00054679 | 0.00109358 |
| completo__mediana | 202608 | 0.63257462 | 0.63446769 | 0.00095795 | 0.00191591 |
| completo__mediana | 202609 | 0.62358839 | 0.62450947 | 0.00049929 | 0.00099858 |
| completo__mediana | 202610 | 0.63167952 | 0.63306291 | 0.00074386 | 0.00148773 |
| completo__mediana | 202611 | 0.61503844 | 0.61745323 | 0.00127565 | 0.00255131 |
| completo__original | 202606 | 0.62508543 | 0.62590507 | 0.00042867 | 0.00085734 |
| completo__original | 202607 | 0.62937933 | 0.63083909 | 0.00073675 | 0.00147350 |
| completo__original | 202608 | 0.57417008 | 0.63446769 | 0.03473553 | 0.06947106 |
| completo__original | 202609 | 0.62371489 | 0.62436888 | 0.00033830 | 0.00067660 |
| completo__original | 202610 | 0.63167952 | 0.63350778 | 0.00095343 | 0.00190687 |
| completo__original | 202611 | 0.61480326 | 0.61745323 | 0.00143886 | 0.00287772 |
| ultimos6__mediana | 202606 | 0.62508543 | 0.62627264 | 0.00062465 | 0.00124930 |
| ultimos6__mediana | 202607 | 0.62934707 | 0.63030482 | 0.00054679 | 0.00109358 |
| ultimos6__mediana | 202608 | 0.63257462 | 0.63446769 | 0.00095795 | 0.00191591 |
| ultimos6__mediana | 202609 | 0.62374974 | 0.62489529 | 0.00063344 | 0.00126689 |
| ultimos6__mediana | 202610 | 0.63182125 | 0.63228502 | 0.00023277 | 0.00046554 |
| ultimos6__mediana | 202611 | 0.61437817 | 0.61698356 | 0.00140834 | 0.00281667 |
| ultimos6__original | 202606 | 0.62508543 | 0.62590507 | 0.00042867 | 0.00085734 |
| ultimos6__original | 202607 | 0.62937933 | 0.63083909 | 0.00073675 | 0.00147350 |
| ultimos6__original | 202608 | 0.57417008 | 0.63446769 | 0.03473553 | 0.06947106 |
| ultimos6__original | 202609 | 0.62389100 | 0.62550228 | 0.00081376 | 0.00162752 |
| ultimos6__original | 202610 | 0.63207656 | 0.63304261 | 0.00055441 | 0.00110881 |
| ultimos6__original | 202611 | 0.61437817 | 0.61692882 | 0.00137023 | 0.00274047 |

En seis meses de historia, la dispersión media AUC entre semillas pasa de 0.00643989 a 0.00073399. El cambio medio de AUC del ensemble es -0.00003495; su peor cambio mensual es -0.00041538.

## Contrastes pareados del ensemble

Bootstrap por cliente, 2000 réplicas, semilla 20261006. Cada cliente recibe una única multiplicidad por réplica, compartida por sus filas, meses y todos los modelos. Se calcula AUC por mes y luego su media. Los empates se ponderan por mitades. Los CSV incluyen IC por mes, segmento y agregado, además de junio-octubre por separado.

| comparacion | segmento | delta_auc | ic95_auc_inferior | ic95_auc_superior | delta_gini | ic95_gini_inferior | ic95_gini_superior |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ultimos6__mediana - ultimos6__original | todos | -0.00003495 | -0.00051459 | 0.00043251 | -0.00006991 | -0.00102919 | 0.00086502 |
| ultimos6__mediana - ultimos6__original | nuevo | -0.00022549 | -0.00120523 | 0.00079226 | -0.00045099 | -0.00241045 | 0.00158453 |
| ultimos6__mediana - ultimos6__original | conocido | 0.00010549 | -0.00042543 | 0.00062548 | 0.00021098 | -0.00085086 | 0.00125096 |
| completo__mediana - completo__original | todos | 0.00002289 | -0.00044885 | 0.00048812 | 0.00004579 | -0.00089771 | 0.00097625 |
| completo__mediana - completo__original | nuevo | -0.00033588 | -0.00133385 | 0.00069631 | -0.00067176 | -0.00266769 | 0.00139262 |
| completo__mediana - completo__original | conocido | 0.00020385 | -0.00032665 | 0.00072433 | 0.00040770 | -0.00065330 | 0.00144867 |
| ultimos6__original - completo__original | todos | 0.00005067 | -0.00025072 | 0.00035716 | 0.00010133 | -0.00050144 | 0.00071432 |
| ultimos6__original - completo__original | nuevo | -0.00010821 | -0.00075804 | 0.00052293 | -0.00021642 | -0.00151609 | 0.00104587 |
| ultimos6__original - completo__original | conocido | 0.00008582 | -0.00022855 | 0.00041334 | 0.00017163 | -0.00045711 | 0.00082667 |
| ultimos6__mediana - completo__mediana | todos | -0.00000718 | -0.00031460 | 0.00030850 | -0.00001436 | -0.00062920 | 0.00061700 |
| ultimos6__mediana - completo__mediana | nuevo | 0.00000218 | -0.00066553 | 0.00065876 | 0.00000436 | -0.00133107 | 0.00131751 |
| ultimos6__mediana - completo__mediana | conocido | -0.00001255 | -0.00034918 | 0.00033627 | -0.00002509 | -0.00069836 | 0.00067254 |

Los intervalos son condicionales a predicciones fijas. No incorporan reentrenamiento, selección de políticas, selección previa del modelo ni cambios futuros de población. No hay ajuste por multiplicidad; los desgloses son descriptivos. No son intervalos de rendimiento esperado en diciembre.

## Contrastes descriptivos por semilla

Calculados adicionalmente, sin intervenir en la decisión adoptada. bootstrap_semillas.csv contiene también los desgloses por mes y segmento.

| comparacion | semilla | delta_auc | ic95_auc_inferior | ic95_auc_superior |
| --- | --- | --- | --- | --- |
| ultimos6__mediana - ultimos6__original | 2026 | -0.00022882 | -0.00056833 | 0.00011494 |
| ultimos6__mediana - ultimos6__original | 2027 | -0.00026170 | -0.00050620 | -0.00003622 |
| ultimos6__mediana - ultimos6__original | 2028 | 0.00988416 | 0.00770806 | 0.01215793 |
| completo__mediana - completo__original | 2026 | -0.00002703 | -0.00032174 | 0.00026516 |
| completo__mediana - completo__original | 2027 | -0.00026495 | -0.00056091 | 0.00003179 |
| completo__mediana - completo__original | 2028 | 0.00988727 | 0.00771718 | 0.01211841 |
| ultimos6__original - completo__original | 2026 | 0.00008525 | -0.00040221 | 0.00057503 |
| ultimos6__original - completo__original | 2027 | 0.00024650 | -0.00017939 | 0.00072313 |
| ultimos6__original - completo__original | 2028 | -0.00010088 | -0.00058174 | 0.00037056 |
| ultimos6__mediana - completo__mediana | 2026 | -0.00011654 | -0.00058366 | 0.00031241 |
| ultimos6__mediana - completo__mediana | 2027 | 0.00024974 | -0.00019430 | 0.00069900 |
| ultimos6__mediana - completo__mediana | 2028 | -0.00010398 | -0.00057138 | 0.00037365 |

## Resultados mensuales por segmento

Se muestran n y positivos junto a Gini para evitar interpretar segmentos pequeños como igualmente precisos. El CSV metricas_mensuales.csv incluye todos los modelos, semillas y segmentos.

| modelo | mes | segmento | n | positivos | auc | gini |
| --- | --- | --- | --- | --- | --- | --- |
| completo__original | 202606 | nuevo | 1834 | 309 | 0.65653563 | 0.31307125 |
| completo__original | 202606 | conocido | 8516 | 1291 | 0.61720305 | 0.23440610 |
| completo__mediana | 202606 | nuevo | 1834 | 309 | 0.65679452 | 0.31358905 |
| completo__mediana | 202606 | conocido | 8516 | 1291 | 0.61727627 | 0.23455255 |
| completo__original | 202607 | nuevo | 950 | 162 | 0.66690951 | 0.33381901 |
| completo__original | 202607 | conocido | 8750 | 1248 | 0.62516385 | 0.25032769 |
| completo__mediana | 202607 | nuevo | 950 | 162 | 0.66694084 | 0.33388168 |
| completo__mediana | 202607 | conocido | 8750 | 1248 | 0.62489971 | 0.24979941 |
| completo__original | 202608 | nuevo | 1760 | 309 | 0.63586769 | 0.27173537 |
| completo__original | 202608 | conocido | 8290 | 1140 | 0.62958839 | 0.25917679 |
| completo__mediana | 202608 | nuevo | 1760 | 309 | 0.63359941 | 0.26719883 |
| completo__mediana | 202608 | conocido | 8290 | 1140 | 0.63124107 | 0.26248215 |
| completo__original | 202609 | nuevo | 1299 | 218 | 0.60736746 | 0.21473491 |
| completo__original | 202609 | conocido | 8601 | 1175 | 0.62573133 | 0.25146266 |
| completo__mediana | 202609 | nuevo | 1299 | 218 | 0.60690068 | 0.21380136 |
| completo__mediana | 202609 | conocido | 8601 | 1175 | 0.62561798 | 0.25123597 |
| completo__original | 202610 | nuevo | 1893 | 341 | 0.64702437 | 0.29404873 |
| completo__original | 202610 | conocido | 8507 | 1287 | 0.62753123 | 0.25506245 |
| completo__mediana | 202610 | nuevo | 1893 | 341 | 0.64685620 | 0.29371240 |
| completo__mediana | 202610 | conocido | 8507 | 1287 | 0.62726089 | 0.25452178 |
| completo__original | 202611 | nuevo | 728 | 143 | 0.59060427 | 0.18120854 |
| completo__original | 202611 | conocido | 8772 | 1296 | 0.61674168 | 0.23348336 |
| completo__mediana | 202611 | nuevo | 728 | 143 | 0.59120196 | 0.18240392 |
| completo__mediana | 202611 | conocido | 8772 | 1296 | 0.61688669 | 0.23377338 |
| ultimos6__original | 202606 | nuevo | 1834 | 309 | 0.65653563 | 0.31307125 |
| ultimos6__original | 202606 | conocido | 8516 | 1291 | 0.61720305 | 0.23440610 |
| ultimos6__mediana | 202606 | nuevo | 1834 | 309 | 0.65679452 | 0.31358905 |
| ultimos6__mediana | 202606 | conocido | 8516 | 1291 | 0.61727627 | 0.23455255 |
| ultimos6__original | 202607 | nuevo | 950 | 162 | 0.66690951 | 0.33381901 |
| ultimos6__original | 202607 | conocido | 8750 | 1248 | 0.62516385 | 0.25032769 |
| ultimos6__mediana | 202607 | nuevo | 950 | 162 | 0.66694084 | 0.33388168 |
| ultimos6__mediana | 202607 | conocido | 8750 | 1248 | 0.62489971 | 0.24979941 |
| ultimos6__original | 202608 | nuevo | 1760 | 309 | 0.63586769 | 0.27173537 |
| ultimos6__original | 202608 | conocido | 8290 | 1140 | 0.62958839 | 0.25917679 |
| ultimos6__mediana | 202608 | nuevo | 1760 | 309 | 0.63359941 | 0.26719883 |
| ultimos6__mediana | 202608 | conocido | 8290 | 1140 | 0.63124107 | 0.26248215 |
| ultimos6__original | 202609 | nuevo | 1299 | 218 | 0.60767723 | 0.21535445 |
| ultimos6__original | 202609 | conocido | 8601 | 1175 | 0.62636109 | 0.25272218 |
| ultimos6__mediana | 202609 | nuevo | 1299 | 218 | 0.60805065 | 0.21610130 |
| ultimos6__mediana | 202609 | conocido | 8601 | 1175 | 0.62600547 | 0.25201093 |
| ultimos6__original | 202610 | nuevo | 1893 | 341 | 0.64626856 | 0.29253711 |
| ultimos6__original | 202610 | conocido | 8507 | 1287 | 0.62736248 | 0.25472496 |
| ultimos6__mediana | 202610 | nuevo | 1893 | 341 | 0.64625722 | 0.29251444 |
| ultimos6__mediana | 202610 | conocido | 8507 | 1287 | 0.62684096 | 0.25368193 |
| ultimos6__original | 202611 | nuevo | 728 | 143 | 0.59040105 | 0.18080210 |
| ultimos6__original | 202611 | conocido | 8772 | 1296 | 0.61679556 | 0.23359111 |
| ultimos6__mediana | 202611 | nuevo | 728 | 143 | 0.59066404 | 0.18132807 |
| ultimos6__mediana | 202611 | conocido | 8772 | 1296 | 0.61684386 | 0.23368772 |

## Procedimiento final de diciembre

Historial completo; regla original. Entrenamiento interno enero-octubre, selección en noviembre para cada semilla. Números originales: {'2026': 713, '2027': 1166, '2028': 626}. Árboles del refit: {'2026': 713, '2027': 1166, '2028': 626}. Refit con 110100 filas enero-noviembre y predict_proba para 9900 filas de diciembre. Promedio uniforme de tres probabilidades, alineado al sample_submission.

Se guardan modelos .cbm, parámetros efectivos, curvas de selección, predicciones por semilla, hashes y submission_fase2_validada.csv. No hay métricas de diciembre porque no se dispone de sus etiquetas. El archivo es nuevo y no se envió a ninguna plataforma.

configuracion_adoptada.json conserva el estado en la fecha de decisión, anterior al refit final. La finalización se registra en diciembre/manifiesto.json y estado_ejecucion.json.

## Reproducción

Con Python 3.14.7 y las versiones registradas, manteniendo baseline_datafest_2.0.py junto al script nuevo:

```powershell
python baseline_datafest_2.1.py audit --output nueva_validacion
python baseline_datafest_2.1.py evaluate --output nueva_validacion
python baseline_datafest_2.1.py finalize --output nueva_validacion
```

audit requiere carpeta vacía. evaluate y finalize reanudan en la misma carpeta y verifican el manifiesto. El informe y las pruebas de artefactos usan por defecto resultados_fase2_validacion. No editar el script congelado entre etapas; un cambio requiere otra ejecución documentada.

## Límites y conclusión

La mediana no se adopta porque no satisface la condición predefinida de AUC medio no inferior en la candidata de seis meses. La diferencia puntual es minúscula y su intervalo incluye cero: esto NO demuestra que la mediana sea peor. Se mantiene el control de iteraciones por coherencia con la regla fijada. El incidente de agosto se mitigó con la alternativa, pero la inestabilidad entre semillas sigue pendiente en el procedimiento adoptado.

La estabilidad se juzga por los resultados observados de todas las semillas y meses; una mejora en el incidente de agosto no garantiza la desaparición de inestabilidad futura. La disponibilidad temporal de los campos bancarios originales sigue sin documentación de origen; las pruebas de causalidad del código no sustituyen esa garantía.

El historial se elige mediante la regla ya registrada; si el intervalo de seis meses frente a completo incluye cero, se adopta completo por simplicidad, no porque se haya demostrado que es superior. Se conservan las tres semillas y su promedio. Los resultados son evidencia retrospectiva de desarrollo, no una promesa del score de diciembre.
