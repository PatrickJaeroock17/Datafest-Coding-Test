"""Informe reproducible y contrastes descriptivos por semilla; no entrena modelos.
Ejecutar despues de evaluate y finalize de baseline_datafest_2.1.py.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('validation',ROOT/'baseline_datafest_2.1.py')
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
OUT=ROOT/'resultados_fase2_validacion'

def table(frame):
    def cell(x):
        if pd.isna(x): return 'No aplica'
        if isinstance(x,(float,np.floating)): return f'{x:.8f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in frame.itertuples(index=False,name=None)])

def seed_bootstrap(pred):
    target=OUT/'bootstrap_semillas.csv'
    if target.exists(): return pd.read_csv(target)
    # Contrastes secundarios descriptivos: no intervienen en configuracion_adoptada.
    ids=np.sort(pred.id_cliente.unique()); prepared={}; points={}
    for (model,seed,month),g in pred.loc[pred.semilla.ne('promedio')].groupby(['modelo','semilla','mes']):
        for segment in ['todos','nuevo','conocido']:
            p=g if segment=='todos' else g.loc[g.segmento.eq(segment)]
            key=(model,seed,month,segment)
            prepared[key]=v.prepare_auc(p.objetivo.to_numpy(),p.prediccion.to_numpy(),np.searchsorted(ids,p.id_cliente.to_numpy()))
            points[key]=v.score(p)
    keys=list(prepared); ix={k:i for i,k in enumerate(keys)}
    reps=v.readj(OUT/'protocolo_fijado.json')['bootstrap']['replicates']; values=np.empty((reps,len(keys)))
    rng=np.random.default_rng(20261006)
    for b in range(reps):
        counts=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids))
        values[b]=[v.weighted_auc(prepared[k],counts) for k in keys]
        if b%500==0: print(f'Bootstrap descriptivo por semilla {b}/{reps}',flush=True)
    contrasts=[('ultimos6__mediana','ultimos6__original'),('completo__mediana','completo__original'),('ultimos6__original','completo__original'),('ultimos6__mediana','completo__mediana')]
    periods={'junio_noviembre':v.MONTHS,'junio_octubre':v.MONTHS[:-1],**{str(m):(m,) for m in v.MONTHS}}
    rows=[]
    for a,b in contrasts:
        for seed in map(str,v.SEEDS):
            for segment in ['todos','nuevo','conocido']:
                for label,months in periods.items():
                    ai=[ix[(a,seed,m,segment)] for m in months]; bi=[ix[(b,seed,m,segment)] for m in months]
                    diff=values[:,ai].mean(axis=1)-values[:,bi].mean(axis=1)
                    low,high=np.nanquantile(diff,[.025,.975]); point=np.mean([points[(a,seed,m,segment)]-points[(b,seed,m,segment)] for m in months])
                    rows.append(dict(comparacion=a+' - '+b,semilla=seed,segmento=segment,periodo=label,delta_auc=point,delta_gini=2*point,ic95_auc_inferior=low,ic95_auc_superior=high,ic95_gini_inferior=2*low,ic95_gini_superior=2*high,replicas_validas=int(np.isfinite(diff).sum())))
    result=pd.DataFrame(rows); result.to_csv(target,index=False)
    np.savez_compressed(OUT/'bootstrap_semillas_replicas.npz',auc=values,keys=np.array(['|'.join(map(str,k)) for k in keys]))
    return result

def main():
    choice=v.readj(OUT/'configuracion_adoptada.json'); verified=v.readj(OUT/'verificacion.json')
    protocol=v.readj(OUT/'protocolo_fijado.json'); final=v.readj(OUT/'diciembre/manifiesto.json')
    pred=v.readpred(OUT/'predicciones_validacion.csv'); met=pd.read_csv(OUT/'metricas_mensuales.csv'); summary=pd.read_csv(OUT/'resumen.csv')
    boot=pd.read_csv(OUT/'bootstrap_pareado.csv'); seeds_boot=seed_bootstrap(pred)
    allrows=met.loc[met.segmento.eq('todos')]
    stability=allrows.loc[allrows.semilla.ne('promedio')].groupby(['modelo','mes']).agg(auc_min_semilla=('auc','min'),auc_max_semilla=('auc','max'),auc_std_semillas=('auc','std')).reset_index()
    stability['gini_std_semillas']=2*stability.auc_std_semillas
    stability.to_csv(OUT/'estabilidad_semillas.csv',index=False)
    counts=met.loc[met.modelo.eq('completo__original')&met.semilla.eq('2026'),['mes','segmento','n','positivos','tasa_positivos']]
    df=v.load(); parts=[]
    test=pd.read_csv(ROOT/'test.csv')
    n_new_dec=int((~test.id_cliente.isin(df.id_cliente)).sum())
    n_new_nov=int((~df.loc[df.mes.eq(202611),'id_cliente'].isin(df.loc[df.mes.lt(202611),'id_cliente'])).sum())
    for month in v.MONTHS:
        inner,stop,refit,outer=v.base.fold_indices(df,month)
        for name,idx in [('interno',inner),('seleccion',stop),('refit',refit),('externo',outer)]:
            sub=df.iloc[idx]; parts.append(dict(mes_evaluacion=month,particion=name,mes_min=int(sub.mes.min()),mes_max=int(sub.mes.max()),n=len(sub),clientes=sub.id_cliente.nunique(),positivos=int(sub.objetivo.sum())))
    pd.DataFrame(parts).to_csv(OUT/'particiones.csv',index=False)
    iterations=[]
    for policy in v.POLICIES:
        for month in v.MONTHS:
            r=[v.readj(OUT/'ajustes'/policy/'original'/str(s)/str(month)/'ajuste.json')['iterations'] for s in v.SEEDS]
            iterations.append(dict(historial=policy,mes=month,semilla_2026=r[0],semilla_2027=r[1],semilla_2028=r[2],mediana=int(np.median(r))))
    pd.DataFrame(iterations).to_csv(OUT/'iteraciones.csv',index=False)
    resolution=('La mediana no se adopta porque no satisface la condición predefinida de AUC medio no inferior en la candidata de seis meses. La diferencia puntual es minúscula y su intervalo incluye cero: esto NO demuestra que la mediana sea peor. Se mantiene el control de iteraciones por coherencia con la regla fijada. El incidente de agosto se mitigó con la alternativa, pero la inestabilidad entre semillas sigue pendiente en el procedimiento adoptado.'
        if choice['regla_iteraciones']=='original' else
        'Se adopta la mediana porque cumple los criterios operativos registrados. La mejora observada de estabilidad no garantiza estabilidad en otros meses ni demuestra automáticamente superioridad estadística de AUC.')
    sections=['# Validación de la candidata Fase 2',
        'Fecha de informe: 6 de octubre de 2026. La candidata es CatBoost d3/L2=10/lr=0.03, 32 variables y pesos uniformes. Este informe no corresponde a los baselines antiguos que ya tenían submission.',
        '## Decisión aplicada',
        f"Historial adoptado: **{choice['historial']}**. Regla de árboles: **{choice['regla_iteraciones']}**. Predicción: promedio uniforme de las tres semillas 2026/2027/2028. Ninguna semilla se descartó. La decisión aplica las reglas fijadas en protocolo_fijado.json; no demuestra superioridad futura.",
        resolution,
        '## Verificación y conservación',
        f"Se reprodujeron {verified['folds_verificados']} folds históricos desde sus predicciones con error máximo AUC {verified['max_error_auc']}. Se verificaron hashes de código, datos, versiones y configuración. Se reprodujo por entrenamiento el fold agosto/2028, con error máximo de predicción cero. originales_sha256.json registra los originales y se comprobó que permanecen intactos.",
        'Los controles de junio-octubre se reutilizan de Fase 1 (historial completo, mismas variables y parámetros) y Fase 2 (seis meses). Las nuevas ejecuciones usan exactamente las versiones originales. No se reutilizan entregas de los baselines antiguos.',
        'Entorno: '+json.dumps(verified['versiones'],ensure_ascii=False),
        '## Diagnóstico de agosto/2028',
        'La curva interna entrenada en enero-junio tiene su máximo AUC de julio en el árbol 55: 0.63059954. El algoritmo evaluó 205 árboles y se detuvo tras 150 sin superar ese máximo. La indexación y el refit con 55 árboles son correctos. El refit enero-julio alcanza AUC entrenamiento 0.58387698 y AUC agosto 0.57417008. No se encontró desalineación de IDs, etiquetas o predicciones.',
        'El número de árboles elegido en la muestra interna no mantuvo el rendimiento al refit. Es compatible con insuficiente aprendizaje del refit de esa semilla a 55 árboles, pero no identifica una causa única ni demuestra por sí solo overfitting o drift. Las curvas internas y su reproducción están en diagnostico_agosto/. No se inspeccionaron múltiples números de árboles externos para elegir el mejor.',
        'La única alternativa usa la mediana de los tres números de árboles seleccionados con el mismo mes previo; no usa las etiquetas del mes externo para elegir árboles. En agosto, mediana(422,500,55)=422. Se mantiene cada semilla, cambiando únicamente su número de árboles.',
        'Comparación del incidente, incluyendo el control:',
        table(met.loc[met.mes.eq(202608)&met.modelo.str.startswith('ultimos6')&met.segmento.eq('todos'),['modelo','semilla','iteraciones','auc','gini','auc_entrenamiento']]),
        '## Protocolo fijado antes de evaluar la alternativa',
        'Registro UTC: '+protocol['created_utc']+'. La hipótesis se formuló tras conocer la anomalía histórica; la comparación sigue siendo de desarrollo, no una prueba independiente.',
        protocol['iteration_adoption'],protocol['history_adoption'],
        'Para cada mes M: entrenamiento interno enero a M-2, selección en M-1, refit enero a M-1, evaluación en M. La ventana de seis meses modifica solo tres variables históricas; conserva todas las filas. Nuevo significa sin observación anterior en el dataset, no necesariamente nuevo cliente bancario.',
        '**Noviembre ya fue consultado: es evaluación retrospectiva, no ciega.** No se hace búsqueda de profundidades, L2, variables, pesos, semillas ni pesos de ensemble.',
        '## Muestras y positivos por mes y segmento',table(counts),
        f'Noviembre tiene {n_new_nov} clientes sin observación previa de {int(df.mes.eq(202611).sum())}; diciembre tiene {n_new_dec} de {len(test)}. La composición cambia. El AUC global no es un promedio lineal de los AUC de segmentos, por lo que no se ajusta artificialmente el Gini con estas proporciones.',
        '## Números de árboles',table(pd.DataFrame(iterations)),
        '## Resultados globales de cada semilla y del promedio',
        table(summary.loc[summary.segmento.eq('todos'),['modelo','semilla','auc_media','gini_media','auc_std_mensual','gini_std_mensual','auc_min','auc_max']]),
        'AUC medio y Gini medio son promedios no ponderados de los seis meses. La desviación mensual no es un intervalo de confianza. No se selecciona la semilla con mayor media.',
        '## Gini mensual de los ensembles',
        table(met.loc[met.semilla.eq('promedio')&met.segmento.eq('todos')].pivot(index='mes',columns='modelo',values='gini').reset_index()),
        '## Dispersión entre semillas',table(stability),
        f"En seis meses de historia, la dispersión media AUC entre semillas pasa de {choice['dispersion_semillas_original']:.8f} a {choice['dispersion_semillas_mediana']:.8f}. El cambio medio de AUC del ensemble es {choice['diferencia_auc_mediana_vs_original']:.8f}; su peor cambio mensual es {choice['peor_diferencia_mensual']:.8f}.",
        '## Contrastes pareados del ensemble',
        'Bootstrap por cliente, 2000 réplicas, semilla 20261006. Cada cliente recibe una única multiplicidad por réplica, compartida por sus filas, meses y todos los modelos. Se calcula AUC por mes y luego su media. Los empates se ponderan por mitades. Los CSV incluyen IC por mes, segmento y agregado, además de junio-octubre por separado.',
        table(boot.loc[boot.periodo.eq('junio_noviembre'),['comparacion','segmento','delta_auc','ic95_auc_inferior','ic95_auc_superior','delta_gini','ic95_gini_inferior','ic95_gini_superior']]),
        'Los intervalos son condicionales a predicciones fijas. No incorporan reentrenamiento, selección de políticas, selección previa del modelo ni cambios futuros de población. No hay ajuste por multiplicidad; los desgloses son descriptivos. No son intervalos de rendimiento esperado en diciembre.',
        '## Contrastes descriptivos por semilla',
        'Calculados adicionalmente, sin intervenir en la decisión adoptada. bootstrap_semillas.csv contiene también los desgloses por mes y segmento.',
        table(seeds_boot.loc[seeds_boot.periodo.eq('junio_noviembre')&seeds_boot.segmento.eq('todos'),['comparacion','semilla','delta_auc','ic95_auc_inferior','ic95_auc_superior']]),
        '## Resultados mensuales por segmento',
        'Se muestran n y positivos junto a Gini para evitar interpretar segmentos pequeños como igualmente precisos. El CSV metricas_mensuales.csv incluye todos los modelos, semillas y segmentos.',
        table(met.loc[met.semilla.eq('promedio')&met.segmento.ne('todos'),['modelo','mes','segmento','n','positivos','auc','gini']]),
        '## Procedimiento final de diciembre',
        f"Historial {final['history']}; regla {final['rule']}. Entrenamiento interno enero-octubre, selección en noviembre para cada semilla. Números originales: {final['selected_original']}. Árboles del refit: {final['refit_rounds']}. Refit con {final['n_training']} filas enero-noviembre y predict_proba para {final['n_prediction']} filas de diciembre. Promedio uniforme de tres probabilidades, alineado al sample_submission.",
        'Se guardan modelos .cbm, parámetros efectivos, curvas de selección, predicciones por semilla, hashes y submission_fase2_validada.csv. No hay métricas de diciembre porque no se dispone de sus etiquetas. El archivo es nuevo y no se envió a ninguna plataforma.',
        'configuracion_adoptada.json conserva el estado en la fecha de decisión, anterior al refit final. La finalización se registra en diciembre/manifiesto.json y estado_ejecucion.json.',
        '## Reproducción',
        'Con Python 3.14.7 y las versiones registradas, manteniendo baseline_datafest_2.0.py junto al script nuevo:',
        '```powershell\npython baseline_datafest_2.1.py audit --output nueva_validacion\npython baseline_datafest_2.1.py evaluate --output nueva_validacion\npython baseline_datafest_2.1.py finalize --output nueva_validacion\n```',
        'audit requiere carpeta vacía. evaluate y finalize reanudan en la misma carpeta y verifican el manifiesto. El informe y las pruebas de artefactos usan por defecto resultados_fase2_validacion. No editar el script congelado entre etapas; un cambio requiere otra ejecución documentada.',
        '## Límites y conclusión',
        resolution,
        'La estabilidad se juzga por los resultados observados de todas las semillas y meses; una mejora en el incidente de agosto no garantiza la desaparición de inestabilidad futura. La disponibilidad temporal de los campos bancarios originales sigue sin documentación de origen; las pruebas de causalidad del código no sustituyen esa garantía.',
        'El historial se elige mediante la regla ya registrada; si el intervalo de seis meses frente a completo incluye cero, se adopta completo por simplicidad, no porque se haya demostrado que es superior. Se conservan las tres semillas y su promedio. Los resultados son evidencia retrospectiva de desarrollo, no una promesa del score de diciembre.'
    ]
    (OUT/'INFORME_VALIDACION.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    (OUT/'requirements.txt').write_text('numpy==2.4.4\npandas==3.0.2\nscikit-learn==1.9.1\ncatboost==1.2.10\n',encoding='utf-8')
    v.writej(OUT/'metricas.json',dict(status='completo',configuracion_adoptada=choice,
        resumen=v.base.records(summary),metricas_mensuales=v.base.records(met),
        bootstrap_ensemble=v.base.records(boot),bootstrap_semillas=v.base.records(seeds_boot),
        estabilidad=v.base.records(stability),diciembre=final,
        advertencias=['Noviembre no es ciego.','Intervalos condicionales a predicciones fijas; sin ajuste por seleccion o multiplicidad.','No existe Gini observado de diciembre.']))
    v.writej(OUT/'codigo_auxiliar_sha256.json',{name:v.sha(ROOT/name) for name in ['test_baseline_datafest_2_1.py','informe_validacion_fase2.py','reutilizar_folds_identicos.py']})
    v.writej(OUT/'estado_ejecucion.json',dict(status='completo',validacion='junio_noviembre',diciembre='modelos_y_predicciones_generados',enviado_plataforma=False,actualizado_utc=v.stamp()))
    print('Informe y contrastes por semilla guardados.',flush=True)

if __name__=='__main__':
    import sys
    if '--bootstrap-only' in sys.argv:
        seed_bootstrap(v.readpred(OUT/'predicciones_validacion.csv'))
    else:
        main()
