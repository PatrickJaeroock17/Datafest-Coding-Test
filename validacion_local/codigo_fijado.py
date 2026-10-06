"""Validación controlada de la candidata Fase 2. No modifica resultados anteriores.

python baseline_datafest_2.1.py audit
python baseline_datafest_2.1.py evaluate
python baseline_datafest_2.1.py finalize

Requiere el entorno registrado de Fase 2. evaluate exige protocolo fijado por audit.
No hay búsqueda de hiperparámetros; solo selección original vs mediana de 3 semillas.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fase2', ROOT/'baseline_datafest_2.0.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SEEDS = (2026, 2027, 2028)
MONTHS = tuple(range(202606, 202612))
POLICIES = ('completo', 'ultimos6')
CONFIG = dict(id='cat_d3_l2_10', depth=3, l2=10)
KEYS = ['id_cliente', 'mes', 'objetivo', 'segmento']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def writej(path, value): base.json_write(Path(path), value)
def stamp(): return datetime.now(timezone.utc).isoformat()
def versions():
    return dict(python=platform.python_version(), packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost']})
def score(p): return float(roc_auc_score(p.objetivo, p.prediccion))
def readpred(path): return pd.read_csv(path, float_precision='round_trip')

def load():
    df=pd.read_csv(ROOT/'train.csv').sort_values(['mes','id_cliente']).reset_index(drop=True)
    assert set(df.mes)==set(range(202601,202612))
    assert not df.isna().any().any() and not df.duplicated(['id_cliente','mes']).any()
    assert df.objetivo.isin([0,1]).all()
    assert df.groupby('mes').objetivo.nunique().eq(2).all()
    ordered=df.sort_values(['id_cliente','mes'])
    assert not (ordered.groupby('id_cliente').objetivo.cumsum()-ordered.objetivo).gt(0).any()
    return df

def aligned(df, month, pred):
    _,_,r,o=base.fold_indices(df,month)
    expected=base.prediction_frame(df,o,pred.prediccion.to_numpy(),r)
    pd.testing.assert_frame_equal(pred[KEYS],expected[KEYS])

def old_folder(policy,seed,month):
    if policy=='completo': return ROOT/'resultados_fase1/ajustes/cat_d3_l2_10'/str(seed)/str(month)
    return ROOT/'resultados_fase2/ajustes/ultimos6_clase_1/cat_d3_l2_10'/str(seed)/str(month)

def features(df):
    xs={p:base.build_features(df.drop(columns='objetivo'),p) for p in POLICIES}
    assert all(len(x.columns)==32 for x in xs.values())
    for m in MONTHS:
        for p in POLICIES:
            pd.testing.assert_frame_equal(xs[p].loc[df.mes.le(m)].reset_index(drop=True),base.build_features(df.loc[df.mes.le(m)].drop(columns='objetivo'),p))
    return xs

def old_inventory():
    paths=[ROOT/'baseline_datafest.py',ROOT/'baseline_datafest_1.0.py',ROOT/'baseline_datafest_2.0.py',ROOT/'train.csv',ROOT/'test.csv',ROOT/'sample_submission.csv']
    for directory in ['resultados_fase1','resultados_fase2','resultados_history','resultados_banking']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def verify_originals(args,df):
    checked=[]; errors=[]
    for phase in [1,2]:
        manifest=readj(ROOT/f'resultados_fase{phase}/configuracion.json')
        assert sha(ROOT/f'baseline_datafest_{phase}.0.py')==manifest['script_sha256']
        assert versions()=={k:manifest[k] for k in ['python','packages']}, 'Usar entorno original; no mezclar versiones.'
        digest=hashlib.sha256(df.loc[df.mes.le(202610)].to_csv(index=False).encode()).hexdigest()
        assert digest==manifest['data_jan_oct_sha256']
        assert manifest['quick'] is False and manifest['iterations']==2500 and manifest['patience']==150 and manifest['threads']==4
    for policy in POLICIES:
        recorded=pd.read_csv(ROOT/('resultados_fase1/metricas_mensuales.csv' if policy=='completo' else 'resultados_fase2/metricas_finalistas.csv'))
        recorded=recorded.loc[recorded.modelo.eq('cat_d3_l2_10')]
        for seed in SEEDS:
            for month in MONTHS[:-1]:
                folder=old_folder(policy,seed,month)
                pred=readpred(folder/'predicciones.csv'); meta=readj(folder/'ajuste.json')
                aligned(df,month,pred)
                a=np.array(meta['curves']['validation']['AUC'])
                assert meta['iterations']==int(np.argmax(a))+1
                for seg in ['todos','nuevo','conocido']:
                    p=pred if seg=='todos' else pred.loc[pred.segmento.eq(seg)]
                    row=recorded.loc[recorded.semilla.astype(str).eq(str(seed)) & recorded.mes.eq(month) & recorded.segmento.eq(seg)].iloc[0]
                    err=abs(score(p)-row.auc); errors.append(err)
                    assert err<1e-12 and len(p)==row.n and p.objetivo.sum()==row.positivos
                checked.append(dict(historial=policy,semilla=seed,mes=month,auc=score(pred),gini=2*score(pred)-1,iteraciones=meta['iterations'],pred_sha256=sha(folder/'predicciones.csv')))
    pd.DataFrame(checked).to_csv(args.output/'metricas_reproducidas.csv',index=False)
    return dict(folds_verificados=len(checked),max_error_auc=max(errors),versiones=versions(),fuente_candidata='resultados_fase2/ajustes/ultimos6_clase_1/cat_d3_l2_10',fecha=stamp())

def audit(args):
    if args.output.exists() and any(args.output.iterdir()): raise ValueError('audit requiere carpeta nueva; evaluate/finalize reanudan sin sobrescribir originales.')
    args.output.mkdir(parents=True)
    writej(args.output/'originales_sha256.json',old_inventory())
    df=load(); xs=features(df)
    report=verify_originals(args,df)
    # Reproducción exacta del incidente, sin cambiar regla ni consultar otros árboles externos.
    folder=args.output/'diagnostico_agosto'; folder.mkdir()
    pred,meta=base.run_fold(df,xs['ultimos6'],CONFIG,dict(id='ultimos6_clase_1',policy='ultimos6',class_mode='1'),2028,202608,args)
    previous=readpred(old_folder('ultimos6',2028,202608)/'predicciones.csv')
    np.testing.assert_allclose(pred.prediccion,previous.prediccion,atol=1e-14,rtol=0)
    assert meta['iterations']==55
    saved=readj(old_folder('ultimos6',2028,202608)/'ajuste.json')
    for split in ['learn','validation']:
        for metric in meta['curves'][split]: np.testing.assert_allclose(meta['curves'][split][metric],saved['curves'][split][metric],atol=1e-13,rtol=0)
    pred.to_csv(folder/'predicciones_reproducidas.csv',index=False); writej(folder/'ajuste_reproducido.json',meta)
    curves=[]
    for seed in SEEDS:
        m=readj(old_folder('ultimos6',seed,202608)/'ajuste.json')
        for i,a in enumerate(m['curves']['validation']['AUC']): curves.append(dict(semilla=seed,iteracion=i+1,auc_julio=a,logloss_julio=m['curves']['validation']['Logloss'][i],logloss_entrenamiento=m['curves']['learn']['Logloss'][i]))
    pd.DataFrame(curves).to_csv(folder/'curvas_internas.csv',index=False)
    report['agosto_2028']=dict(iteraciones=55,auc_externo=score(pred),auc_entrenamiento=meta['train_auc'],auc_interno_julio=meta['calibration_auc_unweighted'],max_error_prediccion=float(np.max(np.abs(pred.prediccion-previous.prediccion))),alineacion='verificada por id_cliente/mes/objetivo/segmento',diagnostico='Se reproduce el maximo interno a 55 y la caida tras refit. No hay evidencia de desalineacion ni off-by-one. La transferencia del numero de arboles entre muestras y semillas es inestable; no se identifica una causa unica del recorrido de boosting.')
    writej(args.output/'verificacion.json',report)
    protocol=dict(created_utc=stamp(),source_sha256=sha(__file__),base_sha256=sha(ROOT/'baseline_datafest_2.0.py'),data_sha256=sha(ROOT/'train.csv'),versions=versions(),months=list(MONTHS),seeds=list(SEEDS),features=xs['ultimos6'].columns.tolist(),parameters=dict(depth=3,l2_leaf_reg=10,learning_rate=.03,iterations_max=2500,patience=150,threads=4,class_weights='uniformes'),policies=list(POLICIES),window_scope='features_only; todas las filas anteriores',control='original: cada semilla usa su propio argmax AUC en M-1, entrenado con meses <M-1',alternative='mediana: mediana de los tres numeros originales (2026/2027/2028) seleccionados exclusivamente con M-1; numero comun para refit de las tres semillas con meses <M',seed_policy='Siempre promedio uniforme de las tres probabilidades; ninguna semilla se elimina o elige retrospectivamente.',iteration_adoption='Adoptar mediana solo si su promedio de AUC mensual del ensemble es >= control, disminuye la media de la desviacion entre semillas por mes y ningun mes pierde mas de 0.002 AUC; en otro caso conservar original. Regla operativa, no prueba de significancia.',history_adoption='Comparar ultimos6 menos completo bajo la regla de iteraciones adoptada: mantener seis meses solo si el limite inferior del IC bootstrap pareado 95% de la diferencia de AUC medio es >0; si no, completo por menor dependencia de una ventana elegida.',iteration_decision_scope='Para decidir regla de iteraciones usar ultimos6, la candidata inicial, meses junio-noviembre. Aplicar la misma regla adoptada a ambos historiales.',bootstrap=dict(replicates=args.bootstrap,seed=20261006,unit='cliente; mismas multiplicidades para todas sus filas, meses y modelos',metric='media no ponderada de AUC mensuales',interval='percentil 95%; condicional a predicciones fijas; sin ajuste por seleccion ni multiplicidad'),november='Ya consultado. Evaluacion retrospectiva, no ciega.',final='Seleccion de arboles con enero-octubre -> noviembre; refit enero-noviembre; predict_proba diciembre; promedio tres semillas.')
    writej(args.output/'protocolo_fijado.json',protocol)
    shutil.copy2(__file__,args.output/'codigo_fijado.py')
    shutil.copy2(ROOT/'baseline_datafest_2.0.py',args.output/'base_fase2_inmutable.py')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    print('PROTOCOLO FIJADO: mediana entre semillas; no se han calculado nuevas metricas externas de la alternativa.',flush=True)

def check_manifest(args):
    p=readj(args.output/'protocolo_fijado.json')
    assert p['source_sha256']==sha(__file__), 'Codigo cambio despues de fijar protocolo.'
    assert p['base_sha256']==sha(ROOT/'baseline_datafest_2.0.py')
    assert p['data_sha256']==sha(ROOT/'train.csv') and p['versions']==versions()
    assert p['bootstrap']['replicates']==args.bootstrap
    assert readj(args.output/'originales_sha256.json')==old_inventory(), 'Los originales cambiaron.'
    return p

def checkpoint(args,policy,rule,seed,month):
    folder=args.output/'ajustes'/policy/rule/str(seed)/str(month); folder.mkdir(parents=True,exist_ok=True); return folder

def refit(df,x,seed,month,rounds,args):
    _,_,r,o=base.fold_indices(df,month)
    m=base.cat_model(CONFIG,seed,rounds,args)
    m.fit(x.iloc[r],df.objetivo.iloc[r],cat_features=base.CATS)
    p=base.prediction_frame(df,o,m.predict_proba(x.iloc[o])[:,1],r)
    return p,dict(iterations=rounds,train_auc=base.auc(df.objetivo.iloc[r],m.predict_proba(x.iloc[r])[:,1]),seed=seed,selection_month=month-1,n_refit=len(r))

def evaluate(args):
    check_manifest(args); df=load(); xs=features(df); metrics=[]; predictions=[]
    for policy in POLICIES:
        for month in MONTHS:
            originals={}; metadata={}
            for seed in SEEDS:
                folder=checkpoint(args,policy,'original',seed,month)
                if (folder/'ajuste.json').exists():
                    pred=readpred(folder/'predicciones.csv'); meta=readj(folder/'ajuste.json')
                elif month<202611:
                    old=old_folder(policy,seed,month); pred=readpred(old/'predicciones.csv'); meta=readj(old/'ajuste.json')
                    meta.update(reused_from=str(old.relative_to(ROOT)),source_prediction_sha256=sha(old/'predicciones.csv'))
                else:
                    print(f'Ajustando original {policy} {month} semilla {seed}',flush=True)
                    pred,meta=base.run_fold(df,xs[policy],CONFIG,dict(id=policy+'_clase_1',policy=policy,class_mode='1'),seed,month,args)
                aligned(df,month,pred)
                pred.to_csv(folder/'predicciones.csv',index=False); writej(folder/'ajuste.json',meta)
                originals[seed]=pred; metadata[seed]=meta
            rounds=int(np.median([metadata[s]['iterations'] for s in SEEDS]))
            for rule in ['original','mediana']:
                members=[]
                for seed in SEEDS:
                    folder=checkpoint(args,policy,rule,seed,month)
                    if rule=='original': pred,meta=originals[seed],metadata[seed]
                    elif (folder/'ajuste.json').exists(): pred,meta=readpred(folder/'predicciones.csv'),readj(folder/'ajuste.json')
                    elif rounds==metadata[seed]['iterations']:
                        pred=originals[seed].copy(); meta=dict(metadata[seed],reused_original_same_iterations=True)
                    else:
                        print(f'Ajustando mediana {policy} {month} semilla {seed}: {rounds} arboles',flush=True)
                        pred,meta=refit(df,xs[policy],seed,month,rounds,args)
                    aligned(df,month,pred)
                    meta.update(rule=rule,history=policy,selection_month=month-1,original_rounds={str(s):metadata[s]['iterations'] for s in SEEDS})
                    pred.to_csv(folder/'predicciones.csv',index=False); writej(folder/'ajuste.json',meta)
                    members.append(pred.prediccion.to_numpy())
                    model_id=policy+'__'+rule
                    metrics.extend(base.segment_metrics(pred,model_id,'catboost',seed,meta))
                    predictions.append(pred.assign(modelo=model_id,semilla=str(seed)))
                ensemble=originals[2026].copy(); ensemble['prediccion']=np.mean(members,axis=0)
                metrics.extend(base.segment_metrics(ensemble,model_id,'ensemble','promedio',{}))
                predictions.append(ensemble.assign(modelo=model_id,semilla='promedio'))
                print(f'{policy} {rule} {month}: AUC promedio={score(ensemble):.8f}',flush=True)
            pd.DataFrame(metrics).to_csv(args.output/'metricas_mensuales.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True)
    assert not pred.duplicated(['modelo','semilla','id_cliente','mes']).any()
    pred.to_csv(args.output/'predicciones_validacion.csv',index=False)
    met=pd.DataFrame(metrics); summary=base.summarize(metrics); summary['gini_std_mensual']=2*summary.auc_std_mensual
    summary.to_csv(args.output/'resumen.csv',index=False)
    # Regla congelada; no selecciona semillas ni excluye meses desfavorables.
    focus=met.loc[met.segmento.eq('todos') & met.modelo.str.startswith('ultimos6')]
    means=focus.loc[focus.semilla.eq('promedio')].pivot(index='mes',columns='modelo',values='auc')
    dispersion=focus.loc[focus.semilla.ne('promedio')].groupby(['modelo','mes']).auc.std().groupby('modelo').mean()
    diff=means['ultimos6__mediana']-means['ultimos6__original']
    adopt=diff.mean()>=0 and dispersion['ultimos6__mediana']<dispersion['ultimos6__original'] and diff.min()>=-.002
    rule='mediana' if adopt else 'original'
    bootstrap=paired_bootstrap(pred,args)
    key=f'ultimos6__{rule} - completo__{rule}'
    hist=bootstrap.loc[bootstrap.comparacion.eq(key)&bootstrap.segmento.eq('todos')&bootstrap.periodo.eq('junio_noviembre')].iloc[0]
    history='ultimos6' if hist.ic95_auc_inferior>0 else 'completo'
    decision=dict(fecha=stamp(),regla_iteraciones=rule,historial=history,semillas=list(SEEDS),prediccion='promedio uniforme de tres predict_proba',diferencia_auc_mediana_vs_original=float(diff.mean()),peor_diferencia_mensual=float(diff.min()),dispersion_semillas_original=float(dispersion['ultimos6__original']),dispersion_semillas_mediana=float(dispersion['ultimos6__mediana']),comparacion_historial=hist.to_dict(),criterio='Aplicacion literal de protocolo_fijado.json; evaluacion retrospectiva con seleccion, noviembre no ciego.',estado='validacion_completa; final diciembre pendiente')
    writej(args.output/'configuracion_adoptada.json',decision)
    check_manifest(args)
    print(json.dumps(decision,ensure_ascii=False),flush=True)

def prepare_auc(y,p,codes):
    order=np.argsort(p,kind='stable'); vals=p[order]
    starts=np.r_[0,np.flatnonzero(vals[1:]!=vals[:-1])+1]
    return y[order],codes[order],starts

def weighted_auc(prepared,counts):
    y,codes,starts=prepared; w=counts[codes]
    pos=np.add.reduceat(w*y,starts); neg=np.add.reduceat(w*(1-y),starts)
    den=pos.sum()*neg.sum()
    return float(np.sum(pos*(np.cumsum(neg)-neg+neg*.5))/den) if den else np.nan

def paired_bootstrap(pred,args):
    # Una sola multiplicidad por cliente/repeticion compartida por TODOS los contrastes.
    ens=pred.loc[pred.semilla.eq('promedio')].copy()
    ids=np.sort(ens.id_cliente.unique()); model_ids=sorted(ens.modelo.unique())
    prepared={}; basepoints={}
    for model in model_ids:
        for month in MONTHS:
            for segment in ['todos','nuevo','conocido']:
                p=ens.loc[ens.modelo.eq(model)&ens.mes.eq(month)]
                if segment!='todos': p=p.loc[p.segmento.eq(segment)]
                codes=np.searchsorted(ids,p.id_cliente.to_numpy()); y=p.objetivo.to_numpy(); pr=p.prediccion.to_numpy()
                prepared[(model,month,segment)]=prepare_auc(y,pr,codes)
                basepoints[(model,month,segment)]=score(p)
    keys=list(prepared); reps=np.empty((args.bootstrap,len(keys)))
    rng=np.random.default_rng(20261006)
    for b in range(args.bootstrap):
        counts=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids))
        reps[b]=[weighted_auc(prepared[k],counts) for k in keys]
        if b%250==0: print(f'Bootstrap por cliente {b}/{args.bootstrap}',flush=True)
    indices={k:i for i,k in enumerate(keys)}
    contrasts=[('ultimos6__mediana','ultimos6__original'),('completo__mediana','completo__original'),('ultimos6__original','completo__original'),('ultimos6__mediana','completo__mediana')]
    rows=[]
    periods={'junio_noviembre':MONTHS,'junio_octubre':MONTHS[:-1],**{str(m):(m,) for m in MONTHS}}
    for a,b in contrasts:
        for seg in ['todos','nuevo','conocido']:
            for label,months in periods.items():
                ai=[indices[(a,m,seg)] for m in months]; bi=[indices[(b,m,seg)] for m in months]
                diffs=reps[:,ai].mean(axis=1)-reps[:,bi].mean(axis=1)
                lo,hi=np.nanquantile(diffs,[.025,.975]); point=np.mean([basepoints[(a,m,seg)]-basepoints[(b,m,seg)] for m in months])
                rows.append(dict(comparacion=a+' - '+b,segmento=seg,periodo=label,delta_auc=float(point),delta_gini=float(2*point),ic95_auc_inferior=float(lo),ic95_auc_superior=float(hi),ic95_gini_inferior=float(2*lo),ic95_gini_superior=float(2*hi),replicas_validas=int(np.isfinite(diffs).sum()),clientes_universo=len(ids)))
    result=pd.DataFrame(rows); result.to_csv(args.output/'bootstrap_pareado.csv',index=False)
    np.savez_compressed(args.output/'bootstrap_replicas.npz',auc=reps,keys=np.array(['|'.join(map(str,k)) for k in keys]))
    return result

def finalize(args):
    check_manifest(args); choice=readj(args.output/'configuracion_adoptada.json'); df=load()
    test=pd.read_csv(ROOT/'test.csv'); sample=pd.read_csv(ROOT/'sample_submission.csv')
    assert set(test.mes)=={202612} and not test.id_cliente.duplicated().any() and not test.isna().any().any()
    assert set(test.columns)==set(df.columns)-{'objetivo'}
    assert sample.id_cliente.equals(test.id_cliente)
    final=args.output/'diciembre'; final.mkdir(exist_ok=True)
    if (final/'manifiesto.json').exists():
        manifest=readj(final/'manifiesto.json')
        assert manifest['choice_sha256']==sha(args.output/'configuracion_adoptada.json')
        for name,digest in manifest['files'].items(): assert sha(final/name)==digest
        print('Diciembre ya completado y hashes verificados.',flush=True); return
    policy=choice['historial']; x=base.build_features(df.drop(columns='objetivo'),policy)
    combined=pd.concat([df.drop(columns='objetivo'),test],ignore_index=True)
    allx=base.build_features(combined,policy)
    pd.testing.assert_frame_equal(allx.iloc[:len(df)].reset_index(drop=True),x)
    xt=allx.iloc[len(df):].reset_index(drop=True)
    inner=np.flatnonzero(df.mes.lt(202611)); stop=np.flatnonzero(df.mes.eq(202611)); selected={}
    for seed in SEEDS:
        path=final/f'seleccion_{seed}.json'
        if path.exists(): meta=readj(path)
        else:
            print(f'Diciembre: seleccion con noviembre, semilla {seed}',flush=True)
            tuner=base.cat_model(CONFIG,seed,args.iterations,args)
            tuner.fit(x.iloc[inner],df.objetivo.iloc[inner],cat_features=base.CATS,eval_set=(x.iloc[stop],df.objetivo.iloc[stop]),early_stopping_rounds=args.patience,use_best_model=True)
            meta=dict(iterations=int(tuner.tree_count_),selection_month=202611,train_until=202610,curves=tuner.get_evals_result()); writej(path,meta)
        selected[seed]=meta['iterations']
    rounds={s:(int(np.median(list(selected.values()))) if choice['regla_iteraciones']=='mediana' else selected[s]) for s in SEEDS}
    predictions=[]
    for seed in SEEDS:
        print(f'Diciembre: refit enero-noviembre, semilla {seed}, {rounds[seed]} arboles',flush=True)
        m=base.cat_model(CONFIG,seed,rounds[seed],args); m.fit(x,df.objetivo,cat_features=base.CATS)
        pp=m.predict_proba(xt)[:,1]; predictions.append(pp); m.save_model(str(final/f'modelo_{seed}.cbm'))
        pd.DataFrame(dict(id_cliente=test.id_cliente,mes=test.mes,prediccion=pp)).to_csv(final/f'predicciones_{seed}.csv',index=False)
        writej(final/f'parametros_{seed}.json',m.get_all_params())
    pred=np.mean(predictions,axis=0); assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    submission=pd.DataFrame(dict(id_cliente=sample.id_cliente,prediccion=pred)); submission.to_csv(final/'submission_fase2_validada.csv',index=False)
    writej(final/'manifiesto.json',dict(created_utc=stamp(),choice_sha256=sha(args.output/'configuracion_adoptada.json'),history=policy,rule=choice['regla_iteraciones'],selected_original=selected,refit_rounds=rounds,features=x.columns.tolist(),n_training=len(df),n_prediction=len(test),test_sha256=sha(ROOT/'test.csv'),versions=versions(),files={p.name:sha(p) for p in final.iterdir() if p.is_file()},notas='Sin etiquetas diciembre, sin metricas diciembre, no enviado a plataforma. Noviembre usado para seleccionar iteraciones.'))
    check_manifest(args)
    print('Diciembre generado con la candidata validada; no enviado.',flush=True)

def self_test():
    y=np.array([0,1,0,1,1]); p=np.array([.2,.3,.3,.7,.7]); w=np.array([2,1,3,2,1])
    assert abs(weighted_auc(prepare_auc(y,p,np.arange(5)),w)-roc_auc_score(y,p,sample_weight=w))<1e-15
    df=load().head(100)
    x=base.build_features(df.drop(columns='objetivo'),'ultimos6'); other=df.copy(); other.objetivo=1-other.objetivo
    pd.testing.assert_frame_equal(x,base.build_features(other,'ultimos6'))
    assert int(np.median([422,500,55]))==422
    print('Tests OK: AUC ponderado con empates, independencia del objetivo, mediana.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','evaluate','finalize','self-test'])
    parser.add_argument('--output',type=Path,default=ROOT/'resultados_fase2_validacion')
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args(); args.output=args.output.resolve()
    args.iterations=2500; args.patience=150; args.threads=4; args.half_life=3
    if args.bootstrap<100: parser.error('Minimo 100 replicas; predeterminado 2000.')
    if args.stage=='self-test': self_test()
    else: globals()[args.stage](args)

if __name__=='__main__': main()
