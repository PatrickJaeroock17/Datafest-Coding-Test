"""Comprobaciones de causalidad y artefactos; no entrena modelos.
python test_baseline_datafest_2_1.py [--completed]
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('validation',ROOT/'baseline_datafest_2.1.py')
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--completed',action='store_true'); args=parser.parse_args()
    df=v.load(); xs=v.features(df)
    spec1=importlib.util.spec_from_file_location('phase1',ROOT/'baseline_datafest_1.0.py')
    f1=importlib.util.module_from_spec(spec1); spec1.loader.exec_module(f1)
    pd.testing.assert_frame_equal(xs['completo'],f1.history_features(df.drop(columns='objetivo')))
    for m in v.MONTHS:
        inner,stop,refit,outer=v.base.fold_indices(df,m)
        assert df.iloc[inner].mes.max()<m-1
        assert df.iloc[stop].mes.eq(m-1).all()
        assert df.iloc[refit].mes.max()<m
        assert len(refit)==int(df.mes.lt(m).sum())
        for p in v.POLICIES:
            changed=df.copy(); changed.loc[changed.mes.ge(m),'objetivo']=1-changed.loc[changed.mes.ge(m),'objetivo']
            pd.testing.assert_frame_equal(v.base.build_features(changed,p),xs[p])
    rng=np.random.default_rng(3)
    for _ in range(20):
        y=rng.integers(0,2,100); pred=rng.integers(0,8,100)/8; weight=rng.integers(1,5,100)
        value=v.weighted_auc(v.prepare_auc(y,pred,np.arange(100)),weight)
        assert abs(value-roc_auc_score(y,pred,sample_weight=weight))<1e-14
    print('OK: equivalencia Fase1/full, exclusividad temporal, objetivo no entra en features, ventanas no eliminan filas y bootstrap con empates.')
    if not args.completed: return
    out=ROOT/'resultados_fase2_validacion'; pred=v.readpred(out/'predicciones_validacion.csv')
    met=pd.read_csv(out/'metricas_mensuales.csv')
    assert len(pred)==int(df.mes.ge(202606).sum())*16
    assert len(met)==4*4*6*3
    assert not pred.duplicated(['modelo','semilla','id_cliente','mes']).any()
    for (model,month),group in pred.groupby(['modelo','mes']):
        components=[]
        for seed in (*map(str,v.SEEDS),'promedio'):
            part=group.loc[group.semilla.eq(seed)].reset_index(drop=True)
            v.aligned(df,month,part)
            if seed!='promedio': components.append(part.prediccion.to_numpy())
            else: np.testing.assert_allclose(part.prediccion,np.mean(components,axis=0),atol=2e-15,rtol=0)
            for seg in ['todos','nuevo','conocido']:
                sub=part if seg=='todos' else part.loc[part.segmento.eq(seg)]
                saved=met.loc[met.modelo.eq(model)&met.semilla.eq(seed)&met.mes.eq(month)&met.segmento.eq(seg)].iloc[0]
                assert abs(v.score(sub)-saved.auc)<1e-14
                assert abs(2*saved.auc-1-saved.gini)<1e-14
                assert len(sub)==saved.n and int(sub.objetivo.sum())==saved.positivos
        policy,rule=model.split('__')
        original=[v.readj(out/'ajustes'/policy/'original'/str(s)/str(month)/'ajuste.json')['iterations'] for s in v.SEEDS]
        for seed in v.SEEDS:
            meta=v.readj(out/'ajustes'/policy/rule/str(seed)/str(month)/'ajuste.json')
            assert meta['selection_month']==month-1
            expected=int(np.median(original)) if rule=='mediana' else original[v.SEEDS.index(seed)]
            assert meta['iterations']==expected
    assert v.old_inventory()==v.readj(out/'originales_sha256.json')
    # Se verifica el artefacto final y el promedio guardado, si ya fue generado.
    final=out/'diciembre'
    if (final/'manifiesto.json').exists():
        sub=v.readpred(final/'submission_fase2_validada.csv'); sample=pd.read_csv(ROOT/'sample_submission.csv')
        assert sub.columns.tolist()==['id_cliente','prediccion'] and sub.id_cliente.equals(sample.id_cliente)
        assert sub.prediccion.between(0,1).all() and sub.prediccion.notna().all()
        parts=[v.readpred(final/f'predicciones_{s}.csv') for s in v.SEEDS]
        for p in parts: assert p.id_cliente.equals(sub.id_cliente)
        np.testing.assert_allclose(sub.prediccion,np.mean([p.prediccion for p in parts],axis=0),atol=2e-15,rtol=0)
        from catboost import CatBoostClassifier
        choice=v.readj(out/'configuracion_adoptada.json'); test=pd.read_csv(ROOT/'test.csv')
        xt=v.base.build_features(pd.concat([df.drop(columns='objetivo'),test],ignore_index=True),choice['historial']).iloc[len(df):]
        xtrain=v.base.build_features(df.drop(columns='objetivo'),choice['historial'])
        final_checks=[]
        for seed,p in zip(v.SEEDS,parts):
            model=CatBoostClassifier(); model.load_model(str(final/f'modelo_{seed}.cbm'))
            np.testing.assert_allclose(model.predict_proba(xt)[:,1],p.prediccion,atol=2e-15,rtol=0)
            final_checks.append(dict(semilla=seed,iteraciones=int(model.tree_count_),
                auc_entrenamiento=float(roc_auc_score(df.objetivo,model.predict_proba(xtrain)[:,1])),
                media_probabilidad_diciembre=float(p.prediccion.mean()),
                minimo_probabilidad_diciembre=float(p.prediccion.min()),maximo_probabilidad_diciembre=float(p.prediccion.max())))
        v.writej(out/'verificacion_modelos_finales.json',dict(modelos=final_checks,
            nota='AUC de entrenamiento es diagnostico; no mide generalizacion ni autoriza cambiar receta. Diciembre sin etiquetas.'))
        print('OK: submission nuevo, orden IDs, probabilidad, promedio y recarga de los tres modelos.')
    print('OK: 288 métricas, 24 ensembles, iteraciones causales, predicciones alineadas y originales intactos.')

if __name__=='__main__': main()
