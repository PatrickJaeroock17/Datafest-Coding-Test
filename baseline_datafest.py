"""CatBoost + LightGBM con validacion mensual y noviembre reservado.

Ejemplo: python baseline_datafest.py --features banking --output resultados_banking
Dependencias: numpy, pandas, scikit-learn, catboost, lightgbm.
No usa objetivo, id_cliente ni informacion futura como caracteristicas.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

CATS = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
ROOT = Path(__file__).resolve().parent


def features(frame, mode='raw', keep_interaction=False):
    """Historial exclusivamente anterior a cada fila, sin usar etiquetas.

    Se permite concatenar train/test porque shift/cumcount nunca consultan
    el futuro. Las ventanas son observaciones previas, no meses garantizados.
    """
    z = frame.drop(columns='objetivo', errors='ignore').copy()
    z['_order'] = np.arange(len(z))
    z = z.sort_values(['id_cliente', 'mes'])
    if mode in ['banking', 'history']:
        z['saldo_ingreso_mensual'] = z.saldo_promedio / (z.ingresos / 12).replace(0, np.nan)
        z['saldo_por_producto'] = z.saldo_promedio / z.numero_productos.replace(0, np.nan)
        z['visitas_por_producto'] = z.visitas_web_ultimos_90_dias / z.numero_productos.replace(0, np.nan)
        flags = ['tiene_tarjeta_credito', 'tiene_prestamo', 'tiene_seguro']
        z['productos_flags'] = z[flags].astype(int).sum(axis=1)
        z['movil_x_visitas'] = z.activo_movil.astype(int) * z.visitas_web_ultimos_90_dias
        for c in ['ingresos', 'saldo_promedio', 'dias_ultima_transaccion']:
            z['log_' + c] = np.log1p(z[c].clip(lower=0))
    if mode == 'history':
        z['_month_index'] = (z.mes // 100) * 12 + z.mes % 100
        g = z.groupby('id_cliente', sort=False)
        z['observaciones_previas'] = g.cumcount()
        z['meses_desde_observacion_previa'] = z._month_index - g._month_index.shift(1)
        z['meses_desde_primera_observacion'] = z._month_index - g._month_index.transform('first')
        if keep_interaction:
            z['interaccion_previa'] = g.dias_ultima_interaccion.shift(1)
            z['interaccion_media_3_previas'] = g.dias_ultima_interaccion.transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            z['cambio_interaccion'] = z.dias_ultima_interaccion - z.interaccion_previa
        z = z.drop(columns='_month_index')
    z = z.sort_values('_order').drop(columns=['id_cliente', 'mes', '_order'])
    if not keep_interaction:
        z = z.drop(columns='dias_ultima_interaccion')
    for c in CATS:
        z[c] = z[c].astype(str).fillna('__MISSING__')
    for c in z.select_dtypes(include='bool'):
        z[c] = z[c].astype('int8')
    z = z.replace([np.inf, -np.inf], np.nan)
    nums = [c for c in z if c not in CATS]
    # CatBoost requiere valores numericos finitos; -1 marca historia inexistente.
    z[nums] = z[nums].fillna(-1)
    return z.reset_index(drop=True)


def fit_predict(kind, x, y, dest, rounds, valid=None):
    if kind == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=rounds, depth=5, learning_rate=0.03,
                                   l2_leaf_reg=8, loss_function='Logloss', eval_metric='AUC',
                                   random_seed=2026, thread_count=4,
                                   allow_writing_files=False, verbose=False)
        kwargs = {} if valid is None else dict(eval_set=valid, early_stopping_rounds=150)
        model.fit(x, y, cat_features=CATS, **kwargs)
        return model.predict_proba(dest)[:, 1], model.tree_count_
    import lightgbm as lgb
    x, dest = x.copy(), dest.copy()
    v = None if valid is None else valid[0].copy()
    for c in CATS:
        dtype = pd.CategoricalDtype(categories=x[c].unique())
        x[c], dest[c] = x[c].astype(dtype), dest[c].astype(dtype)
        if v is not None:
            v[c] = v[c].astype(dtype)
    model = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.03,
                              num_leaves=15, max_depth=-1, min_child_samples=150,
                              colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
                              reg_lambda=8, random_state=2026, n_jobs=4, verbosity=-1)
    kwargs = {} if valid is None else dict(eval_set=[(v, valid[1])], eval_metric='auc',
                                          callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    # Desactiva la metrica por defecto para que early stopping use solo AUC.
    model.set_params(metric='auc')
    model.fit(x, y, categorical_feature=CATS, **kwargs)
    return model.predict_proba(dest)[:, 1], model.best_iteration_ or rounds


def main():
    from sklearn.metrics import roc_auc_score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', choices=['raw', 'banking', 'history'], default='raw')
    parser.add_argument('--keep-interaction', action='store_true')
    parser.add_argument('--finalize', action='store_true', help='Evaluar noviembre y generar submission tras congelar la configuracion')
    parser.add_argument('--output', type=Path, default=ROOT / 'resultados_baseline')
    args = parser.parse_args()
    train, test = [pd.read_csv(ROOT / f'{n}.csv') for n in ['train', 'test']]
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    assert train.objetivo.isin([0, 1]).all()
    assert not train.duplicated(['id_cliente', 'mes']).any()
    assert not test.duplicated(['id_cliente', 'mes']).any()
    assert test.mes.eq(202612).all() and train.mes.max() == 202611
    assert sample.columns.tolist() == ['id_cliente', 'prediccion']
    assert sample.id_cliente.equals(test.id_cliente)
    combined = pd.concat([train.drop(columns='objetivo'), test], ignore_index=True)
    x = features(combined, args.features, args.keep_interaction)
    xt, x_test, y = x.iloc[:len(train)], x.iloc[len(train):], train.objetivo
    models = ['catboost', 'lightgbm']
    iterations = {k: [] for k in models}
    folds = []
    oof = []
    # Seleccion exclusivamente agosto-octubre. Noviembre NO determina el blend.
    for month in [202608, 202609, 202610]:
        tr, va = train.mes.lt(month), train.mes.eq(month)
        fold = train.loc[va, ['id_cliente', 'mes', 'objetivo']].copy()
        for kind in models:
            pred, count = fit_predict(kind, xt.loc[tr], y.loc[tr], xt.loc[va],
                                      2500, valid=(xt.loc[va], y.loc[va]))
            fold[kind] = pred
            iterations[kind].append(int(count))
            auc = roc_auc_score(y.loc[va], pred)
            folds.append(dict(month=month, model=kind, auc=auc, gini=2*auc-1, iterations=int(count)))
        oof.append(fold)
        print(f'Mes {month}: {folds[-2:]}', flush=True)
    dev = pd.concat(oof)
    blend_scores = []
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        monthly = [roc_auc_score(g.objetivo, w*g.catboost+(1-w)*g.lightgbm)
                   for _, g in dev.groupby('mes')]
        blend_scores.append(dict(catboost_weight=w, mean_auc=float(np.mean(monthly)), monthly_auc=monthly))
    # Comparacion por media de AUC mensuales, no AUC agrupado de meses distintos.
    best = max(blend_scores, key=lambda r: r['mean_auc'])
    weight = best['catboost_weight']
    fixed_rounds = {k: max(1, int(np.median(v))) for k,v in iterations.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    dev.to_csv(args.output / 'predicciones_desarrollo.csv', index=False)
    development = dict(config=dict(features=args.features, keep_interaction=args.keep_interaction),
                       folds=folds, blend_candidates=blend_scores, chosen_weight=weight,
                       final_iterations=fixed_rounds)
    (args.output / 'metricas_desarrollo.json').write_text(json.dumps(development, indent=2), encoding='utf-8')
    print('Mejor blend agosto-octubre:', best, flush=True)
    if not args.finalize:
        print('Noviembre reservado. Usar --finalize cuando la configuracion este congelada.')
        return
    tr, va = train.mes.lt(202611), train.mes.eq(202611)
    holdout = train.loc[va, ['id_cliente', 'mes', 'objetivo']].copy()
    for kind in models:
        holdout[kind], _ = fit_predict(kind, xt.loc[tr], y.loc[tr], xt.loc[va], fixed_rounds[kind])
    holdout['blend'] = weight*holdout.catboost+(1-weight)*holdout.lightgbm
    holdout_auc = {k: float(roc_auc_score(holdout.objetivo, holdout[k])) for k in models+['blend']}
    print('Noviembre reservado:', holdout_auc, flush=True)
    # Congelar configuracion antes de consultar noviembre. Repetir decisiones
    # usando este resultado convierte noviembre en otro conjunto de ajuste.
    final = {}
    for kind in models:
        final[kind], _ = fit_predict(kind, xt, y, x_test, fixed_rounds[kind])
    probabilities = weight*final['catboost']+(1-weight)*final['lightgbm']
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    submission = pd.DataFrame({'id_cliente': test.id_cliente, 'prediccion': probabilities})
    assert len(submission) == len(sample) and submission.id_cliente.equals(sample.id_cliente)
    args.output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output / 'submission.csv', index=False)
    dev.to_csv(args.output / 'predicciones_desarrollo.csv', index=False)
    holdout.to_csv(args.output / 'predicciones_noviembre.csv', index=False)
    report = dict(config=dict(features=args.features, keep_interaction=args.keep_interaction),
                  folds=folds, blend_candidates=blend_scores, chosen_weight=weight,
                  final_iterations=fixed_rounds, november_auc=holdout_auc,
                  november_gini={k: 2*v-1 for k,v in holdout_auc.items()})
    (args.output / 'metricas.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'CSV generado: {args.output / "submission.csv"}', flush=True)


if __name__ == '__main__':
    main()
